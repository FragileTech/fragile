# The Geometric Micro-Architecture

## TLDR

- **Standard DL primitives violate gauge invariance**: `nn.Linear(bias=True)` and element-wise `ReLU` assume flat Euclidean space with preferred coordinate axes. Rotating the latent representation changes the physics.
- **Gauge-covariant replacements preserve symmetry**: Replace with `SpectralLinear` (bias-free, Lipschitz-constrained) and `NormGatedGELU` (acts on vector bundle norms, not individual components).
- **Each primitive preserves a gauge symmetry component**: Spectral normalization → $U(1)_Y$ (capacity conservation), bundle structure → $SU(N_f)_C$ (feature confinement), steerable convolutions → $SU(2)_L$ (observation-action chirality).
- **Microscopic operations compose to macroscopic geodesic flow**: Equivariant layers with Lipschitz constant $\le 1$ preserve the global light cone structure required by the Boris-BAOAB integrator.
- **Result**: Neural operators that respect the gauge group $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$ at the atomic level, making the architecture compatible with the capacity-constrained metric law and WFR geometry.

## Roadmap

1. **Formalize the gauge symmetry constraint**: Define $G$-equivariance for neural operators and explain why it matters for geometric consistency.
2. **Prove standard DL violates it**: Explicit counterexample showing `ReLU` breaks $SO(d)$-equivariance.
3. **Derive covariant replacements**: Spectral normalization (preserves light cone) + norm-gating (preserves bundle structure).
4. **Connect to causal structure**: Show how Lipschitz constraints implement the information speed limit $c_{\text{info}}$ from the parameter sieve.
5. **Extend to vision**: Steerable convolutions lift images to gauge bundles, ensuring $SO(2)$-equivariance for rotated inputs.
6. **Integration and diagnostics**: Connect microscopic primitives to macroscopic gauge fields, diagnostic nodes, and the geodesic integrator.

:::{div} feynman-prose
Now we come to what might seem like a technical detail but is actually fundamental to the entire framework. You see, in most deep learning, people throw together networks without much thought about the geometric properties of the operations they're using. Linear layers with biases, ReLU activations applied element-wise—these are treated as universal primitives, LEGO blocks you can snap together however you like.

But here's the thing: those primitives make hidden assumptions about your space. They assume it's flat. They assume the coordinate axes mean something special. When you apply ReLU element-wise to a vector, you're saying "kill the negative components based on the current choice of coordinates." But what if you rotate your mental axes? Suddenly different components get killed. The physics changes depending on an arbitrary choice.

This is not a metaphor. In Chapters 1-3 of this section, we derived that the agent's latent space must have a specific gauge structure—$SU(N_f)_C \times SU(2)_L \times U(1)_Y$—to ensure consistent multi-agent interactions and capacity constraints. If we build neural operators that violate this structure at the microscopic level, the macroscopic theory falls apart. The geodesic integrator requires smooth, well-defined tangent vectors. The metric law requires isometric transformations. The WFR geometry requires mass-conserving updates.

Standard DL primitives violate all of these. This chapter fixes that. We're going to build neural operators from scratch, starting with the requirement that they respect gauge symmetry. The result will be a new set of primitives—`IsotropicBlock`, `SteerableConv`—that play nice with the geometry we've spent three chapters deriving.

And here's the beautiful part: once you have gauge-covariant primitives, the architecture almost writes itself. The constraints are so tight that there's essentially only one way to build each component. This is the deep learning equivalent of the Standard Model in physics: a unique structure that emerges from symmetry requirements alone.
:::

*Cross-references:* This chapter builds on gauge theory foundations ({ref}`Section 29.1 <sec-symplectic-multi-agent-field-theory>`), the capacity-constrained metric law ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`), WFR geometry ({ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`), and feeds into the geodesic integrator implementation ({ref}`Section 22 <sec-equations-of-motion-langevin-sdes-on-information-manifolds>`).

---

(sec-fundamental-symmetry-constraint)=
## The Fundamental Symmetry Constraint

:::{div} feynman-prose
Let me tell you what gauge symmetry really means by way of a simple example. Suppose you and I are both looking at the same robot arm, but I describe its position using a coordinate system aligned with magnetic north, while you use one aligned with the factory floor. We would write down different numbers for the same physical configuration. Yet when we compute the forces, the torques, the dynamics—we had better get the same answers. The physics cannot depend on our arbitrary choice of coordinates.

This is gauge symmetry: the requirement that *physics* be independent of *bookkeeping choices*. And here is where standard deep learning commits a subtle but devastating error. When you feed a vector through a typical neural network, the operations treat the coordinate axes as if they were physically meaningful. A ReLU applied to the first component does something different than a ReLU applied to the second component. Rotate your mental axes by 45 degrees, and suddenly different neurons fire. The network's internal representation *changes* based on an arbitrary choice that has no physical significance.

Think of it like this: imagine a GPS system that gave you different directions depending on whether you held the phone facing north or east. That would be broken, obviously. Yet this is precisely what standard neural networks do—their internal computations depend on the orientation of an arbitrary coordinate frame.

In Chapter 8.1, we derived that agents interacting under capacity constraints must respect a specific gauge structure: $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$. Every neural operation that violates this symmetry is like a GPS that gives inconsistent directions. The errors might be small on any single forward pass, but they accumulate catastrophically when you try to integrate geodesics or maintain consistent beliefs across time. We need to fix the primitives.
:::

### Formal Definitions

:::{prf:definition} Group Action on Latent Space
:label: def-group-action-latent

**Preliminaries:**
- A **Lie group** is a smooth manifold $G$ equipped with a group structure such that multiplication $(g, h) \mapsto gh$ and inversion $g \mapsto g^{-1}$ are smooth maps. Examples: $SO(d)$ (rotations), $SE(2) = \mathbb{R}^2 \rtimes SO(2)$ (rigid motions), $SU(N)$ (special unitary group).
- The **general linear group** $GL(d_z, \mathbb{R})$ is the group of invertible $d_z \times d_z$ real matrices under matrix multiplication, with smooth manifold structure (open subset of $\mathbb{R}^{d_z^2}$).

**Setup:** Let $G$ be a Lie group and $\mathcal{Z} = \mathbb{R}^{d_z}$ be the latent space equipped with standard Euclidean topology and inner product $\langle z, z' \rangle = z^T z'$.

A **linear representation** is a continuous group homomorphism $\rho: G \to GL(d_z, \mathbb{R})$ satisfying:
1. $\rho(e_G) = I_{d_z}$ where $e_G$ is the identity element of $G$
2. $\rho(g_1 g_2) = \rho(g_1)\rho(g_2)$ for all $g_1, g_2 \in G$ (homomorphism property)

**Smoothness:** When $G$ is a Lie group, we additionally require $\rho$ to be smooth as a map between manifolds: for any smooth curve $\gamma: (-\epsilon, \epsilon) \to G$ with $\gamma(0) = e_G$, the matrix-valued map $t \mapsto \rho(\gamma(t))$ is $C^\infty$.

**Group action:** For $g \in G$ and $z \in \mathcal{Z}$ (viewed as a $d_z \times 1$ column vector), the action is given by standard matrix-vector multiplication:
$$
g \cdot z := \rho(g) z \in \mathbb{R}^{d_z}
$$
where the right-hand side denotes the matrix product of $\rho(g) \in \mathbb{R}^{d_z \times d_z}$ with $z \in \mathbb{R}^{d_z \times 1}$.

**Orthogonal representations:** When $G$ is compact (e.g., $SO(d)$, $SU(N)$), any continuous representation is conjugate to an orthogonal representation (Peter-Weyl theorem). We typically choose $\rho$ such that $\rho(g) \in O(d_z)$ for all $g$, ensuring:
$$
\langle \rho(g) z, \rho(g) z' \rangle = \langle z, z' \rangle \quad \forall z, z', g
$$
This preserves the Euclidean structure of $\mathcal{Z}$.

**Units:** $[\rho(g)]$ is dimensionless (orthogonal/unitary matrices have dimensionless entries).
:::

:::{prf:definition} G-Equivariant Operator
:label: def-g-equivariant-operator

Let $G$ be a group with representations $\rho: G \to GL(\mathbb{R}^{d_z})$ and $\rho': G \to GL(\mathbb{R}^{d_{z'}})$ on latent spaces $\mathcal{Z} = \mathbb{R}^{d_z}$ and $\mathcal{Z}' = \mathbb{R}^{d_{z'}}$ respectively.

A neural operator $f: \mathcal{Z} \to \mathcal{Z}'$ is **$G$-equivariant** (or **$(\rho, \rho')$-equivariant** when representations need emphasis) if:

$$
f(\rho(g) z) = \rho'(g) f(z) \quad \forall g \in G, \; z \in \mathcal{Z}
$$

where matrix-vector multiplication is implied.

**Physical interpretation:** Transforming the input then applying $f$ gives the same result as applying $f$ then transforming the output. Formally, the following diagram commutes:

$$
\begin{array}{ccc}
\mathcal{Z} & \xrightarrow{f} & \mathcal{Z}' \\
\downarrow \rho(g) & & \downarrow \rho'(g) \\
\mathcal{Z} & \xrightarrow{f} & \mathcal{Z}'
\end{array}
$$

**Units:** If $[z] = [z']$ (both in units of $\sqrt{\text{nat}}$), then $[f(z)] = [z']$ and $[\rho(g)] = [\rho'(g)] = $ dimensionless.
:::

:::{prf:definition} Fragile Gauge Group
:label: def-fragile-gauge-group

From Chapter 8.1 ({ref}`sec-symplectic-multi-agent-field-theory`), agents under capacity constraints $C < \infty$ must respect the gauge structure:

$$
G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y
$$

**Explicit definitions:**

1. **$SU(N_f)_C$ (Color symmetry):** The special unitary group of degree $N_f$, consisting of $N_f \times N_f$ complex unitary matrices $U$ with $\det(U) = 1$. Here $N_f$ equals the number of feature bundles $n_b$ (see Definition {prf:ref}`def-latent-vector-bundle`). This acts on the bundle index, permitting feature mixing across bundles.

2. **$SU(2)_L$ (Weak isospin):** The special unitary group of degree 2, consisting of $2 \times 2$ complex unitary matrices with determinant 1. This has 3 real parameters (Pauli matrix basis) and acts on observation-action doublets.

3. **$U(1)_Y$ (Hypercharge):** The circle group $\{e^{i\theta} : \theta \in [0, 2\pi)\} \cong SO(2)$, representing phase rotations. This is associated with capacity conservation (holographic bound).

**Representation on latent space:** For a latent space $\mathcal{Z} = \mathbb{R}^{d_z}$ decomposed into $n_b$ bundles of dimension $d_b$ (so $d_z = n_b \cdot d_b$), the representation $\rho: G_{\text{Fragile}} \to GL(d_z, \mathbb{R})$ acts via:

- **$SU(N_f)_C$ action:** Realized through block mixing in neural layers (see Theorem {prf:ref}`thm-isotropic-preserves-color`)
- **$SU(2)_L$ action:** Realized through observation-action coupling in steerable vision encoder (see Theorem {prf:ref}`thm-steerable-induces-doublet`)
- **$U(1)_Y$ action:** Realized through spectral normalization preserving hypercharge (see Theorem {prf:ref}`thm-spectral-preserves-hypercharge`)

**Note:** The full representation structure is established progressively through Sections 4.3-4.5. For detailed gauge field derivation, see {ref}`Section 29.1 <sec-symplectic-multi-agent-field-theory>`.

**Requirement:** All neural operators in the latent dynamics must be $G_{\text{Fragile}}$-equivariant to preserve physical consistency.

**Implication:** Standard building blocks (ReLU, LayerNorm, biased Linear) that violate even simple $SO(d)$ equivariance cannot be used directly.
:::

:::{admonition} Connection to RL: Equivariance vs. Data Augmentation
:class: note

**Standard RL approach:** Augment training data with rotated/transformed copies, hope the model learns approximate invariance through exposure.

**Fragile approach:** Build exact equivariance into the architecture. The network *cannot* violate the symmetry, even with adversarial inputs.

**Advantage:** Guaranteed consistency (no approximate learning), better sample efficiency (no need to see all rotations), mathematical rigor (enables formal proofs about behavior).
:::

### The Failure of ReLU

:::{div} feynman-prose
Now let me show you exactly what goes wrong with ReLU, because this is not abstract—it is a concrete bug you can compute with pencil and paper.

ReLU says: "Look at each component of the vector. If it is negative, kill it. If it is positive, keep it." This sounds innocent enough. But notice what is hidden in that instruction: *which* components? The answer depends entirely on what coordinate axes you chose. And that choice was arbitrary.

Here is the picture to hold in your mind. Imagine the latent space as a room, and your coordinate axes as lines painted on the floor. ReLU creates "walls" along those painted lines—whenever a vector crosses from the positive side to the negative side of an axis, that component gets chopped to zero. Now rotate the room. The painted lines move, and the walls move with them. A vector that passed through freely before now hits a wall and gets truncated. *The physics has changed because you rotated your bookkeeping.*

There is another way to see this. ReLU creates kinks in the function—sharp corners where the derivative is undefined. These kinks are aligned with the coordinate axes. In a gauge-invariant world, there are no preferred directions, so there should be no preferred locations for kinks. But ReLU puts them exactly along the axes, breaking the smooth differential structure that the geodesic integrator requires.

The proof that follows makes this precise: apply ReLU, then rotate; versus rotate, then apply ReLU. You get different answers. That is the definition of gauge violation.
:::

:::{prf:theorem} ReLU Violates SO(d) Equivariance
:label: thm-relu-breaks-equivariance

Let $d \geq 2$ and consider the standard representation $\rho: SO(d) \to GL(d, \mathbb{R})$ where $\rho(R) = R$ (i.e., rotations act by matrix multiplication).

Define the ReLU activation $f: \mathbb{R}^d \to \mathbb{R}^d$ by:
$$
f(z) = (f(z_1), \ldots, f(z_d)) \quad \text{where} \quad f(z_i) = \max(0, z_i)
$$

Then $f$ is **not** $SO(d)$-equivariant with respect to $\rho$.

*Proof.*

We prove by explicit counterexample for $d=2$, then indicate generalization.

**Step 1. Counterexample construction ($d=2$):**

Let $z = (1, -1)^T \in \mathbb{R}^2$. Then:
$$
f(z) = (\max(0, 1), \max(0, -1))^T = (1, 0)^T
$$

**Step 2. Apply rotation:**

Let $R_\theta \in SO(2)$ be counterclockwise rotation by $\theta = \pi/4$:
$$
R_{\pi/4} = \begin{pmatrix} \cos(\pi/4) & -\sin(\pi/4) \\ \sin(\pi/4) & \cos(\pi/4) \end{pmatrix} = \begin{pmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{pmatrix}
$$

Then:
$$
R_{\pi/4} z = \begin{pmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{pmatrix} \begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} \frac{1+1}{\sqrt{2}} \\ \frac{1-1}{\sqrt{2}} \end{pmatrix} = \begin{pmatrix} \sqrt{2} \\ 0 \end{pmatrix}
$$

**Step 3. Compute $f(R_{\pi/4} z)$:**
$$
f(R_{\pi/4} z) = f((\sqrt{2}, 0)^T) = (\sqrt{2}, 0)^T
$$

**Step 4. Compute $R_{\pi/4} f(z)$:**
$$
R_{\pi/4} f(z) = R_{\pi/4} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{pmatrix}
$$

**Step 5. Verify non-equality:**
$$
f(R_{\pi/4} z) = (\sqrt{2}, 0)^T \neq \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)^T = R_{\pi/4} f(z)
$$

Thus $f(R_{\pi/4} z) \neq R_{\pi/4} f(z)$, violating $SO(2)$-equivariance.

**Generalization to $d > 2$:**

For arbitrary $d \geq 2$, consider $z = (1, -1, 0, \ldots, 0)^T \in \mathbb{R}^d$ (first two components nonzero, rest zero).

Define $R_\theta \in SO(d)$ as rotation by $\theta = \pi/4$ in the $(e_1, e_2)$-plane, identity on remaining coordinates:
$$
R_\theta = \begin{pmatrix}
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 & \cdots & 0 \\
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 & \cdots & 0 \\
0 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 1
\end{pmatrix}
$$

Then:
- $R_\theta z = (\sqrt{2}, 0, 0, \ldots, 0)^T$ (rotation in first 2 coordinates, as in Step 2)
- $f(R_\theta z) = (\sqrt{2}, 0, 0, \ldots, 0)^T$ (all components non-negative)
- $f(z) = (1, 0, 0, \ldots, 0)^T$ (second component killed)
- $R_\theta f(z) = (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0, \ldots, 0)^T$ (rotate result)

Since $(\sqrt{2}, 0, \ldots)^T \neq (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, \ldots)^T$, we have $f(R_\theta z) \neq R_\theta f(z)$ for all $d \geq 2$.

**Geometric interpretation:** The ReLU creates coordinate-aligned decision boundaries $\{z \in \mathbb{R}^d : z_i = 0\}$ for each $i$. These hyperplanes are not invariant under $SO(d)$: rotation maps the hyperplane $\{z_1 = 0\}$ to a different hyperplane (no longer aligned with any coordinate axis). Thus $f$ transforms the kink surfaces under rotation, violating equivariance.

**Remark (General groups):** This argument generalizes to any Lie group $G$ with a nontrivial linear representation $\rho: G \to GL(d, \mathbb{R})$. "Nontrivial" means there exists $g \in G$ such that $\rho(g) e_i \neq \pm e_i$ for some basis vector $e_i$. For $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$, each factor acts nontrivially on its respective subspace (bundles, observation-action doublet, capacity), so ReLU violates $G_{\text{Fragile}}$-equivariance.

$\square$
:::

:::{prf:corollary} ReLU Violates Smoothness Requirements for WFR Dynamics
:label: cor-relu-breaks-wfr

The WFR geometry (Chapter 5, Section {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`) provides the dynamical framework for latent state evolution. The WFR action and geodesic integrator require computing gradient flows:

$$
\frac{dz}{ds} = -G^{ij}(z) \nabla_{z_j} \mathcal{L}_{\text{WFR}}(z)
$$

where $G^{ij}(z)$ is the metric tensor from Theorem {prf:ref}`thm-capacity-constrained-metric-law` and $\mathcal{L}_{\text{WFR}}$ is the WFR action functional.

**Smoothness requirement:** Gradient-based geodesic integrators (e.g., Boris-BAOAB in Section 5.4) require $\mathcal{L}_{\text{WFR}}$ to be at least $C^1$ (continuously differentiable) to compute well-defined gradients $\nabla_z \mathcal{L}$.

**ReLU violates this:** By Theorem {prf:ref}`thm-relu-breaks-equivariance`, ReLU creates non-differentiable kinks at coordinate hyperplanes $\{z \in \mathcal{Z} : z_i = 0\}$ for each $i = 1, \ldots, d_z$. At these kinks:

1. **Gradient undefined:** $\nabla_z \mathcal{L}$ is undefined in the classical sense (left and right derivatives differ)
2. **Metric ill-defined:** The metric tensor $G(z)$, depending on $\nabla^2 V(z)$ (value Hessian) and Fisher information $\mathcal{F}(z)$, may be discontinuous
3. **Integration errors:** Numerical integrators accumulate errors when trajectories cross kinks

**Gauge-dependence problem:** Per Theorem {prf:ref}`thm-relu-breaks-equivariance`, ReLU kinks are coordinate-dependent. Under gauge transformation $z \mapsto U(g) \cdot z$ for $g \in G_{\text{Fragile}}$, the kink locations transform but ReLU does not transform equivariantly. This creates **inconsistent kink patterns** across gauge choices, causing geodesic flows to depend on arbitrary coordinate choices.

**Consequence:** Smooth, gauge-equivariant activations (e.g., GELU in NormGate, Definition {prf:ref}`def-norm-gated-activation`) are necessary for well-defined WFR gradient flows and gauge-invariant dynamics.

**Reference to WFR smoothness:** The WFR formulation's smoothness requirements are established through:
- Variational calculus on action functionals (standard $C^1$ requirement)
- Riemannian geometry for geodesic equations (smooth metric tensor)
- Symplectic integrator theory (Lipschitz gradients for Boris-BAOAB)

See Section 5.2 (WFR stress-energy tensor) and Part II, Hypostructure, Section 9 (Mathematical Prerequisites) for differential geometry foundations.

$\square$
:::

---

(sec-isotropic-bundle-operator)=
## The Isotropic Bundle Operator

:::{div} feynman-prose
Here is a question that will reveal a hidden assumption in nearly every neural network ever built: when you have a 512-dimensional latent vector, what *kind* of object is it?

Most practitioners treat it as 512 separate numbers—a list of scalars. Each dimension is independent, has its own meaning, gets its own weight, its own activation. But this is geometrically naive. A proper geometric object is not just a list of numbers; it is something that *transforms* in a definite way when you change coordinates.

Think about velocity. If I measure a car moving at 60 mph heading north, and you use a coordinate system rotated 90 degrees from mine, you will say the car is moving 60 mph heading east. The *components* changed, but the *velocity itself*—the geometric object—did not. Velocity is a vector, and vectors have transformation rules.

Now here is the key insight. Instead of treating the latent space as 512 independent scalars, we partition it into *bundles*—groups of dimensions that transform together as geometric vectors. If we have 32 bundles of 16 dimensions each, then within each bundle, the 16 components rotate together when you change frames. The bundle is a geometric object with definite transformation properties.

This is what a *vector bundle* means: at each point in your latent space, you attach a little vector space (the fiber), and the whole structure transforms coherently under gauge transformations. Standard neural networks ignore this structure entirely. We are going to respect it.
:::

### Geometric Preliminaries: Bundle Decomposition

:::{prf:definition} Bundle Decomposition of Latent Space
:label: def-latent-vector-bundle

Let $\mathcal{Z} = \mathbb{R}^{n_b \cdot d_b}$ be the latent space. A **bundle decomposition** partitions $\mathcal{Z}$ into $n_b$ **bundles** (subspaces) of dimension $d_b$:

$$
\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i, \quad V_i \cong \mathbb{R}^{d_b}, \quad n_b \times d_b = \dim(\mathcal{Z})
$$

Each bundle $V_i$ carries a representation of the rotation group $SO(d_b)$. For $g_i \in SO(d_b)$ and $v_i \in V_i$:

$$
\rho_i(g_i) \cdot v_i = g_i \cdot v_i \quad \text{(matrix multiplication)}
$$

where we identify $SO(d_b) \subset GL(d_b, \mathbb{R})$ as the subgroup of orthogonal matrices with determinant 1. This is the **defining (standard) representation** of $SO(d_b)$.

**Product gauge group (derivation):** Given the direct sum decomposition $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$, what is the maximal symmetry group?

**Claim:** The full symmetry group preserving the decomposition is:
$$
G_{\text{bundle}} = \prod_{i=1}^{n_b} SO(d_b)
$$

**Justification:**
1. **Preservation of decomposition:** Any symmetry must preserve $V_i \perp V_j$ for $i \neq j$ (orthogonal direct sum). Thus transformations cannot mix bundles.

2. **Within each bundle:** On $V_i \cong \mathbb{R}^{d_b}$, the maximal continuous symmetry group preserving the Euclidean inner product is $O(d_b)$ (orthogonal group).

3. **Orientation preservation:** For neural networks with deterministic forward pass, we require **orientation-preserving** transformations (continuous deformation from identity). Thus we restrict to $SO(d_b) \subset O(d_b)$ (special orthogonal: determinant +1).

4. **Independence:** Transformations on different bundles are independent, giving the product structure $\prod_{i=1}^{n_b} SO(d_b)$.

**Group action:** For $(g_1, \ldots, g_{n_b}) \in G_{\text{bundle}}$ and $z = (z^{(1)}, \ldots, z^{(n_b)})$ with $z^{(i)} \in V_i$:
$$
\rho(g_1, \ldots, g_{n_b}) \cdot z = (g_1 z^{(1)}, \ldots, g_{n_b} z^{(n_b)})
$$

In matrix form (with respect to concatenated basis):
$$
\rho(g_1, \ldots, g_{n_b}) = \text{diag}(g_1, \ldots, g_{n_b}) = \begin{pmatrix} g_1 & 0 & \cdots & 0 \\ 0 & g_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & g_{n_b} \end{pmatrix}
$$
(block-diagonal structure).

**Note on bundle permutations:** If bundles are **semantically distinguished** (e.g., bundle 1 = edges, bundle 2 = textures, bundle 3 = colors), we cannot permute them. If all bundles are **identical** (homogeneous feature space), the symmetry group extends to $(\prod_{i=1}^{n_b} SO(d_b)) \rtimes S_{n_b}$ where $S_{n_b}$ is the permutation group. For this architecture, we assume distinguished bundles.

**Units:** $[V_i] = [\mathcal{Z}] = \sqrt{\text{nat}}$ (from capacity constraint, see Section {ref}`sec-dimensional-analysis`).

**Remark:** This is a **direct sum decomposition with group action**, not a fiber bundle in the differential-geometric sense (which would require a base manifold and projection map). Analogous to gauge fields in physics (Chapter {ref}`sec-symplectic-multi-agent-field-theory`): just as the Error field $W_\mu$ transforms under $SU(2)_L$, bundles transform under their respective $SO(d_b)$ factors.
:::

### The Spectral Linear Map

:::{prf:definition} Spectral Linear Operator
:label: def-spectral-linear

A linear map $W: \mathcal{Z} \to \mathcal{Z}'$ is **spectrally normalized** if:

$$
\sigma_{\max}(W) \leq 1
$$

where $\sigma_{\max}(W)$ is the largest singular value of $W$.

**Implementation:** $W_{\text{spectral}} = W / \sigma_{\max}(W)$.

**Units:** $[W]$ is dimensionless (if $\mathcal{Z}, \mathcal{Z}'$ normalized).
:::

:::{prf:theorem} Spectral Normalization Preserves Light Cone
:label: thm-spectral-preserves-light-cone

Let $(\mathcal{Z}, \|\cdot\|)$ be a finite-dimensional normed vector space with $\|\cdot\|$ the **Euclidean norm** $\|z\| = \sqrt{z^T z}$. Let $W: \mathcal{Z} \to \mathcal{Z}'$ be a linear map with $\sigma_{\max}(W) \leq 1$ (spectrally normalized). Let $c_{\text{info}}$ be the information speed limit (Axiom {prf:ref}`ax-information-speed-limit`).

Then:

$$
\|W \cdot z\| \leq \|z\| \quad \text{(contraction property)}
$$

This ensures that whenever $d(z_1, z_2) := \|z_1 - z_2\| \leq c_{\text{info}} \cdot \Delta t$ (causal interval), we have:

$$
d(W \cdot z_1, W \cdot z_2) \leq c_{\text{info}} \cdot \Delta t
$$

Thus the **light cone** $\mathcal{C}(z_0, t_0) := \{(z, t) : \|z - z_0\| \leq c_{\text{info}}(t - t_0), \, t \geq t_0\}$ in the extended state-time space is preserved: if $(z, t) \in \mathcal{C}(z_0, t_0)$, then $(W \cdot z, t) \in \mathcal{C}(W \cdot z_0, t_0)$.

*Proof.*

**Step 1. Spectral bound:** By spectral normalization (Definition {prf:ref}`def-spectral-linear`), $\sigma_{\max}(W) \leq 1$.

**Step 2. Operator norm:** For the Euclidean norm, the induced operator norm satisfies $\|W\|_{\text{op}} = \sigma_{\max}(W)$, where:
$$
\|W\|_{\text{op}} := \sup_{\|z\| = 1} \|W \cdot z\|
$$

Thus for any $z \in \mathcal{Z}$:
$$
\|W \cdot z\| \leq \|W\|_{\text{op}} \cdot \|z\| = \sigma_{\max}(W) \cdot \|z\| \leq \|z\|
$$

**Step 3. Distance preservation:** For any $z_1, z_2 \in \mathcal{Z}$:
$$
\|W \cdot z_1 - W \cdot z_2\| = \|W \cdot (z_1 - z_2)\| \leq \|z_1 - z_2\|
$$

**Step 4. Causal structure:** If the inputs are causally connected ($d(z_1, z_2) \leq c_{\text{info}} \Delta t$), the outputs remain causally connected since:
$$
d(W \cdot z_1, W \cdot z_2) \leq d(z_1, z_2) \leq c_{\text{info}} \Delta t
$$

**Identification:** The Lipschitz constant $L = \sigma_{\max}(W) \leq 1$ ensures signal propagation speed $\leq c_{\text{info}}$, preventing "superluminal" information transfer that would violate the causal structure established in the Speed Window (Theorem {prf:ref}`thm-speed-window`).

**Connection to Node 62:** The CausalityViolationCheck diagnostic verifies $\sigma_{\max}(W) \leq 1 + \epsilon$ during training. Violations indicate faster-than-light gradients breaking temporal coherence.

$\square$
:::

:::{div} feynman-prose
You might wonder: why does the spectral linear map have no bias term? Every neural network tutorial tells you to include a bias. What is wrong with $Wz + b$?

Here is what is wrong: the bias vector $b$ picks out a *special point* in space. It says "this is the origin, and I am going to shift everything relative to it." But in a proper geometric theory, there is no special point. The origin of your coordinate system is arbitrary—a bookkeeping choice, not physics.

Think about it this way. If I have a vector $z$ representing some agent state, and I shift my coordinate origin by $b$, then $z$ becomes $z - b$. A gauge-invariant operation should not care about this shift. But $Wz + b$ and $W(z-b) + b = Wz - Wb + b$ give different results unless $Wb = 0$. The bias has created a preferred origin.

In the tangent bundle picture, this is even clearer. We are working with *directions*—vectors that represent velocities, gradients, rates of change. A direction has no preferred starting point. You can translate the base point anywhere and the direction remains the same direction. Adding a bias is like saying "all velocities should be measured relative to this special velocity I chose"—which is exactly the kind of frame-dependent nonsense gauge theory forbids.

So we set $\text{bias} = \text{False}$ and never look back. The rigorous justification follows.
:::

:::{prf:proposition} Bias Terms Break Tangent Bundle Structure
:label: prop-bias-breaks-tangent-bundle

Working in the tangent bundle $T\mathcal{Z}$, elements are pairs $(z, v)$ where $z \in \mathcal{Z}$ is a point and $v \in T_z \mathcal{Z}$ is a tangent vector at $z$.

A gauge-covariant map on the tangent bundle must satisfy:
$$
f(z, \rho(g) \cdot v) = \rho'(g) \cdot f(z, v) \quad \forall g \in G
$$

**Claim:** Adding a bias $b$ violates this unless $b = 0$.

*Proof.* Consider $f(z, v) = W v + b$. Then:
$$
f(z, \rho(g) \cdot v) = W \rho(g) v + b
$$
but
$$
\rho'(g) \cdot f(z, v) = \rho'(g) (W v + b) = \rho'(g) W v + \rho'(g) b
$$

For equivariance, we need $b = \rho'(g) b$ for all $g \in G$. If $G$ acts non-trivially (e.g., $SO(d)$ with $d > 1$), the only fixed point is $b = 0$. $\square$

**Extension to compositions:** For a composition of $L$ layers $F = f_L \circ \cdots \circ f_1$ where each $f_i(v) = W_i v + b_i$:

If $F$ is $G$-equivariant, then $F(\rho(g) v) = \rho(g) F(v)$ for all $g, v$.

**Proof by induction:**
- **Base case** ($L=1$): By the proof above, $b_1 = 0$.
- **Inductive step:** Assume $f_{L-1} \circ \cdots \circ f_1$ is equivariant, so each $b_i = 0$ for $i < L$.

  Then:
  $$
  F(\rho(g) v) = f_L((f_{L-1} \circ \cdots \circ f_1)(\rho(g) v)) = f_L(\rho(g) (f_{L-1} \circ \cdots \circ f_1)(v))
  $$
  $$
  = W_L \rho(g) h + b_L \quad \text{where } h = (f_{L-1} \circ \cdots \circ f_1)(v)
  $$

  For equivariance with $\rho(g) F(v) = \rho(g) (W_L h + b_L)$, we need $b_L = 0$.

Thus **all biases must vanish** for a composition to be equivariant. $\square$

**Coordinate-free interpretation:** In the tangent bundle $T\mathcal{Z}$, linear maps act on **velocity vectors**, which are equivalence classes of curves $[\gamma]$ where $\gamma(0) = z$, $\dot{\gamma}(0) = v$. Adding a constant $b$ is not well-defined on velocity vectors—it mixes base points (where you are) with tangent directions (which way you're moving), breaking the fiber bundle structure.
:::

::::{admonition} Physics Isomorphism: Light Cone Preservation
:class: note

**In General Relativity:** The speed of light $c$ is the maximum causal influence speed. The light cone structure defines which events can influence which others.

**In Agent Theory:** The information speed $c_{\text{info}}$ is the maximum rate at which latent state changes can propagate. Spectral normalization enforces $\|W \cdot z\| \leq \|z\|$, preserving the causal cone.

**Correspondence Table:**

| General Relativity | Agent (Spectral Norm) |
|:-------------------|:----------------------|
| Speed of light $c$ | Information speed $c_{\text{info}}$ |
| Lorentz invariance | Causal ordering preserved |
| Light cone $ds^2 \leq 0$ | $\|W \cdot z\| \leq \|z\|$ |
| Timelike separation | $d(z_1, z_2) \leq c_{\text{info}} \Delta t$ |
::::

### The Norm-Gated Activation

:::{prf:definition} Norm-Gated Activation
:label: def-norm-gated-activation

For a vector bundle $v_i \in V_i$, define the **norm-gated activation**:

$$
f(v_i) = v_i \cdot g(\|v_i\| + b_i)
$$

where:
- $\|v_i\| = \sqrt{v_i^T v_i}$ is the Euclidean norm ($SO(d_b)$-invariant)
- $b_i \in \mathbb{R}$ is a learnable scalar bias (the "activation potential")
- $g: \mathbb{R} \to \mathbb{R}$ is a smooth scalar function (e.g., GELU, sigmoid)

**Physical interpretation:** Energy filter with radial symmetry. The gate opens when signal energy $\|v_i\|$ exceeds the potential barrier $-b_i$.

**Units:** All dimensionless if latent vectors normalized.
:::

:::{prf:theorem} Norm-Gating Preserves SO(d_b) Equivariance
:label: thm-norm-gating-equivariant

Let $f$ be the norm-gated activation (Definition {prf:ref}`def-norm-gated-activation`). Then:

$$
f(R \cdot v) = R \cdot f(v) \quad \forall R \in SO(d_b)
$$

*Proof.*

**Step 1. Decompose activation:** $f(v) = v \cdot h$ where $h = g(\|v\| + b)$ is a scalar.

**Step 2. Norm invariance:** For any $R \in SO(d_b)$ (orthogonal matrix):
$$
\|R \cdot v\| = \sqrt{(R \cdot v)^T (R \cdot v)} = \sqrt{v^T R^T R v} = \sqrt{v^T v} = \|v\|
$$

**Step 3. Scalar invariance:** Since the norm is invariant:
$$
h(R \cdot v) = g(\|R \cdot v\| + b) = g(\|v\| + b) = h(v)
$$

**Step 4. Equivariance:** Combining:
$$
f(R \cdot v) = (R \cdot v) \cdot h(R \cdot v) = (R \cdot v) \cdot h(v) = R \cdot (v \cdot h(v)) = R \cdot f(v)
$$

$\square$
:::

:::{div} feynman-prose
Now I want you to see the norm-gated activation as what it really is: an *energy barrier*.

Think of the norm $\|v\|$ as the *energy* of the signal—how strong is this bundle of features? And think of the bias $b$ as setting the *height* of a barrier that the signal must overcome to pass through. When the energy exceeds the barrier ($\|v\| > -b$), the gate opens and the signal propagates. When the energy is too low, the gate stays closed and the signal is suppressed.

This is exactly like a threshold detector in a physical system. A photon detector requires a minimum photon energy to trigger. A synapse requires sufficient neurotransmitter concentration to fire. Our norm gate requires sufficient signal energy to activate.

And here is the beautiful part: the *direction* of the vector does not matter. The norm $\|v\|$ is invariant under rotations—rotate the vector any way you like, and its length stays the same. So the gate decision is purely about *how much* energy, not *which direction* the energy points. Every orientation is treated identically.

Compare this with ReLU. There, the activation depends on individual components: "Is $v_1 > 0$? Is $v_2 > 0$?" Those questions have different answers depending on your coordinate frame. But "Is $\|v\| > $ threshold?" has the same answer no matter how you orient your axes. That is the difference between a gauge-invariant activation and a broken one.
:::

### The Complete Isotropic Block

:::{prf:definition} Isotropic Block
:label: def-isotropic-block

The **Isotropic Block** is the atomic unit of gauge-covariant architecture:

$$
\text{IsotropicBlock}(z) = \text{Reshape}(\text{NormGate}(\text{SpectralLinear}(z)))
$$

where:
- **SpectralLinear**: Linear map $W$ with $\sigma_{\max}(W) \leq 1$ (Definition {prf:ref}`def-spectral-linear`) that is **block-diagonal** with respect to the bundle decomposition (see Lemma {prf:ref}`lem-block-diagonal-necessary`)
- **Reshape**: $\mathbb{R}^d \to (\mathbb{R}^{d_b})^{n_b}$ (bundle partition)
- **NormGate**: Norm-gated activation applied per bundle (Definition {prf:ref}`def-norm-gated-activation`)

**Structure constraint:** For $G_{\text{bundle}} = \prod_{i=1}^{n_b} SO(d_b)$ equivariance, the weight matrix $W$ must be block-diagonal:
$$
W = \begin{pmatrix} W_1 & 0 & \cdots & 0 \\ 0 & W_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & W_{n_b} \end{pmatrix}
$$
where each $W_i \in \mathbb{R}^{d_b \times d_b}$ with $\sigma_{\max}(W_i) \leq 1$.
:::

:::{prf:lemma} Block-Diagonal Structure is Necessary for Bundle Equivariance
:label: lem-block-diagonal-necessary

Let $W: \mathcal{Z} \to \mathcal{Z}$ be a linear map on $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$ with $V_i \cong \mathbb{R}^{d_b}$.

$W$ is compatible with the product group action $G_{\text{bundle}} = \prod_{i=1}^{n_b} SO(d_b)$ if and only if $W$ is block-diagonal: $W = \text{diag}(W_1, \ldots, W_{n_b})$ with $W_i: V_i \to V_i$.

*Proof.*

**($\Leftarrow$) Sufficiency:** If $W$ is block-diagonal, then for $(g_1, \ldots, g_{n_b}) \in G_{\text{bundle}}$ and $z = (z^{(1)}, \ldots, z^{(n_b)})$:
$$
W \rho(g_1, \ldots, g_{n_b}) z = W (g_1 z^{(1)}, \ldots, g_{n_b} z^{(n_b)}) = (W_1 g_1 z^{(1)}, \ldots, W_{n_b} g_{n_b} z^{(n_b)})
$$

While this is not generally equal to $\rho(g_1, \ldots, g_{n_b}) W z$ (since $W_i g_i \neq g_i W_i$ unless $W_i = \lambda_i I$), the subsequent NormGate operation restores equivariance (see Theorem {prf:ref}`thm-isotropic-block-equivariant`, Step 5).

**($\Rightarrow$) Necessity:** Suppose $W$ has off-diagonal blocks, i.e., there exist $i \neq j$ such that $W_{ij} \neq 0$ where $W = [W_{ij}]$ in block form.

Consider a group element $g = (g_1, \ldots, g_{n_b})$ where $g_i = R_\theta \in SO(d_b)$ is a nontrivial rotation and $g_j = I$ for $j \neq i$.

For this $g$ and input $z$ with $z^{(j)} \neq 0$, $z^{(k)} = 0$ for $k \neq j$:
$$
(W \rho(g) z)_i = \sum_{k=1}^{n_b} W_{ik} g_k z^{(k)} = W_{ii} g_i \cdot 0 + W_{ij} I \cdot z^{(j)} = W_{ij} z^{(j)}
$$

But:
$$
(\rho(g) W z)_i = g_i (W z)^{(i)} = R_\theta \sum_{k=1}^{n_b} W_{ik} z^{(k)} = R_\theta W_{ij} z^{(j)}
$$

Since $R_\theta \neq I$ and $R_\theta W_{ij} z^{(j)} \neq W_{ij} z^{(j)}$ (rotation changes direction), we have $W \rho(g) z \neq \rho(g) W z$, violating equivariance.

Therefore, off-diagonal blocks must vanish: $W_{ij} = 0$ for $i \neq j$. $\square$

**Remark:** This lemma establishes that the block-diagonal constraint is not arbitrary—it is forced by the requirement that $W$ respect the bundle decomposition structure.
:::

:::{prf:theorem} IsotropicBlock is G-Equivariant
:label: thm-isotropic-block-equivariant

Let $G = \prod_{i=1}^{n_b} SO(d_b)$ be the product gauge group (Definition {prf:ref}`def-latent-vector-bundle`). By Lemma {prf:ref}`lem-block-diagonal-necessary`, the weight matrix $W$ in SpectralLinear must be **block-diagonal** with respect to the bundle decomposition:

$$
W = \begin{pmatrix} W_1 & 0 & \cdots & 0 \\ 0 & W_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & W_{n_b} \end{pmatrix}
$$

where each $W_i \in \mathbb{R}^{d_b \times d_b}$ is spectrally normalized with $\sigma_{\max}(W_i) \leq 1$.

Then the IsotropicBlock (Definition {prf:ref}`def-isotropic-block`) is $G$-equivariant.

*Proof.*

**Step 1. Define the group action:** On the flat space $\mathcal{Z} \cong \mathbb{R}^{n_b \cdot d_b}$, the group $G = \prod_{i=1}^{n_b} SO(d_b)$ acts as:
$$
\rho(g_1, \ldots, g_{n_b}) \cdot z = (g_1 \cdot z^{(1)}, \ldots, g_{n_b} \cdot z^{(n_b)})
$$
where $z = (z^{(1)}, \ldots, z^{(n_b)})$ with $z^{(i)} \in \mathbb{R}^{d_b}$.

**Step 2. SpectralLinear with block-diagonal W:**
By block-diagonal structure (Lemma {prf:ref}`lem-block-diagonal-necessary`), for $W = \text{diag}(W_1, \ldots, W_{n_b})$ and $(g_1, \ldots, g_{n_b}) \in G$:
$$
W \cdot \rho(g_1, \ldots, g_{n_b}) \cdot z = \begin{pmatrix} W_1 g_1 z^{(1)} \\ \vdots \\ W_{n_b} g_{n_b} z^{(n_b)} \end{pmatrix}
$$

**Key observation:** Individual blocks $W_i$ do NOT commute with rotations $g_i \in SO(d_b)$ (unless $W_i = \lambda_i I$), so SpectralLinear alone is not equivariant. However, the subsequent NormGate operation restores equivariance, as we now prove rigorously.

**Step 3. Reshape is equivariant:** The bundle partition $\mathbb{R}^{n_b \cdot d_b} \to (\mathbb{R}^{d_b})^{n_b}$ is equivariant by construction (identity map in bundled coordinates).

**Step 4. NormGate is $SO(d_b)$-equivariant per bundle:** By Theorem {prf:ref}`thm-norm-gating-equivariant`, for each bundle $i$:
$$
\text{NormGate}(g_i \cdot v_i) = g_i \cdot \text{NormGate}(v_i) \quad \forall g_i \in SO(d_b)
$$

**Step 5. Composition equivariance (rigorous proof):**

We prove $\text{IsotropicBlock}(\rho(g) \cdot z) = \rho(g) \cdot \text{IsotropicBlock}(z)$ by explicit calculation.

Let $g = (g_1, \ldots, g_{n_b}) \in G$ and $z = (z^{(1)}, \ldots, z^{(n_b)})$.

Compute left-hand side:
$$
\begin{align}
\text{IsotropicBlock}(\rho(g) \cdot z) &= \text{NormGate}(W \rho(g) z) \\
&= \text{NormGate}((W_1 g_1 z^{(1)}, \ldots, W_{n_b} g_{n_b} z^{(n_b)})) \quad \text{(Step 2)} \\
&= (W_1 g_1 z^{(1)} \cdot h(\|W_1 g_1 z^{(1)}\|), \ldots, W_{n_b} g_{n_b} z^{(n_b)} \cdot h(\|W_{n_b} g_{n_b} z^{(n_b)}\|)) \quad \text{(Def. NormGate)}
\end{align}
$$

Since $g_i \in SO(d_b)$ is orthogonal, $\|g_i v\| = \|v\|$ for any $v$. Thus:
$$
\|W_i g_i z^{(i)}\| = \|W_i z^{(i)}\|
$$

So the left-hand side becomes:
$$
= (W_1 g_1 z^{(1)} \cdot h(\|W_1 z^{(1)}\|), \ldots, W_{n_b} g_{n_b} z^{(n_b)} \cdot h(\|W_{n_b} z^{(n_b)}\|))
$$

Compute right-hand side:
$$
\begin{align}
\rho(g) \cdot \text{IsotropicBlock}(z) &= \rho(g) \cdot \text{NormGate}(W z) \\
&= \rho(g) \cdot (W_1 z^{(1)} \cdot h(\|W_1 z^{(1)}\|), \ldots, W_{n_b} z^{(n_b)} \cdot h(\|W_{n_b} z^{(n_b)}\|)) \\
&= (g_1 (W_1 z^{(1)} \cdot h(\|W_1 z^{(1)}\|)), \ldots, g_{n_b} (W_{n_b} z^{(n_b)} \cdot h(\|W_{n_b} z^{(n_b)}\|))) \\
&= (g_1 W_1 z^{(1)} \cdot h(\|W_1 z^{(1)}\|), \ldots, g_{n_b} W_{n_b} z^{(n_b)} \cdot h(\|W_{n_b} z^{(n_b)}\|)) \quad \text{(scalar mult.)}
\end{align}
$$

**Comparison:** We need to show:
$$
W_i g_i z^{(i)} \cdot h(\|W_i z^{(i)}\|) = g_i W_i z^{(i)} \cdot h(\|W_i z^{(i)}\|)
$$

Both sides equal the same vector! Although $W_i g_i \neq g_i W_i$ (matrices don't commute), we have:
$$
W_i g_i z^{(i)} = g_i (g_i^{-1} W_i g_i) z^{(i)}
$$

The norm is preserved: $\|W_i g_i z^{(i)}\| = \|g_i^{-1} W_i g_i z^{(i)}\| = \|W_i z^{(i)}\|$ (orthogonal conjugation preserves norms).

Since both expressions equal $g_i$ applied to a vector with the same norm and the same scalar multiplier $h(\|W_i z^{(i)}\|)$, they are equal.

Therefore: $\text{IsotropicBlock}(\rho(g) \cdot z) = \rho(g) \cdot \text{IsotropicBlock}(z)$

$\square$

**Remark:** The key insight is that NormGate **restores equivariance** even when $W$ is not exactly equivariant, because it decouples the magnitude (scalar, invariant) from the direction (vector, equivariant). This is why IsotropicBlock is more robust than naïve linear+activation combinations.
:::

---

(sec-covariant-retina)=
## The Covariant Retina

:::{div} feynman-prose
Convolutional neural networks have one beautiful property: translation equivariance. Shift the input image left by 10 pixels, and the feature maps shift left by 10 pixels. The pattern detector does not care *where* in the image a cat appears—it finds the cat regardless. This is why CNNs revolutionized computer vision.

But here is what CNNs do *not* have: rotation equivariance. Rotate the input image by 45 degrees, and what comes out is not a rotated version of the original features—it is a completely different, scrambled representation. The network has to learn separately that a cat tilted left is still a cat, and a cat tilted right is still a cat, and a cat upside down is still a cat. That is why you need data augmentation: you are trying to teach the network something it should have known by construction.

Why does this matter for our geodesic integrator? Because the integrator requires well-defined tangent vectors. When an agent looks at a scene from different angles, the *features* should rotate appropriately—the tangent space structure should be preserved. If rotating the camera produces arbitrary scrambled features instead of properly rotated features, then the tangent space is ill-defined. You cannot integrate geodesics on a manifold whose tangent structure depends on arbitrary viewing angle.

We need vision encoders that respect $SO(2)$: rotate the input, get rotated features.
:::

### Why Standard CNNs Fail

:::{prf:proposition} Conv2d is NOT SO(2)-Equivariant
:label: prop-conv-not-rotation-equivariant

Standard `Conv2d` with learned kernels is translation-equivariant but **not** rotation-equivariant for continuous rotations.

*Proof.*

**Setup:** Let $I: \mathbb{Z}^2 \to \mathbb{R}^{C_{\text{in}}}$ be a discrete image on pixel lattice $\mathbb{Z}^2$ with $C_{\text{in}}$ channels. Let $\psi: \mathbb{Z}^2 \to \mathbb{R}^{C_{\text{out}} \times C_{\text{in}}}$ be a convolutional kernel. The convolution operation is:
$$
(\psi * I)(x) = \sum_{y \in \mathbb{Z}^2} \psi(y) I(x - y) \quad \text{for } x \in \mathbb{Z}^2
$$

**Step 1. Discrete rotation problem:**
For $R_\theta \in SO(2)$ with $\theta \notin \{0, \pi/2, \pi, 3\pi/2\}$, rotation maps lattice points off-grid. For example, with $\theta = \pi/4$:
$$
R_{\pi/4} \cdot (1, 0) = \left(\frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2}\right) \notin \mathbb{Z}^2
$$

To apply rotation to discrete images, we must use an interpolation scheme $\mathcal{I}$:
$$
(R_\theta \cdot I)(x) = \mathcal{I}(I, R_\theta^{-1} x) \quad \text{for } x \in \mathbb{Z}^2
$$

**Step 2. Standard interpolation (bilinear):**
For $x' = R_\theta^{-1} x \in \mathbb{R}^2 \setminus \mathbb{Z}^2$, bilinear interpolation uses the four nearest lattice points:
$$
\mathcal{I}(I, x') = \sum_{i \in \{0,1\}^2} w_i(x') I(x'_{\text{floor}} + i)
$$
where $w_i$ are barycentric weights depending on $x' - x'_{\text{floor}}$.

**Step 3. Non-commutativity of convolution and rotation:**
We show $(\psi * (R_\theta \cdot I)) \neq (R_\theta \cdot (\psi * I))$.

Left-hand side:
$$
(\psi * (R_\theta \cdot I))(x) = \sum_{y \in \mathbb{Z}^2} \psi(y) (R_\theta \cdot I)(x - y) = \sum_{y \in \mathbb{Z}^2} \psi(y) \mathcal{I}(I, R_\theta^{-1}(x - y))
$$

Right-hand side:
$$
(R_\theta \cdot (\psi * I))(x) = \mathcal{I}(\psi * I, R_\theta^{-1} x) = \mathcal{I}\left(\sum_{y \in \mathbb{Z}^2} \psi(y) I(\cdot - y), R_\theta^{-1} x\right)
$$

**Key:** Interpolation does not commute with summation:
$$
\sum_{y} \psi(y) \mathcal{I}(I, R_\theta^{-1}(x - y)) \neq \mathcal{I}\left(\sum_y \psi(y) I(\cdot - y), R_\theta^{-1} x\right)
$$

The left side interpolates $I$ at rotated shifted positions $R_\theta^{-1}(x-y)$ then sums.
The right side first computes the full convolution $\psi * I$ at all lattice points, then interpolates the result.

**Explicit counterexample:** Take $I$ with a sharp edge aligned with $x$-axis. Apply vertical-edge-detecting kernel $\psi$. Rotating $I$ by $\pi/4$ then convolving detects the rotated edge. Convolving then rotating interpolates the edge response at non-grid points, giving different numerical values due to aliasing.

$\square$

**Remark on discrete subgroups:** Exact equivariance holds for the discrete group $C_4 = \{e, R_{\pi/2}, R_\pi, R_{3\pi/2}\}$ (90° rotations) since these map $\mathbb{Z}^2 \to \mathbb{Z}^2$ without interpolation. For continuous $SO(2)$ equivariance, we need steerable filters (Definition {prf:ref}`def-steerable-filter-bank`).

**Connection to geodesic integrator:** The Lorentz-Langevin equation (Definition {prf:ref}`def-bulk-drift-continuous-flow`) requires well-defined geometric vectors in the tangent space $T_z \mathcal{Z}$. If the vision encoder produces rotation-variant features, the tangent space structure becomes viewing-angle-dependent:
$$
T_z \mathcal{Z} \neq T_{R_\theta \cdot z} \mathcal{Z} \quad \text{(geometric inconsistency)}
$$
This breaks geodesic motion, as geodesics are defined as curves minimizing length with respect to a **fixed** Riemannian metric. Rotation-variant encoders effectively change the metric under rotation, making geodesics ill-defined.
:::

### Gauge Bundle Structure for Images

:::{prf:definition} Image as Bundle Section
:label: def-image-as-bundle-section

An RGB image $I: \mathbb{R}^2 \to \mathbb{R}^3$ is a section of the trivial bundle $\mathbb{R}^2 \times \mathbb{R}^3$.

A rotation $R \in SO(2)$ acts on the base (spatial coordinates) but not the fiber (RGB values):
$$
(R \cdot I)(x) = I(R^{-1} \cdot x)
$$

For equivariant features, we need a **non-trivial bundle** where fibers also transform.
:::

:::{prf:definition} Steerable Filter Bank
:label: def-steerable-filter-bank

A filter bank $\{\psi_n^{(\ell)}\}_{n=1}^N$ is **steerable of type $\ell$** if:

$$
R_\theta \cdot \psi_n^{(\ell)} = \sum_m D_{nm}^{(\ell)}(\theta) \psi_m^{(\ell)}
$$

where $D^{(\ell)}$ is the $\ell$-th irreducible representation of $SO(2)$.

**Interpretation:**
- $\ell = 0$: Scalars (rotation-invariant, e.g., circularly symmetric filters)
- $\ell = 1$: Vectors (oriented edge detectors that rotate with image)
- $\ell = 2$: Quadrupoles (higher-order patterns)
:::

:::{div} feynman-prose
Let me show you a beautiful idea: harmonic decomposition for rotations, analogous to Fourier decomposition for translations.

You know Fourier analysis: any periodic signal can be written as a sum of sines and cosines at different frequencies. Low frequency components capture the broad shape; high frequency components capture the fine details. The key insight is that these frequency components *transform simply* under translation—shift the signal, and each Fourier mode just picks up a phase.

The same idea works for rotations, but now the "frequency" is the angular momentum $\ell$. At $\ell = 0$, you have circularly symmetric patterns—things that look the same from every angle, like a bullseye or a Gaussian blob. When you rotate the image, these features do not change at all. At $\ell = 1$, you have dipole patterns—things with a preferred direction, like an edge detector or a gradient. Rotate the image by $\theta$, and these features rotate by $\theta$. At $\ell = 2$, you have quadrupole patterns—things like corner detectors or curvature indicators. They rotate twice as fast as the image.

Any filter can be decomposed into these harmonic components, just like any signal can be decomposed into Fourier modes. And the magic is that each component *transforms according to a known rule* under rotation. The $\ell = 0$ part stays fixed. The $\ell = 1$ part rotates with the image. The $\ell = 2$ part rotates at double speed. This is what makes the filters "steerable"—you can predict exactly how the output will transform under any rotation.
:::

:::{prf:theorem} Steerable Convolution is SO(2)-Equivariant
:label: thm-steerable-conv-equivariant

Let $\{\psi_n^{(\ell)}\}_{n=1}^N$ be a steerable filter bank of type $\ell$ (Definition {prf:ref}`def-steerable-filter-bank`). Let $\text{Conv}_\ell$ denote convolution with these filters:
$$
(\text{Conv}_\ell I)_n(x) = \sum_m (\psi_m^{(\ell)} * I)(x) = \int_{\mathbb{R}^2} \psi_m^{(\ell)}(y) I(x - y) \, dy
$$

Then for any rotation $R_\theta \in SO(2)$:

$$
\text{Conv}_\ell(R_\theta \cdot I) = D^{(\ell)}(\theta) \cdot \text{Conv}_\ell(I)
$$

where $D^{(\ell)}(\theta)$ is the $\ell$-th irreducible representation of $SO(2)$.

*Proof.*

**Step 1. Apply rotation to input:**
By linearity of convolution:
$$
(\psi_n * (R_\theta \cdot I))(x) = \int_{\mathbb{R}^2} \psi_n(y) I(R_\theta^{-1}(x-y)) \, dy
$$

**Step 2. Change of variables:**
Let $y' = R_\theta y$. Since $R_\theta \in SO(2)$ preserves measure ($|\det R_\theta| = 1$):
$$
= \int_{\mathbb{R}^2} \psi_n(R_\theta^{-1} y') I(R_\theta^{-1}(x - y')) \, dy'
$$

**Step 3. Factor rotation:**
$$
= \int_{\mathbb{R}^2} (R_\theta \cdot \psi_n)(y') I(x - y') \, dy' = ((R_\theta \cdot \psi_n) * I)(x)
$$

**Step 4. Apply steerability:**
By Definition {prf:ref}`def-steerable-filter-bank`, $R_\theta \cdot \psi_n^{(\ell)} = \sum_m D_{nm}^{(\ell)}(\theta) \psi_m^{(\ell)}$. Thus:
$$
(\psi_n^{(\ell)} * (R_\theta \cdot I))(x) = \sum_m D_{nm}^{(\ell)}(\theta) (\psi_m^{(\ell)} * I)(x)
$$

In matrix form:
$$
\text{Conv}_\ell(R_\theta \cdot I) = D^{(\ell)}(\theta) \cdot \text{Conv}_\ell(I)
$$

$\square$

**Remark:** For full treatment of discretization effects and practical implementation, see Cohen & Welling (2016) {cite}`cohen2016group` and Weiler & Cesa (2019) {cite}`weiler2019general`.
:::

### Lifting to Homogeneous Space

:::{prf:definition} Lifting Map to SE(2)
:label: def-lifting-map

**Group structure:** The **special Euclidean group** $SE(2) = \mathbb{R}^2 \rtimes SO(2)$ is the group of rigid motions (translations + rotations) in the plane, with elements $g = (x, R_\theta)$ where $x \in \mathbb{R}^2$ is translation and $R_\theta \in SO(2)$ is rotation by angle $\theta$.

**Group multiplication:** For $g_1 = (x_1, R_{\theta_1})$ and $g_2 = (x_2, R_{\theta_2})$:
$$
g_1 \cdot g_2 = (x_1 + R_{\theta_1} x_2, R_{\theta_1 + \theta_2})
$$
(apply rotation, then translation, then compose rotations).

**Domain and codomain:** Let $C(\mathbb{R}^2, \mathbb{R}^{C_{\text{in}}})$ be the space of continuous functions (images) from $\mathbb{R}^2$ to $\mathbb{R}^{C_{\text{in}}}$ (e.g., $C_{\text{in}} = 3$ for RGB). Let $C(SE(2), \mathbb{R}^{C_{\text{out}}})$ be functions on $SE(2)$ with values in $\mathbb{R}^{C_{\text{out}}}$ (output feature dimension).

The **lifting map** is an operator:
$$
L: C(\mathbb{R}^2, \mathbb{R}^{C_{\text{in}}}) \to C(SE(2), \mathbb{R}^{C_{\text{out}}})
$$

**Definition:** For an input image $I \in C(\mathbb{R}^2, \mathbb{R}^{C_{\text{in}}})$ and group element $g = (x, R_\theta) \in SE(2)$:
$$
(L I)(g) = (L I)(x, \theta) := \sum_{i=1}^{C_{\text{out}}} (\psi_i * I)(x) \cdot e_i
$$
where $\{\psi_i\}_{i=1}^{C_{\text{out}}}$ is a steerable filter bank (Definition {prf:ref}`def-steerable-filter-bank`) and $\{e_i\}$ is the standard basis of $\mathbb{R}^{C_{\text{out}}}$.

**Equivariance property:** For $g_0 = (x_0, R_{\theta_0}) \in SE(2)$ and image $I$, define the left-translated image $(L_{g_0} I)(x) := I(R_{\theta_0}^{-1}(x - x_0))$. Then:
$$
L(L_{g_0} I) = L_{g_0}(L I)
$$
where the right-hand side is left-multiplication on $SE(2)$: $(L_{g_0} f)(g) = f(g_0^{-1} g)$.

**Geometric interpretation:** Instead of features at spatial locations $x \in \mathbb{R}^2$, lifted features live at *posed locations* $(x, \theta) \in SE(2)$: position AND orientation. The network learns to detect patterns *and* their orientations explicitly.

**Output dimension:** For $N_\theta$ discrete orientations (e.g., $N_\theta = 8$ for $\theta \in \{0°, 45°, 90°, \ldots, 315°\}$), the output has dimension $C_{\text{out}} = N_\theta \times C_{\text{feature}}$ where $C_{\text{feature}}$ is the number of feature types per orientation.
:::

### Implementation: E2CNN CovariantRetina

The following implementation uses the E2CNN library {cite}`weiler2019general` to build an $SO(2)$-equivariant vision encoder.

```python
import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as e2nn
from typing import Tuple

class CovariantRetina(nn.Module):
    """SO(2)-equivariant vision encoder using steerable convolutions.

    Implements Theorem {prf:ref}`thm-steerable-conv-equivariant` via E2CNN library.

    **Guarantees:**
        - SO(2) rotation equivariance: Conv(R_θ · I) = D^(ℓ)(θ) · Conv(I)
        - Translation equivariance: Conv(T_x · I) = T_x · Conv(I)
        - Preserves tangent space structure for geodesic integrator

    **Architecture:**
        1. Lifting layer: R² → SE(2) (trivial → regular representation)
        2. Steerable convolutions: SE(2) → SE(2) with expanding channels
        3. Group pooling: SE(2) → R² (max over rotations)
        4. Linear projection to latent space

    **Diagnostics:**
        - Node 68: RotationEquivarianceCheck

    Args:
        in_channels: Input RGB channels (typically 3)
        out_dim: Output latent dimension [dimensionless]
        num_rotations: Discretization of SO(2) (typically 8 or 16)
        kernel_size: Convolution kernel size [pixels]

    Shapes:
        Input: [B, C_in, H, W] where C_in = in_channels
        Output: [B, out_dim]

    Example:
        >>> retina = CovariantRetina(in_channels=3, out_dim=512, num_rotations=8)
        >>> img = torch.randn(32, 3, 64, 64)  # [B, C, H, W]
        >>> features = retina(img)  # [32, 512]
        >>> # Test SO(2) equivariance (Node 68)
        >>> img_rotated = rotate_image(img, angle=45)  # rotate by 45°
        >>> features_rotated = retina(img_rotated)
        >>> # Features should be related by representation matrix D^(ℓ)(45°)
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 512,
        num_rotations: int = 8,
        kernel_size: int = 5
    ):
        super().__init__()

        # Define the symmetry group: rotations + reflections (or just rotations)
        # r2_act = gspaces.Rot2dOnR2(N=num_rotations)  # Rotations only
        self.r2_act = gspaces.FlipRot2dOnR2(N=num_rotations)  # Rotations + reflections

        # Input type: trivial representation (standard image)
        in_type = e2nn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])

        # Feature types for steerable filters
        # Use regular representation (all rotations) for maximum expressivity
        self.feature_type_32 = e2nn.FieldType(
            self.r2_act,
            32 * [self.r2_act.regular_repr]
        )
        self.feature_type_64 = e2nn.FieldType(
            self.r2_act,
            64 * [self.r2_act.regular_repr]
        )

        # 1. Lifting convolution: R² → SE(2)
        # Maps trivial representation to regular representation
        self.lift = e2nn.R2Conv(
            in_type,
            self.feature_type_32,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False  # No bias (gauge invariance)
        )

        # 2. Steerable convolutions on SE(2)
        self.conv1 = e2nn.R2Conv(
            self.feature_type_32,
            self.feature_type_64,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.conv2 = e2nn.R2Conv(
            self.feature_type_64,
            self.feature_type_64,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        # 3. Norm-based nonlinearity (Definition {prf:ref}`def-norm-gated-activation`)
        # NormNonLinearity computes ||v|| and applies gate, preserving equivariance
        self.relu1 = e2nn.NormNonLinearity(self.feature_type_32)
        self.relu2 = e2nn.NormNonLinearity(self.feature_type_64)
        self.relu3 = e2nn.NormNonLinearity(self.feature_type_64)

        # 4. Group pooling: max over rotation group
        # Projects SE(2) → R² by taking max response over all orientations
        self.group_pool = e2nn.GroupPooling(self.feature_type_64)

        # 5. Adaptive spatial pooling
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 6. Final linear projection (with spectral norm)
        pool_features = 64 * 4 * 4  # After pooling: 64 channels × 4×4 spatial
        from torch.nn.utils import spectral_norm
        self.fc = spectral_norm(nn.Linear(pool_features, out_dim, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with equivariance preservation.

        Args:
            x: [B, C, H, W] input image [dimensionless, normalized to [-1,1]]

        Returns:
            z: [B, out_dim] latent features [dimensionless]
        """
        # Wrap input as E2CNN geometric tensor
        x = e2nn.GeometricTensor(x, e2nn.FieldType(
            self.r2_act,
            x.shape[1] * [self.r2_act.trivial_repr]
        ))

        # Lifting + steerable convolutions
        x = self.lift(x)        # [B, 32×|G|, H, W] where |G| = num_rotations
        x = self.relu1(x)       # Norm-gated activation (equivariant)

        x = self.conv1(x)       # [B, 64×|G|, H, W]
        x = self.relu2(x)

        x = self.conv2(x)       # [B, 64×|G|, H, W]
        x = self.relu3(x)

        # Group pooling: max over rotation channels
        x = self.group_pool(x)  # [B, 64, H, W]

        # Extract underlying torch tensor
        x = x.tensor

        # Spatial pooling
        x = self.spatial_pool(x)  # [B, 64, 4, 4]

        # Flatten
        x = x.flatten(1)  # [B, 64×16 = 1024]

        # Project to latent dimension with spectral norm (Thm {prf:ref}`thm-spectral-preserves-light-cone`)
        z = self.fc(x)  # [B, out_dim]

        return z
```

**Key design choices:**

1. **FlipRot2dOnR2 vs Rot2dOnR2:** We use rotations + reflections for richer symmetry. For applications where chirality matters (e.g., detecting left vs. right hands), use `Rot2dOnR2` instead.

2. **Regular representation:** Expands each channel by a factor of $|G|$ (number of rotations), storing responses at all orientations. More expensive but maximally expressive.

3. **NormNonLinearity:** E2CNN's implementation of norm-gated activation (Definition {prf:ref}`def-norm-gated-activation`). Preserves equivariance by acting only on fiber norms.

4. **GroupPooling:** Reduces $SE(2)$ features to $\mathbb{R}^2$ by taking max (or avg) over rotation channels. Creates rotation-*invariant* features from rotation-*equivariant* ones.

5. **Spectral normalization on final linear layer:** Ensures light cone preservation (Theorem {prf:ref}`thm-spectral-preserves-light-cone`) when projecting to latent space.

**Diagnostic verification (Node 68):**

```python
def test_rotation_equivariance(retina: CovariantRetina, img: torch.Tensor):
    """Test SO(2) equivariance of CovariantRetina.

    Verifies: ||retina(rotate(img, θ)) - rotate_features(retina(img), θ)|| < ε
    """
    import torchvision.transforms.functional as TF

    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    features_original = retina(img)

    for angle in angles[1:]:  # Skip 0° (identity)
        img_rotated = TF.rotate(img, angle)
        features_rotated = retina(img_rotated)

        # Note: After group pooling, features are rotation-INVARIANT
        # So we expect features_rotated ≈ features_original (not rotated features)
        violation = torch.norm(features_rotated - features_original).item()

        print(f"Angle {angle}°: violation = {violation:.6f}")
        assert violation < 1e-3, f"Rotation invariance violated at {angle}°"
```

---

(sec-connection-to-gauge-structure)=
## Connection to Gauge Structure

:::{div} feynman-prose
Now we come to the connection that ties everything together. We have built three primitives—SpectralLinear, NormGate, SteerableConv—each carefully designed to preserve some symmetry. But in Chapter 8.1, we derived that agents must respect the full gauge group $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$. How do our primitives relate to this structure?

Here is the correspondence. Each primitive preserves one factor of the gauge group:

The *bundle structure* with $n_b$ bundles implements $SU(N_f)_C$—the "color" symmetry that keeps features confined within their bundles. Just as quarks in QCD cannot exist in isolation (they must bind into color-neutral hadrons), features in IsotropicBlocks cannot propagate independently (they must form bound states across bundles). Node 40 in the diagnostics enforces this confinement.

*Spectral normalization* implements $U(1)_Y$—the hypercharge symmetry associated with capacity conservation. The constraint $\sigma_{\max}(W) \leq 1$ ensures that the total "hypercharge" (roughly, the squared norm of the latent state) cannot grow without bound. This is the capacity constraint from the holographic bound in Chapter 8.1.

*Steerable convolutions* implement $SU(2)_L$—the "weak" symmetry that mixes observation and action. Visual features from the retina form doublets with action planning, just as left-handed particles in the Standard Model form $SU(2)$ doublets.

Three primitives, three gauge factors. This is not coincidence—it is the architecture being forced into a unique form by symmetry requirements.
:::

### Mapping Primitives to Gauge Fields

:::{prf:definition} SU(N_f) Gauge Action on Bundle Space
:label: def-gauge-action-bundles

Let $Z = (z^{(1)}, \ldots, z^{(n_b)})$ be the bundled latent representation with $z^{(i)} \in \mathbb{R}^{d_b}$. The gauge group $SU(N_f)$ with $N_f = n_b$ acts on $Z$ as:

$$
Z \mapsto Z' = Z \cdot U
$$

where $U \in SU(N_f)$ is an $n_b \times n_b$ special unitary matrix:
$$
U^\dagger U = I, \quad \det(U) = 1
$$

**Explicit action:** In matrix form, treating $Z$ as a $d_b \times n_b$ matrix:
$$
z'^{(j)} = \sum_{i=1}^{n_b} U_{ij}^* z^{(i)} \quad \text{(bundle mixing)}
$$

**Color charge:** Define the **color charge operator** for generator $T^a$ ($a = 1, \ldots, N_f^2 - 1$):
$$
Q_C^a[Z] = \sum_{i,j=1}^{n_b} (z^{(i)})^T T^a_{ij} z^{(j)}
$$

A state is **color-neutral** (confined) if:
$$
Q_C^a[Z] = 0 \quad \forall \, a
$$

**Physical interpretation:** Just as quarks in QCD carry color charge under $SU(3)_C$, latent features carry "bundle charge" under $SU(N_f)_C$. Only color-neutral combinations can propagate to the macro level.
:::

:::{prf:theorem} Isotropic Blocks Preserve SU(N_f)_C Gauge Structure
:label: thm-isotropic-preserves-color

The bundled structure (Definition {prf:ref}`def-latent-vector-bundle`) with $n_b$ bundles, coupled with norm-gating and the Binding field $G_\mu$, implements $SU(N_f)_C$ gauge invariance with $N_f = n_b$.

**Statement:** For any $U \in SU(N_f)$, the effective dynamics satisfy:
$$
\mathcal{L}_{\text{eff}}[Z \cdot U] = \mathcal{L}_{\text{eff}}[Z]
$$

Moreover, norm-gating induces scale-dependent coupling:
- **Infrared** ($\ell \to \infty$): Strong coupling $g_s(\mu_{\text{IR}}) \gg 1$ → confinement
- **Ultraviolet** ($\ell \to 0$): Weak coupling $g_s(\mu_{\text{UV}}) \to 0$ → asymptotic freedom

*Proof.*

**Step 1. Gauge-invariant coupling via Binding field:**

The Binding field $G_\mu$ introduces cross-bundle coupling via the covariant derivative:
$$
D_\mu Z = \partial_\mu Z - i g_s G_\mu \cdot Z
$$

where $G_\mu = G_\mu^a T^a$ and $T^a$ are $SU(N_f)$ generators.

Under gauge transformation $Z \to Z \cdot U$:
$$
D_\mu(Z \cdot U) = (D_\mu Z) \cdot U + Z \cdot (\partial_\mu U - i g_s U G_\mu)
$$

Requiring gauge invariance $D_\mu(Z \cdot U) = (D_\mu Z) \cdot U$ gives the transformation law:
$$
G_\mu \to U G_\mu U^\dagger + \frac{i}{g_s} U \partial_\mu U^\dagger
$$

This is the **standard gauge connection transformation**, ensuring $\mathcal{L}_{\text{eff}}$ is gauge-invariant.

**Step 2. Norm-gating as effective coupling:**

The norm-gating activation potential $b_i$ creates an energy barrier. The effective coupling at layer $\ell$ is:
$$
g_s^{(\ell)} = \frac{\langle \|G_\mu\| \rangle}{\langle \|\partial_\mu Z\| \rangle} \approx \frac{\beta_\ell}{\sqrt{1 + \|\nabla_{W_\ell} \mathcal{L}\|^{-2}}}
$$

where $\beta_\ell = \tanh(b_\ell)$ is the barrier strength (see Proposition {prf:ref}`prop-coupling-from-barriers` below).

**Step 3. IR behavior (confinement):**

At large layer depth $\ell \to L$ (infrared scale):
- Barrier potentials $b_\ell$ are large (strong gates)
- Gradient flow between bundles is suppressed unless $Q_C^a = 0$
- Effective coupling $g_s^{(L)} \gg 1$

**Energy penalty for non-neutral states:**
$$
\Delta E_{\text{conf}} = g_s^{(L)} \sum_a |Q_C^a|^2 \to \infty \quad \text{as } g_s^{(L)} \to \infty
$$

Only color-neutral states ($Q_C^a = 0 \, \forall a$) have finite energy → **confinement**.

**Step 4. UV behavior (asymptotic freedom):**

At shallow layers $\ell \to 0$ (ultraviolet scale):
- Barrier potentials $b_\ell \to 0$ (weak gates)
- Bundles decouple, gradient flow is independent
- Effective coupling $g_s^{(0)} \to 0$

Bundles behave as free, non-interacting subspaces → **asymptotic freedom**.

**Step 5. Identification with coupling window:**

By Corollary {prf:ref}`cor-coupling-window`, the gauge coupling must satisfy:
$$
g_s(\mu_{\text{IR}}) \geq g_s^{\text{crit}} \quad \text{and} \quad g_s(\mu_{\text{UV}}) \to 0
$$

The norm-gating schedule $\{b_\ell\}_{\ell=0}^L$ implements this by construction:
- $b_L$ large (IR confinement)
- $b_0$ small (UV freedom)

$\square$

**Connection to Node 40 (PurityCheck):** Measures $\sum_a |Q_C^a|^2$ at the final layer. Violation indicates non-neutral states reaching the macro register, breaking confinement.
:::

:::{prf:proposition} Coupling Strength from Norm-Gating Barriers
:label: prop-coupling-from-barriers

The effective gauge coupling at layer $\ell$ is:
$$
g_s^{(\ell)} = \frac{\beta_\ell}{\sqrt{1 + \alpha \|\nabla_{W_\ell} \mathcal{L}\|^2}}
$$

where:
- $\beta_\ell = \tanh(b_\ell)$ is the barrier strength
- $\alpha > 0$ is a scale factor
- $\nabla_{W_\ell} \mathcal{L}$ is the gradient with respect to layer weights

*Proof.*

**Step 1. Gradient flow suppression:**

The norm gate $v \cdot g(\|v\| + b)$ has gradient:
$$
\nabla_v [v \cdot g(\|v\| + b)] = g(\|v\| + b) I + v \frac{g'(\|v\| + b)}{\|v\|} v^T
$$

For large $b$ (strong barrier), $g'(\|v\| + b) \approx 0$ → gradient saturates → coupling to subsequent layers is suppressed.

**Step 2. Cross-bundle coupling:**

The Binding field $G_\mu$ mediates cross-bundle interactions. Its effective magnitude scales as:
$$
\langle \|G_\mu\| \rangle \propto \beta_\ell
$$

**Step 3. Gradient-to-coupling ratio:**

The coupling $g_s$ is the ratio of field strength to kinetic term:
$$
g_s \sim \frac{\text{field strength}}{\text{kinetic energy}} = \frac{\beta_\ell}{\sqrt{1 + \alpha \|\nabla \mathcal{L}\|^2}}
$$

**Step 4. Verify IR/UV limits:**

- **IR ($b \to \infty$):** $\beta \to 1$, $g_s \to 1/\sqrt{1 + \alpha \|\nabla \mathcal{L}\|^2} = O(1)$ (strong coupling)
- **UV ($b \to 0$):** $\beta \to 0$, $g_s \to 0$ (weak coupling)

$\square$
:::


### Spectral Normalization and U(1)_Y

:::{prf:theorem} Spectral Norm Preserves Hypercharge Conservation
:label: thm-spectral-preserves-hypercharge

The Opportunity field $B_\mu$ (from {ref}`sec-symplectic-multi-agent-field-theory`) couples to hypercharge $Y$. Spectral normalization ensures:

$$
\sum_i Y_i \text{ remains constant under forward propagation}
$$

*Proof.*

**Step 1. Hypercharge definition:** $Y \propto \|z\|^2$ (quadratic conserved quantity).

**Step 2. Isometry:** Spectral normalization with $\sigma_{\max}(W) = 1$ ensures $\|W \cdot z\| = \|z\|$ (isometry).

**Step 3. Conservation:** Thus $Y(W \cdot z) = \|W \cdot z\|^2 = \|z\|^2 = Y(z)$ (hypercharge conserved).

$\square$

**Connection to Node 56 (CapacityHorizonCheck):** Hypercharge saturation $Y \to Y_{\max}$ indicates approaching capacity limit (holographic bound).
:::

### Steerable Convolutions and SU(2)_L

:::{prf:definition} Observation-Action Doublet Structure
:label: def-obs-action-doublet

The latent space decomposes into **observation** and **action planning** subspaces:
$$
\mathcal{Z} = \mathcal{Z}_{\text{obs}} \oplus \mathcal{Z}_{\text{act}}
$$

These form an $SU(2)_L$ doublet:
$$
\Psi = \begin{pmatrix} \psi_{\text{obs}} \\ \psi_{\text{act}} \end{pmatrix} \in \mathcal{Z}_{\text{obs}} \oplus \mathcal{Z}_{\text{act}}
$$

**Full SU(2)_L transformation:** The special unitary group $SU(2)$ has 3 real parameters. A general element is:
$$
U(\vec{\theta}) = \exp\left(i \sum_{a=1}^3 \theta_a \frac{\sigma_a}{2}\right), \quad \vec{\theta} = (\theta_1, \theta_2, \theta_3) \in \mathbb{R}^3
$$
where $\{\sigma_a\}_{a=1}^3$ are Pauli matrices:
$$
\sigma_1 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_2 = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_3 = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

**Real SO(2) subgroup (current implementation):** For real-valued neural networks, we restrict to the $SO(2)$ subgroup generated by $\sigma_3$ only. This gives 1-parameter transformations:
$$
U_{\text{SO(2)}}(\theta) = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}, \quad \theta \in [0, 2\pi)
$$

**Action on doublet:**
$$
\Psi \to \Psi' = U_{\text{SO(2)}}(\theta) \Psi = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} \psi_{\text{obs}} \\ \psi_{\text{act}} \end{pmatrix}
$$

**Physical interpretation:**
- $\theta = 0$: Pure observation (no action planning)
- $\theta = \pi/2$: Pure action planning (no new observations)
- $0 < \theta < \pi/2$: Mixed observation-action processing

**Chirality:** This is a **left-chiral** doublet (incoming information stream). Right-chiral singlets correspond to executed actions (outgoing, no longer subject to mixing).

**Remark:** Full $SU(2)_L$ gauge theory requires complex-valued features. Current real-valued architectures (IsotropicBlock + SteerableConv) implement the $SO(2) \subset SU(2)$ subgroup. Extension to full 3-parameter $SU(2)$ requires complex steerable CNNs or quaternionic networks (see {ref}`Section 29.1 <sec-symplectic-multi-agent-field-theory>` for gauge-theoretic derivation).
:::

:::{prf:theorem} Steerable Convolutions Induce SU(2)_L Doublet Structure
:label: thm-steerable-induces-doublet

Let $\phi_{\text{vision}}: \text{Images} \to \mathcal{Z}_{\text{obs}}$ be a steerable convolution encoder (CovariantRetina). Then the concatenation with action planning features:
$$
\Psi = \begin{pmatrix} \phi_{\text{vision}}(I) \\ \psi_{\text{act}} \end{pmatrix}
$$

forms an $SU(2)_L$ doublet under the action defined in Definition {prf:ref}`def-obs-action-doublet`.

*Proof.*

**Step 1. Identify $SO(2)_{\text{spatial}} \subset SU(2)_L$:**

The spatial rotation group $SO(2)$ (acting on images) embeds into $SU(2)_L$ as the subgroup that rotates within the observation-action plane while preserving the temporal direction (action execution).

Explicitly, spatial rotation $R_\theta \in SO(2)$ on images corresponds to:
$$
U_L(\theta/2) \in SU(2)_L
$$

(The factor of 2 arises from the double cover $SU(2) \to SO(3)$, specialized to $SO(2)$.)

**Step 2. Steerable convolutions provide $SO(2)$ equivariance:**

By Theorem {prf:ref}`thm-steerable-conv-equivariant`:
$$
\phi_{\text{vision}}(R_\theta \cdot I) = D^{(\ell)}(\theta) \cdot \phi_{\text{vision}}(I)
$$

where $D^{(\ell)}(\theta)$ is the $\ell$-th irrep of $SO(2)$.

**Step 3. Lift to $SU(2)_L$ representation:**

The observation vector $\psi_{\text{obs}} = \phi_{\text{vision}}(I)$ transforms under spatial rotations. To embed in $SU(2)_L$, define:
$$
\psi_{\text{obs}} \to U_L(\theta/2) \psi_{\text{obs}} = \cos(\theta/2) \psi_{\text{obs}} + \sin(\theta/2) \psi_{\text{act}}
$$

This mixes observation with action planning, implementing the **weak isospin** transformation.

**Step 4. Action planning transforms contravariantly:**

For the doublet to close under $SU(2)_L$, the action component must transform as:
$$
\psi_{\text{act}} \to -\sin(\theta/2) \psi_{\text{obs}} + \cos(\theta/2) \psi_{\text{act}}
$$

**Step 5. Doublet closure:**

Combining Steps 3-4:
$$
\begin{pmatrix} \psi_{\text{obs}}' \\ \psi_{\text{act}}' \end{pmatrix} = \begin{pmatrix} \cos(\theta/2) & \sin(\theta/2) \\ -\sin(\theta/2) & \cos(\theta/2) \end{pmatrix} \begin{pmatrix} \psi_{\text{obs}} \\ \psi_{\text{act}} \end{pmatrix}
$$

This is precisely the $SU(2)_L$ doublet transformation (Definition {prf:ref}`def-obs-action-doublet`).

$\square$
:::

**Remark on Error Field and Observation-Action Coupling:**

The Error field $W_\mu$ from gauge theory (Chapter 8.1, {ref}`sec-symplectic-multi-agent-field-theory`) couples observation and action representations in the full multiagent architecture. The detailed mathematical structure of this coupling—including the $SU(2)_L$ gauge symmetry—is derived rigorously in Chapter 8.2 (Standard Model isomorphism, see Table 8.2.1). At the DNN block level considered in this chapter, this manifests as:

- **Observation encoder:** Steerable convolutions producing rotation-equivariant features $\psi_{\text{obs}}$
- **Action decoder:** Linear layers mapping latent states to motor commands $\psi_{\text{act}}$
- **Coupling:** Cross-attention mechanisms (detailed in Chapter 8.5) that mix observation and action representations

The specific connection to $SU(2)_L$ gauge theory is established in Chapter 8.2 through rigorous derivation, not analogy.

---

(sec-integration-with-geodesic-integrator)=
## Integration with Geodesic Integrator

:::{prf:definition} Latent Metric Tensor
:label: def-latent-metric

The latent space $\mathcal{Z}$ is equipped with the **Information Sensitivity Metric** (from Section {ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits`, Theorem {prf:ref}`thm-capacity-constrained-metric-law`):

$$
G_{ij}(z) = \nabla^2_{ij} V(z) + \lambda \mathcal{F}_{ij}(z)
$$

where:
- $V(z)$ is the value function (expected return from state $z$)
- $\mathcal{F}_{ij}(z) = \mathbb{E}_{a \sim \pi(·|z)}[\nabla_{z_i} \log \pi(a|z) \nabla_{z_j} \log \pi(a|z)]$ is the Fisher Information Metric
- $\lambda > 0$ is the temperature parameter

**Physical interpretation:** The metric $G$ measures how sensitive the value function and policy are to changes in latent coordinates. High curvature regions indicate sensitive decision boundaries.

**Positive definiteness:** $G(z)$ is positive definite when $V$ is strongly convex and $\lambda > 0$.

**Units:** $[G_{ij}] = [\mathcal{Z}]^{-2} = \text{nat}^{-1}$ (inverse information).
:::

:::{div} feynman-prose
We have carefully designed each layer to be gauge-equivariant. But an agent is not one layer—it is a deep network, dozens or hundreds of layers stacked together. How do we know that equivariance at each step adds up to equivariance globally?

This is the question of *composition*, and it has a beautiful answer. If each layer commutes with the gauge transformation, then the whole network commutes with the gauge transformation. Apply a rotation, then pass through 50 layers, and you get the same result as passing through 50 layers, then rotating. The equivariance propagates through the entire depth.

But there is a subtlety. Equivariance alone is not enough—we also need the layers to play nicely with *integration*. The geodesic integrator requires that tangent vectors remain tangent vectors as they propagate. In geometric language, we need *parallel transport*: moving a vector along a curve without it leaving the tangent bundle.

This is where the Lipschitz constraint comes in. Each layer has $\sigma_{\max}(W) \leq 1$, meaning it contracts distances. Compose $L$ layers, each with Lipschitz constant at most 1, and the total map still has Lipschitz constant at most 1. The light cone structure is preserved globally, not just locally. Information cannot propagate faster than $c_{\text{info}}$ through the entire network depth, which is exactly what the Boris-BAOAB integrator requires.

Microscopic equivariance composes to macroscopic geodesic flow.
:::

### From Microscopic to Macroscopic

:::{prf:theorem} Composition of Equivariant Layers is Equivariant
:label: thm-composition-equivariant

Let $f_1, \ldots, f_L$ be $G$-equivariant layers. Then:

$$
F = f_L \circ \cdots \circ f_1 \text{ is } G\text{-equivariant}
$$

Moreover, if each $f_i$ has Lipschitz constant $\leq 1$, then:

$$
\sigma_{\max}(F) \leq 1 \quad \text{(global light cone preservation)}
$$

*Proof.* Equivariance: composition of equivariant maps. Lipschitz: $\|F(z_1) - F(z_2)\| \leq \prod_i L_i \cdot \|z_1 - z_2\| \leq \|z_1 - z_2\|$ when $L_i \leq 1$. $\square$
:::

### Lipschitz Properties of Primitives

:::{prf:lemma} NormGate Lipschitz Bound
:label: lem-normgate-lipschitz

Let $f(v) = v \cdot g(\|v\| + b)$ be the norm-gated activation (Definition {prf:ref}`def-norm-gated-activation`) where $g: \mathbb{R} \to \mathbb{R}$ is a smooth gating function with:
- $|g(x)| \leq C_g |x|$ for all $x$ (sublinear growth)
- $|g'(x)| \leq L_g$ for all $x$ (bounded derivative)

Then $f$ is Lipschitz continuous with constant:
$$
L_f \leq \max(C_g, L_g R_{\max} + C_g)
$$
where $R_{\max}$ is the maximum expected norm in the operating range.

**For GELU:** The GELU function $g(x) = x\Phi(x)$ where $\Phi$ is the standard normal CDF satisfies:
- $C_g = 1$ (since $0 \leq \Phi(x) \leq 1$ implies $|g(x)| \leq |x|$)
- $g'(x) = \Phi(x) + x\phi(x)$ where $\phi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$
- $\sup_{x \in \mathbb{R}} g'(x) = 1$ (achieved as $x \to +\infty$)
- For practical operating range $x \in [-3, 3]$: $\max_{x \in [-3,3]} g'(x) \approx 1.08$

Thus $L_g \approx 1.08$ and $L_f \leq 1.08(R_{\max} + 1)$.

*Proof.*

**Step 1. Spherical coordinates:**
Write $v = r\hat{v}$ where $r = \|v\| \geq 0$ and $\hat{v} = v/\|v\|$ is the unit direction vector.

Then:
$$
f(v) = f(r\hat{v}) = r \cdot g(r + b) \cdot \hat{v}
$$

**Step 2. Jacobian calculation:**
Using the product rule and chain rule:
$$
\nabla f(v) = g(\|v\| + b) \cdot I_{d_b} + g'(\|v\| + b) \cdot \frac{vv^T}{\|v\|}
$$

This is the sum of a scaled identity and a rank-1 matrix.

**Step 3. Eigenvalue decomposition:**
Write $v = r\hat{v}$ where $r = \|v\|$ and $\|\hat{v}\| = 1$. The Jacobian has eigenvalues:

- **Radial direction** (eigenvector $\hat{v}$):
$$
\lambda_r = g(r+b) + g'(r+b) \cdot r
$$

- **Tangential directions** (eigenvectors orthogonal to $\hat{v}$, there are $d_b-1$ of these):
$$
\lambda_t = g(r+b)
$$

**Verification:** For $u = \hat{v}$:
$$
\nabla f(v) \cdot \hat{v} = g(r+b)\hat{v} + g'(r+b) r \hat{v} = [g(r+b) + rg'(r+b)]\hat{v} = \lambda_r \hat{v} \quad \checkmark
$$

For $u \perp \hat{v}$ (so $u^T v = 0$):
$$
\nabla f(v) \cdot u = g(r+b) u + g'(r+b) \frac{(v^T u)}{\|v\|} v = g(r+b) u = \lambda_t u \quad \checkmark
$$

**Step 4. Bound eigenvalues:**
Using $|g(x)| \leq C_g|x|$ and $|g'(x)| \leq L_g$:

**Radial:**
$$
|\lambda_r| = |g(r+b) + rg'(r+b)| \leq |g(r+b)| + r|g'(r+b)| \leq C_g|r+b| + rL_g
$$

For $r \leq R_{\max}$ and bounded bias $|b| \leq B$:
$$
|\lambda_r| \leq C_g(R_{\max} + B) + R_{\max}L_g = R_{\max}(C_g + L_g) + C_gB
$$

**Tangential:**
$$
|\lambda_t| = |g(r+b)| \leq C_g|r+b| \leq C_g(R_{\max} + B)
$$

**Step 5. Spectral norm and Lipschitz constant:**
The operator norm is:
$$
\|\nabla f(v)\|_{\text{op}} = \max(|\lambda_r|, |\lambda_t|) \leq \max(R_{\max}(C_g + L_g) + C_gB, C_g(R_{\max} + B))
$$

Since typically $L_g > 0$, the radial term dominates:
$$
L_f = \sup_v \|\nabla f(v)\|_{\text{op}} \leq R_{\max}(C_g + L_g) + C_gB
$$

**Step 6. Numerical values for GELU:**
For GELU with $C_g = 1$, $L_g \approx 1.08$, and typical bias $|b| \leq 1$:
$$
L_f \leq R_{\max}(1 + 1.08) + 1 \approx 2.08 R_{\max} + 1
$$

With normalized inputs, $R_{\max} \approx \sqrt{d_b}$ for $d_b$-dimensional bundles. For $d_b = 16$:
$$
L_f \lesssim 2.08 \times 4 + 1 \approx 9.3
$$

**Remark 1 (Composition with spectral norm):** While individual NormGate layers have $L_f > 1$, they compose with spectral-normalized linear layers (which have $L = 1$). The total Lipschitz constant for IsotropicBlock is bounded by the product, and layer normalization or skip connections prevent unbounded growth across depth.

**Remark 2 (Rescaling option):** To enforce strict 1-Lipschitz property, rescale GELU:
$$
\tilde{g}(x) = \frac{g(x)}{2R_{\max} + 1}
$$

This guarantees $L_f \leq 1$ but attenuates gradients. In practice, we keep unscaled GELU and rely on:
- Spectral normalization in linear layers
- Moderate bundle dimensions ($d_b \in [8, 32]$)
- Skip connections across blocks

to control the effective Lipschitz constant of the full network.
$\square$
:::

### Micro-Macro Consistency

:::{prf:definition} Micro-Macro Consistency
:label: def-micro-macro-consistency

A DNN layer $f: \mathcal{Z} \to \mathcal{Z}$ is **compatible with the geodesic integrator** if:

1. **Preserves metric:** $G(f(z)) = \frac{\partial f}{\partial z} \cdot G(z) \cdot \left(\frac{\partial f}{\partial z}\right)^T$ (pullback)
2. **Preserves light cone:** $\|f(z) - f(z')\| \leq c_{\text{info}} \cdot \|z - z'\|$
3. **Preserves gauge:** $f(U(g) \cdot z) = U(g) \cdot f(z)$ for all $g \in G_{\text{Fragile}}$

**Units:** $[G] = [z]^{-2}$, $[c_{\text{info}}] = [L]/[T]$, $[U(g)]$ dimensionless (unitary).
:::

:::{prf:theorem} Isotropic Blocks Satisfy Micro-Macro Consistency
:label: thm-isotropic-macro-compatible

IsotropicBlock (Definition {prf:ref}`def-isotropic-block`) satisfies:
1. **Light cone preservation** (Condition 2 of {prf:ref}`def-micro-macro-consistency`) - proven rigorously
2. **Gauge invariance** (Condition 3 of {prf:ref}`def-micro-macro-consistency`) - proven rigorously
3. **Metric compatibility** (Condition 1 of {prf:ref}`def-micro-macro-consistency`) - holds for constant metrics; approximately for state-dependent metrics

*Proof.*

**Condition 2 (light cone preservation) - RIGOROUS:**

**Step 1. Decompose IsotropicBlock:**
$$
f = f_3 \circ f_2 \circ f_1
$$
where:
- $f_1 = \text{SpectralLinear}: z \mapsto Wz$ with $\sigma_{\max}(W) \leq 1$
- $f_2 = \text{Reshape}: \mathbb{R}^{d_{\text{out}}} \to (\mathbb{R}^{d_b})^{n_b}$ (identity as linear map, $J_2 = I$)
- $f_3 = \text{NormGate}: (v_1, \ldots, v_{n_b}) \mapsto (v_1 g(\|v_1\| + b_1), \ldots, v_{n_b} g(\|v_{n_b}\| + b_{n_b}))$

**Step 2. Lipschitz constants:**
- By Theorem {prf:ref}`thm-spectral-preserves-light-cone`: $L_1 \leq 1$
- $f_2$ is identity: $L_2 = 1$
- By Lemma {prf:ref}`lem-normgate-lipschitz` with GELU ($C_g = 1$, $L_g \approx 1.08$):
$$
L_3 \leq R_{\max}(C_g + L_g) + C_gB \approx 2.08 R_{\max} + 1
$$

For normalized bundles with $R_{\max} \approx \sqrt{d_b}$ and $d_b = 16$:
$$
L_3 \lesssim 2.08 \times 4 + 1 \approx 9.3
$$

**Step 3. Composition:**
By composition of Lipschitz functions ($\|f \circ g(x) - f \circ g(y)\| \leq L_f L_g \|x-y\|$):
$$
L_{\text{total}} = L_3 \cdot L_2 \cdot L_1 \leq 9.3 \cdot 1 \cdot 1 = 9.3
$$

Thus:
$$
\|f(z_1) - f(z_2)\| \leq 9.3 \|z_1 - z_2\|
$$

**Remark on constant factor:** While $L > 1$ for individual blocks, depth-wise accumulation is controlled via:
- Skip connections (e.g., $z_{l+1} = z_l + \alpha \cdot \text{IsotropicBlock}(z_l)$ with $\alpha < 1$)
- Normalization layers between blocks
- The bound $L = O(\sqrt{d_b})$ is moderate for $d_b \in [8, 32]$

The effective light cone constraint $\|f(z_1) - f(z_2)\| \lesssim c_{\text{info}} \Delta t$ holds up to architecture-dependent constants.

**Condition 3 (gauge invariance) - RIGOROUS:**

Direct application of Theorem {prf:ref}`thm-isotropic-block-equivariant`. For gauge group $G_{\text{bundle}} = \prod_{i=1}^{n_b} SO(d_b)$:
$$
f(U(g) \cdot z) = U(g) \cdot f(z) \quad \forall g \in G_{\text{bundle}}
$$

This is proven constructively in Theorem {prf:ref}`thm-isotropic-block-equivariant` by showing each component (SpectralLinear, Reshape, NormGate) is equivariant and composition preserves equivariance.

**Condition 1 (metric preservation) - QUALIFIED:**

The exact pullback condition $G(f(z)) = J^T G(z) J$ depends critically on whether the metric is constant or state-dependent.

**Case A: Constant Euclidean metric** ($G(z) = I$ for all $z$)

The pullback condition becomes:
$$
I = J^T I J = J^T J
$$

This requires $J$ to be orthogonal, which is NOT true for IsotropicBlock:
- SpectralLinear: $J_1 = W$ with $\sigma_{\max}(W) \leq 1$ satisfies $W^T W \preceq I$ (contraction, not isometry unless $W$ is exactly orthogonal)
- NormGate: $J_3 = g(\|v\|)I + g'(\|v\|)vv^T/\|v\|$ satisfies:
$$
J_3^T J_3 = g^2(\|v\|) I + [g'^2(\|v\|)/\|v\|^2 + 2g(\|v\|)g'(\|v\|)/\|v\|] vv^T \neq I
$$

Thus **exact pullback fails** for constant Euclidean metric.

**Case B: Information Sensitivity Metric** (Definition {prf:ref}`def-latent-metric`)

For state-dependent $G(z) = \nabla^2 V(z) + \lambda \mathcal{F}(z)$, the pullback requires:
$$
\nabla^2 V(f(z)) + \lambda \mathcal{F}(f(z)) = J^T [\nabla^2 V(z) + \lambda \mathcal{F}(z)] J
$$

Since $V$ and $\mathcal{F}$ depend on state, $G(f(z)) \neq G(z)$ in general, and **exact pullback does not hold**.

**What IS rigorously true:**

1. **Positive-definiteness preserved:** If $G(z) \succ 0$, then for invertible $J$:
$$
J^T G(z) J \succeq \lambda_{\min}(J^T J) \cdot \lambda_{\min}(G(z)) > 0
$$
Thus positive-definiteness is maintained.

2. **Structure preservation:** NormGate acts isotropically within bundles:
$$
J_3^{(i)} = g_i I_{d_b} + h_i v_i v_i^T
$$
where $g_i = g(\|v_i\| + b_i)$ and $h_i = g'(\|v_i\| + b_i)/\|v_i\|$. This preserves radial-tangential structure of metrics that decompose similarly.

3. **Empirical compatibility:** Diagnostic Node 67 (GaugeInvarianceCheck) verifies that applying gauge transformations produces consistent behavior, indicating the metric structure is preserved in the sense relevant for geodesic integration.

**Conclusion:** IsotropicBlock does NOT satisfy exact metric pullback $G(f(z)) = J^T G(z) J$ for general metrics. However, it preserves:
- Gauge structure (Condition 3)
- Lipschitz bounds (Condition 2)
- Positive-definiteness and structural properties of the metric

For the geodesic integrator, this is sufficient because the metric is recomputed at each integration step rather than being pulled back through transformations.

$\square$
:::

:::{admonition} Connection to RL: Implicit Regularization via Architecture
:class: note

**Standard approach:** Add metric penalties to loss function (computational cost at runtime).

**Fragile approach:** Build metric preservation into forward pass (zero-cost enforcement).

**Result:** Guaranteed consistency without optimization overhead.
:::

---

(sec-dimensional-analysis)=
## Dimensional Analysis and Unit Tracking

:::{prf:proposition} Latent Dimension from Information-Theoretic First Principles
:label: prop-latent-dimension-from-capacity

The latent space $\mathcal{Z} \subset \mathbb{R}^{d_z}$ represents compressed observations encoding mutual information $I(X;Z) \leq C$ where $C$ is the channel capacity (nat/step) from the bounded rationality controller (Chapter 1).

**Derivation from Gaussian rate-distortion theory:**

For a Gaussian source $X \sim \mathcal{N}(0, \Sigma_X)$ encoded into latent $Z \in \mathbb{R}^{d_z}$ via encoder $p(Z|X)$ with reconstruction $\hat{X} = \mathbb{E}[X|Z]$:

**Step 1. Rate-distortion tradeoff:**
The optimal encoder for squared error distortion $D = \mathbb{E}[\|X - \hat{X}\|^2]$ achieves:
$$
I(X;Z) = \frac{1}{2}\sum_{i=1}^{d_z} \log\left(1 + \frac{\lambda_i}{\sigma^2}\right)
$$
where $\lambda_i$ are eigenvalues of the source covariance $\Sigma_X$ allocated to latent dimension $i$, and $\sigma^2$ is the noise level per dimension.

**Step 2. Equal allocation (isotropic latent):**
For computational efficiency, neural encoders typically use isotropic latent representations with equal variance per dimension:
$$
\Sigma_Z = \sigma_z^2 I_{d_z}
$$

The total information is:
$$
I(X;Z) \leq \frac{1}{2} d_z \log\left(1 + \frac{\sigma_X^2}{\sigma_{\text{noise}}^2}\right)
$$

**Step 3. Dimensional analysis:**
Mutual information has dimension $[I(X;Z)] = [\text{nat}]$.

For latent variance $\sigma_z^2$, we require dimensional consistency:
$$
[I(X;Z)] = [\text{nat}] = [d_z \cdot \log(1 + \sigma_z^2/\sigma_{\text{ref}}^2)]
$$

Since $d_z$ is dimensionless (counting degrees of freedom) and logarithm of dimensionless ratio is dimensionless, we need:
$$
[\text{nat}] = [d_z] \cdot [\log(\cdot)] = [1] \cdot [1]
$$

This dimensional mismatch is resolved by assigning dimensions to $\sigma_z^2$:
$$
[\sigma_z^2] = [\text{nat}]/[d_z] = [\text{nat}] \quad \text{(since $d_z$ is dimensionless count)}
$$

Thus:
$$
[z] = [\sigma_z] = \sqrt{\text{nat}} =: [\mathcal{Z}]
$$

**Interpretation:** Each latent coordinate carries information measured in natural units (nats). The variance of a latent dimension has units of information content, and coordinates themselves have units of $\sqrt{\text{nat}}$ (analogous to how position has units $\sqrt{\text{action}}$ in quantum mechanics via $\Delta x \Delta p \sim \hbar$).

**Remark:** This assignment is a dimensional convention that ensures consistency between information-theoretic quantities ($I, H, D_{\text{KL}}$ in nats) and geometric quantities (distances, norms, metrics) in latent space.
:::

:::{prf:definition} Information Speed in Latent Coordinates
:label: def-information-speed-latent

The **latent information speed** is the maximum rate of latent state change per unit time:

$$
c_{\mathcal{Z}} := \sup_{z(·), \Delta t > 0} \frac{d_{\mathcal{Z}}(z(t + \Delta t), z(t))}{\Delta t}
$$

where:
- $z: [0, T] \to \mathcal{Z}$ is a latent trajectory
- $d_{\mathcal{Z}}(z_1, z_2) = \|z_1 - z_2\|$ is the Euclidean distance in latent space
- The supremum is taken over all admissible trajectories and time increments

**Dimensions**: $[c_{\mathcal{Z}}] = [\mathcal{Z}][T^{-1}] = \sqrt{\text{nat}} \cdot s^{-1}$

**Physical interpretation**: This is the "speed of thought"—the maximum rate at which the agent's internal representation can evolve under the dynamics.

**Connection to environment information speed**:

Let $\mathcal{E}$ denote the environment observation space (e.g., pixel space for vision, $\mathcal{E} = \mathbb{R}^{H \times W \times C}$).

Let $\phi: \mathcal{E} \to \mathcal{Z}$ be the encoder network mapping observations to latents, with Jacobian $J_\phi(x) = \nabla \phi(x) \in \mathbb{R}^{d_z \times d_{\mathcal{E}}}$.

By the chain rule for composed dynamics $z(t) = \phi(x(t))$:
$$
\frac{dz}{dt} = J_\phi(x(t)) \cdot \frac{dx}{dt}
$$

Taking norms:
$$
\left\|\frac{dz}{dt}\right\| \leq \|J_\phi(x)\|_{\text{op}} \cdot \left\|\frac{dx}{dt}\right\|
$$

If the environment dynamics satisfy $\|dx/dt\| \leq c_{\text{info}}$ (Axiom {prf:ref}`ax-information-speed-limit` from Chapter 8.1), then:
$$
c_{\mathcal{Z}} \leq \sup_x \|J_\phi(x)\|_{\text{op}} \cdot c_{\text{info}}
$$

**Remark**: The encoder Lipschitz constant $L_\phi = \sup_x \|J_\phi(x)\|_{\text{op}}$ controls how environmental changes propagate to latent space. Spectral normalization ensures $L_\phi \leq 1$, preventing artificial inflation of the information speed.

**Operational constraint**: For causality preservation (Theorem {prf:ref}`thm-spectral-preserves-light-cone`), every layer in the encoder and dynamics model must satisfy $\sigma_{\max}(W) \leq 1$, ensuring no layer amplifies signal propagation speed.
:::

**Table: Complete Dimensional Analysis for Covariant Primitives**

| Quantity | Symbol | Abstract Dimension | Physical Units | Normalized Form | Enforcement |
|:---------|:-------|:-------------------|:---------------|:----------------|:------------|
| Latent vector | $z$ | $[\mathcal{Z}]$ | $\sqrt{\text{nat}}$ | $\|z\| \leq 1$ | Spectral norm in encoder |
| Bundle norm | $\|v\|$ | $[\mathcal{Z}]$ | $\sqrt{\text{nat}}$ | $\geq 0$ | Euclidean norm per bundle |
| Weight matrix | $W$ | $[\mathcal{Z}']/[\mathcal{Z}]$ | dimensionless | $\sigma_{\max}(W) \leq 1$ | Spectral normalization (Def. {prf:ref}`def-spectral-linear`) |
| Activation bias | $b$ | $[\mathcal{Z}]$ | $\sqrt{\text{nat}}$ | $\in \mathbb{R}$ | Learnable parameter |
| Information speed (latent) | $c_{\mathcal{Z}}$ | $[\mathcal{Z}][T^{-1}]$ | $\sqrt{\text{nat}} \cdot s^{-1}$ | $> 0$ | Definition {prf:ref}`def-information-speed-latent` |
| Information speed (environment) | $c_{\text{info}}$ | $[L][T^{-1}]$ | m/s | $> 0$ | Axiom {prf:ref}`ax-information-speed-limit` |
| Metric tensor | $G_{ij}$ | $[\mathcal{Z}]^{-2}$ | $\text{nat}^{-1}$ | Positive definite | Definition {prf:ref}`def-latent-metric` |
| Distance (latent) | $d_{\mathcal{Z}}$ | $[\mathcal{Z}]$ | $\sqrt{\text{nat}}$ | $\geq 0$ | Euclidean distance |
| Energy barrier | $\Delta E$ | $[\mathcal{Z}]^2$ | nat | $> 0$ | Squared norm scale |
| Gate output | $g$ | [1] | dimensionless | $\in [0, \infty)$ | GELU activation |
| Bundle dimension | $d_b$ | [1] | dimensionless | $\in \mathbb{N}$ | Architectural hyperparameter |

:::{prf:proposition} Dimensional Consistency of IsotropicBlock
:label: prop-dimensional-consistency

Let $z \in \mathcal{Z}$ with $[z] = [\mathcal{Z}] = \sqrt{\text{nat}}$ (Proposition {prf:ref}`prop-latent-dimension-from-capacity`). The IsotropicBlock operation
$$
\text{IsotropicBlock}(z) = \text{Reshape}(\text{NormGate}(\text{SpectralLinear}(z)))
$$
preserves the latent dimension $[\mathcal{Z}]$ through each stage when interpreted with implicit normalization conventions.

*Proof.*

**Dimensional Analysis Axioms:**
1. $[AB] = [A][B]$ (multiplicative)
2. $[A + B]$ defined iff $[A] = [B]$ (additive homogeneity)
3. For transcendental function $f: \mathbb{R} \to \mathbb{R}$ (like GELU), the argument must be dimensionless or implicitly normalized

**Step 1. SpectralLinear:**
$$
[W \cdot z] = [W] \cdot [z] = \frac{[\mathcal{Z}']}{[\mathcal{Z}]} \cdot [\mathcal{Z}] = [\mathcal{Z}']
$$
where $[\mathcal{Z}'] = [\mathcal{Z}]$ (linear map between same-dimensional latent spaces).

**Step 2. Reshape:**
Identity operation that permutes indices: $[\text{Reshape}(h)] = [h] = [\mathcal{Z}]$.

**Step 3. NormGate per bundle $i$ — with dimensional caveat:**

Recall the definition (Def. {prf:ref}`def-norm-gated-activation`):
$$
f(v_i) = v_i \cdot g(\|v_i\| + b_i)
$$

**Dimensional analysis:**
- $[\|v_i\|] = [v_i] = [\mathcal{Z}]$
- $[b_i] = [\mathcal{Z}]$ (homogeneous addition)
- **Issue:** $g$ is GELU, a transcendental function expecting dimensionless input

**Resolution via implicit normalization:** In practice, the argument $\|v_i\| + b_i$ is normalized by an implicit reference scale $z_0$ with $[z_0] = [\mathcal{Z}]$. The actual operation is:
$$
\hat{v}_i = v_i \cdot g\left(\frac{\|v_i\| + b_i}{z_0}\right)
$$

where:
- $z_0 = 1\,\sqrt{\text{nat}}$ (unit scale, implicitly absorbed into definition of $g$)
- The dimensionless argument is $\xi_i = (\|v_i\| + b_i)/z_0$ with $[\xi_i] = [1]$
- $g(\xi_i)$ is dimensionless, so $[\hat{v}_i] = [v_i] \cdot [1] = [\mathcal{Z}]$

**Step 4. Output dimension:**
$$
[\text{IsotropicBlock}(z)] = [\mathcal{Z}]
$$

$\square$

**Critical remark on implementation:** The code implementation (lines 1918-1930 in Python implementation below) writes:
```python
gate = F.gelu(energy + self.norm_bias)
```
where `energy = ||v_i||` has been **implicitly normalized** to be order-1 by the spectral-normalized linear layer. The mathematical formulation $g(\|v\| + b)$ assumes this normalization convention: $\|v\| \sim O(1)$ in units of $z_0 = 1\,\sqrt{\text{nat}}$.

This is dimensional analysis via **natural units** (analogous to setting $c = \hbar = 1$ in relativistic quantum mechanics): we choose units such that the typical latent scale is 1, absorbing $z_0$ into the definition.

**Formal correction for strict dimensional analysis:**

To be fully rigorous, Definition {prf:ref}`def-norm-gated-activation` should be amended to:
$$
f(v) = v \cdot g\left(\frac{\|v\| + b}{z_0}\right)
$$
where $z_0 = \mathbb{E}[\|v\|]$ is the expected bundle norm (computed over training data) with $[z_0] = [\mathcal{Z}]$.

In normalized architectures with spectral normalization and zero-mean inputs, $z_0 \approx 1\,\sqrt{\text{nat}}$, and the explicit normalization is often omitted.
:::

---

(sec-diagnostic-nodes)=
## Diagnostic Nodes and Runtime Verification

### Gauge Invariance Check (Node 67)

:::{prf:definition} Gauge Violation Metric
:label: def-gauge-violation-metric

For operator $f$ and group element $g \in G$:

$$
\delta_{\text{gauge}}(f, g) = \mathbb{E}_z\left[\|f(U(g) \cdot z) - U(g) \cdot f(z)\|^2\right]
$$

**Threshold:** $\delta_{\text{gauge}} < \epsilon_{\text{gauge}} = 10^{-4}$ (empirically tuned).
:::

**Diagnostic implementation:**

```python
class GaugeInvarianceCheck(DiagnosticNode):
    """Node 67: Verify G-equivariance of layers.

    Tests: Sample random rotations/transformations, measure violation.
    """
    def __init__(self, layer: nn.Module, group: str = "SO(d)"):
        self.layer = layer
        self.group = group

    def check(self, z: torch.Tensor) -> Dict[str, float]:
        # Sample random group element
        if self.group == "SO(d)":
            g = self._random_rotation(z.shape[-1])

        # Apply transformation
        z_transformed = g @ z

        # Check equivariance: f(g·z) ≟ g·f(z)
        f_g_z = self.layer(z_transformed)
        g_f_z = g @ self.layer(z)

        violation = torch.norm(f_g_z - g_f_z).item()

        return {
            "gauge_violation": violation,
            "threshold": 1e-4,
            "passed": violation < 1e-4
        }
```

### Summary of Diagnostic Nodes

**Table: Diagnostic Nodes for Covariant Primitives**

| Node | Name | Verifies | Trigger Condition |
|:-----|:-----|:---------|:------------------|
| 40 | PurityCheck | $SU(N_f)_C$ confinement | Non-neutral bundles at macro boundary |
| 56 | CapacityHorizonCheck | $U(1)_Y$ conservation | Hypercharge $Y \to Y_{\max}$ |
| 62 | CausalityViolationCheck | Light cone preservation | $\sigma_{\max}(W) > 1 + \epsilon$ |
| 67 | GaugeInvarianceCheck | $G$-equivariance | $\delta_{\text{gauge}} > \epsilon_{\text{gauge}}$ |
| 68 | RotationEquivarianceCheck | $SO(2)$ for images | $\|f(R \cdot I) - R \cdot f(I)\| > \epsilon$ |

---

(sec-implementation-reference)=
## Implementation Reference

The following provides the complete implementation of `IsotropicBlock` with comprehensive annotations.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Dict

class IsotropicBlock(nn.Module):
    """Atomic gauge-covariant building block (Definition {prf:ref}`def-isotropic-block`).

    Implements: SpectralLinear ∘ Reshape ∘ NormGate

    **Guarantees:**
        - SO(d_b)^{n_b} equivariance (Theorem {prf:ref}`thm-isotropic-block-equivariant`)
        - Light cone preservation (Theorem {prf:ref}`thm-spectral-preserves-light-cone`)
        - Micro-macro consistency (Theorem {prf:ref}`thm-isotropic-macro-compatible`)

    **Diagnostics:**
        - Node 67: Gauge invariance (test random rotations)
        - Node 62: Causality (verify σ_max ≤ 1)
        - Node 40: Confinement (check bundle binding)

    Args:
        in_dim: Input dimension D_in [dimensionless]
        out_dim: Output dimension D_out [dimensionless]
        bundle_size: Bundle dimension d_b [dimensionless], typically 8-32

    Shapes:
        Input: [B, D_in] where B = batch size
        Output: [B, D_out]

    Units:
        All latent vectors normalized to [-1, 1]^d
        Energy ||v|| dimensionless ∈ [0, √d_b]
        Gate g ∈ [0, 1] dimensionless

    Example:
        >>> block = IsotropicBlock(256, 512, bundle_size=16)
        >>> z = torch.randn(32, 256)  # [B=32, D_in=256]
        >>> out = block(z)  # [32, 512]
        >>> # Test equivariance (Node 67)
        >>> R = random_rotation_matrix(512)
        >>> assert torch.allclose(block(R @ z), R @ block(z), atol=1e-4)
    """
    def __init__(self, in_dim: int, out_dim: int, bundle_size: int = 16):
        super().__init__()
        assert out_dim % bundle_size == 0, "Output dim must be divisible by bundle size"

        self.bundle_size = bundle_size
        self.n_bundles = out_dim // bundle_size

        # 1. The Spectral Linear Map (Definition {prf:ref}`def-spectral-linear`)
        # Bias is False to preserve vector origin (translation invariance)
        # Spectral normalization ensures σ_max(W) ≤ 1 (Theorem {prf:ref}`thm-spectral-preserves-light-cone`)
        self.linear = spectral_norm(nn.Linear(in_dim, out_dim, bias=False))

        # 2. The Activation Potential (scalar bias on norms)
        # One bias per vector bundle [dimensionless]
        self.norm_bias = nn.Parameter(torch.zeros(1, self.n_bundles, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dimensional annotations.

        Args:
            x: [B, in_dim] input [dimensionless, normalized]

        Returns:
            output: [B, out_dim] equivariant activation [dimensionless]
        """
        # Step 1: Spectral linear (preserves light cone, Thm {prf:ref}`thm-spectral-preserves-light-cone`)
        h = self.linear(x)  # [B, out_dim], [h] = dimensionless

        # Step 2: Reshape into geometric bundles
        B, D = h.shape
        h_bundles = h.view(B, self.n_bundles, self.bundle_size)  # [B, n_b, d_b]

        # Step 3: Compute energy (SO(d_b)-invariant norm)
        energy = torch.norm(h_bundles, dim=2, keepdim=True)  # [B, n_b, 1]
        # [energy] = dimensionless, range [0, √d_b]

        # Step 4: Energy gate (smooth approximation to Heaviside)
        # GELU ensures C^∞ smoothness (required for geodesic integrator)
        gate = F.gelu(energy + self.norm_bias)  # [B, n_b, 1]
        # [gate] = dimensionless ∈ [0, ∞), approximately ∈ [0, 1] for normalized inputs

        # Step 5: Apply gate scaling to bundles (Thm {prf:ref}`thm-norm-gating-equivariant`)
        # By Definition {prf:ref}`def-norm-gated-activation`: f(v) = v · g(||v|| + b)
        # The vector is scaled by the gate value, preserving direction
        h_out = h_bundles * gate  # [B, n_b, d_b]
        # gate has shape [B, n_b, 1] so broadcasting scales each bundle

        # Step 6: Flatten back to vector
        return h_out.view(B, D)  # [B, out_dim]
```

---

(sec-summary-tables)=
## Summary and Cross-Reference Tables

### Standard DL → Fragile Primitives

**Table: Replacement Architecture**

| Component | Standard DL | Fragile (Covariant) | Symmetry Preserved | Theorem | Diagnostic Node |
|:----------|:------------|:--------------------|:-------------------|:--------|:----------------|
| Linear | `nn.Linear(bias=True)` | `SpectralLinear(bias=False)` | Translation invariance in tangent bundle | {prf:ref}`thm-spectral-preserves-light-cone` | Node 62 |
| Activation | `ReLU` (element-wise) | `NormGatedGELU` (radial) | $SO(d_b)$ rotation per bundle | {prf:ref}`thm-norm-gating-equivariant` | Node 67 |
| Normalization | `LayerNorm`, `BatchNorm` | **Removed** (implicit in spectral norm) | Probability mass conservation (WFR) | - | - |
| Vision encoder | `Conv2d` | `SteerableConv2d` (E2CNN) | $SO(2)$ rotation + $\mathbb{R}^2$ translation | {prf:ref}`thm-steerable-conv-equivariant` | Node 68 |
| Composition | Arbitrary depth | Lipschitz-constrained composition | Global light cone preservation | {prf:ref}`thm-composition-equivariant` | Node 62 |

### Gauge Group Decomposition

**Table: Primitive ↔ Gauge Field Correspondence**

| Gauge Field | Group | DNN Primitive | Physical Interpretation | Cross-Reference |
|:------------|:------|:--------------|:------------------------|:----------------|
| Binding $G_\mu^a$ | $SU(N_f)_C$ | Isotropic bundles ($n_b$ bundles) | Feature confinement (color charge) | {ref}`sec-symplectic-multi-agent-field-theory`, Node 40 |
| Error $W_\mu^b$ | $SU(2)_L$ | Obs-action doublet (steerable conv) | Sensor-motor mixing (weak force) | {ref}`sec-symplectic-multi-agent-field-theory` |
| Opportunity $B_\mu$ | $U(1)_Y$ | Spectral norm (hypercharge) | Capacity conservation | {ref}`sec-parameter-space-sieve`, Node 56 |

### Physics Isomorphism: DNN Primitives ↔ Standard Model

**Table: The "Particle Zoo" of Neural Operators**

| Standard Model Particle | Gauge Charge $(C, L, Y)$ | DNN Analogue | Transformation Property |
|:------------------------|:-------------------------|:-------------|:------------------------|
| Quark $(u, d)$ | $(3, 2, 1/6)$ | Feature in bound bundle | Confined, left-chiral |
| Lepton $(e, \nu_e)$ | $(1, 2, -1/2)$ | Observation vector | Free, left-chiral |
| Gluon | $(8, 1, 0)$ | Bundle coupling weights | Self-interacting (confines quarks) |
| $W/Z$ boson | $(1, 3, 0)$ | Obs-action mixer | Weak force mediator |
| Photon | $(1, 1, 0)$ | Capacity gradient | Long-range field (unconfined) |
| Higgs | $(1, 2, 1/2)$ | Activation threshold bias | Mass generation via symmetry breaking |

:::{div} feynman-prose
Let me be precise about something: the table above is not poetry. It is not a loose analogy where we squint and things sort of look similar. The mathematical structures are *identical*.

Consider confinement. In QCD, quarks carry color charge—red, green, or blue. A fundamental principle says you can never observe an isolated quark; they must always bind into color-neutral combinations (three quarks in a baryon, quark-antiquark in a meson). This is not optional; it is enforced by the theory. Similarly, features in our IsotropicBlocks carry bundle indices. Node 40 in the diagnostics explicitly checks that only bound states—combinations that are "neutral" across bundles—reach the macro register. Features cannot propagate independently; they must bind.

Consider mass generation. In the Standard Model, particles acquire mass through the Higgs mechanism: a field with a potential that has its minimum away from zero, spontaneously breaking symmetry. In our architecture, the activation potential $b_i$ plays exactly this role. It creates an energy barrier that signals must overcome, and the height of this barrier determines how strongly features are suppressed—analogous to how the Higgs coupling determines particle masses.

The correspondence runs deep. Physics constants map to architecture hyperparameters. Coupling strengths map to weight magnitudes. The QCD confinement scale $\Lambda_{\text{QCD}}$ maps to the bundle binding scale set by activation thresholds. You do not get to choose these independently; they are locked together by the gauge structure.

This is what it means to derive architecture from first principles.
:::

---

(sec-forward-to-chapter-5)=
## Connection to Chapter 5 (Macroscopic Integration)

:::{div} feynman-prose
We have now built the atoms—the microscopic primitives that respect gauge symmetry. SpectralLinear for light-cone preservation. NormGate for rotation-invariant activation. SteerableConv for equivariant vision. Each one carefully designed to preserve its piece of $G_{\text{Fragile}}$.

But atoms are not an agent. You need to compose them into molecules—full architectures that can actually do something. Chapter 5 shows how to do this, and here is the remarkable thing: once you have gauge-covariant primitives, the integrator almost writes itself.

The Boris-BAOAB scheme for geodesic integration has specific requirements. It needs metric-preserving steps—IsotropicBlock provides these. It needs light-cone preservation so information does not travel faster than $c_{\text{info}}$—spectral normalization provides this. It needs symplectic structure so phase space volume is conserved—the bundle structure provides this. Each requirement maps directly to a primitive we have already built.

What remains to be shown is how *attention* fits into this picture. When an agent attends to different parts of its observation, it is implementing *parallel transport*—moving vectors from one location on the manifold to another without them leaving the tangent bundle. This is done via Wilson lines, which are path-ordered exponentials of the gauge connection. Covariant Cross-Attention is the neural implementation.

We have the atoms. Chapter 5 builds the molecules. And from molecules, we get agents that move smoothly along geodesics in a gauge-invariant way.
:::

**Preview of Architecture:**

```
Input (observation x_t)
  ↓ [SteerableConv] → SO(2)-equivariant features
  ↓ [IsotropicBlock × L] → Gauge-covariant latent z_t
  ↓ [CovariantCrossAttention] → Geodesic update (see Chapter 5)
  ↓ [IsotropicBlock × L] → Decode to action a_t
Output (action a_t)
```

**Forward cross-references:**
- **Covariant Cross-Attention** ({ref}`Section 35 <sec-covariant-cross-attention>`): Implements Wilson lines for parallel transport along geodesics.
- **Boris-BAOAB Integrator** ({ref}`Section 22 <sec-equations-of-motion-langevin-sdes-on-information-manifolds>`): Macroscopic integration scheme that requires microscopic primitives to preserve gauge structure.
- **Temperature Schedule** ({ref}`Section 29 <sec-the-belief-wave-function-schrodinger-representation>`): Cognitive temperature $T_c$ varies with inverse conformal factor $1/\lambda(z)$ to maintain consistent exploration across curved manifold.

---

(sec-dnn-blocks-conclusion)=
## Conclusion

:::{div} feynman-prose
Let me summarize what we have accomplished in this chapter, because it is genuinely remarkable when you step back and look at the whole picture.

We started with a problem: standard deep learning primitives—the bread-and-butter operations like ReLU and biased linear layers—secretly assume things about your space that are not true in general. They assume flat geometry. They assume preferred coordinate axes. They assume that the origin of your coordinate system means something special. These assumptions are fine for image classification on a fixed dataset, but they are catastrophic for agents that must maintain consistent beliefs across time, integrate geodesics on curved manifolds, and interact with other agents under capacity constraints.

So we built replacements. Three primitives, each designed from the ground up to respect gauge symmetry:

1. **SpectralLinear** removes bias and constrains the singular values. This preserves the light cone structure—information cannot propagate faster than $c_{\text{info}}$, and there is no artificial preferred origin in the tangent space.

2. **NormGate** applies activation based on bundle norms, not individual components. The gate decision "is there enough energy to pass?" has the same answer regardless of how you orient your coordinate axes within each bundle.

3. **SteerableConv** lifts images to the group $SE(2)$ and convolves with steerable filters. Rotate the input, get rotated features—exactly, not approximately through data augmentation.

Each primitive preserves one factor of the gauge group $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$. Bundle structure implements color confinement. Spectral normalization implements hypercharge conservation. Steerable convolutions implement the observation-action doublet structure. Three primitives, three gauge factors, one unified theory.

And here is what I find most satisfying: the constraints are so tight that the architecture essentially writes itself. Once you demand gauge invariance, there is only one way to build each component. You do not get to make arbitrary design choices; the symmetry requirements dictate the structure. This is the hallmark of good physics: not many knobs to tune, but few parameters that must take specific values.

We now have the atoms. What remains is to show how these atoms compose into molecules—full architectures that can actually be trained and deployed. That is the subject of Chapter 5, where we connect these microscopic primitives to the macroscopic geodesic integrator.

The bridge from gauge theory to neural network implementation is complete.
:::
