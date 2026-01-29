# Agent reference (no proofs)

## TLDR

- This is a **proof-free glossary** of every labeled definition/axiom/theorem in Volume 1 (`docs/source/1_agent/`).
- Use it as a fast lookup when reading lectures; each item links back to its original context via `{prf:ref}`.
- It is intentionally redundant and not narrative: the lectures explain *why*; this file records *what was stated*.

This file is an **exhaustive, proof-free** extraction of every labeled mathematical object
(`definition`, `axiom`, `assumption`, `lemma`, `proposition`, `theorem`, `corollary`,
`conjecture`, `remark`, `algorithm`) in `docs/source/1_agent/`.

Each entry begins with a link to its original statement via `{prf:ref}` and then repeats the
statement verbatim (minus proofs and formatting-only fields like `:label:` / `:class:`).

## Definitions

({prf:ref}`def-bounded-rationality-controller`) *definition* — **Bounded-Rationality Controller**

The agent is a controller with internal state

$$
Z_t := (K_t, Z_{n,t}, Z_{\mathrm{tex},t}) \in \mathcal{Z}=\mathcal{K}\times\mathcal{Z}_n\times\mathcal{Z}_{\mathrm{tex}},

$$
and internal components (Encoder/Shutter, World Model, Critic, Policy). Its evolution is driven only by the observable interaction stream at the interface (observations/feedback) and by its own outgoing control signals (actions).

({prf:ref}`def-boundary-markov-blanket`) *definition* — **Boundary / Markov Blanket**

The boundary variables at time $t$ are the interface tuple

$$
B_t := (x_t,\ r_t,\ d_t,\ \iota_t,\ a_t),

$$
where:
- $x_t\in\mathcal{X}$ is the observation (input sample),
- $r_t\in\mathbb{R}$ is reward/utility (scalar feedback; equivalently negative instantaneous cost),
- $d_t\in\{0,1\}$ is termination (absorbing event / task boundary),
- $\iota_t$ denotes any additional side channels (costs, constraints, termination reasons, privileged signals),
- $a_t\in\mathcal{A}$ is action (control signal sent outward).

({prf:ref}`def-environment-as-generative-process`) *definition* — **Environment as Generative Process**

The "environment" is the conditional law of future interface signals given past interface history. Concretely it is a (possibly history-dependent) kernel on incoming boundary signals conditional on outgoing control:

$$
P_{\partial}(x_{t+1}, r_t, d_t, \iota_{t+1}\mid x_{\le t}, a_{\le t}).

$$
In the Markov case this reduces to the familiar RL kernel

$$
P_{\partial}(x_{t+1}, r_t, d_t, \iota_{t+1}\mid x_t, a_t),

$$
but the **interpretation changes**: $P_{\partial}$ is not "a dataset generator"; it is the **input-output law** that the controller must cope with under partial observability and model mismatch.

This is the categorical move: we do not assume access to the environment's latent variables; we work only with the **law over observable interface variables**.

({prf:ref}`def-agent-symmetry-group-operational`) *definition* — **Agent symmetry group; operational**

Let:
- $G_{\text{obj}}$ be an **objective/feedback gauge** acting on scalar feedback signals (e.g., change of units or baseline shift). A common choice is the positive affine group

  $$
  G_{\text{obj}} := \{(a,b): a>0,\ r\mapsto ar+b\}.

  $$
  (If representing value as a unit-norm phase variable, one may instead use $U(1)$; {ref}`Section 3.3 <sec-defect-functionals-implementing-regulation>`.C treats the real-valued case via projective heads.)
- $G_{\text{spatial}}$ be an **observation gauge** acting on raw observations $x$ (e.g., pose/translation/rotation; choose $SE(3)$, $SE(2)$, $\mathrm{Sim}(2)$, or a task-specific subgroup depending on sensors).
- $S_{|\mathcal{K}|}$ be the **symbol-permutation symmetry** of the discrete macro register: relabeling code indices is unobservable if downstream components depend only on embeddings $\{e_k\}$.
- $\mathrm{Symp}(2n,\mathbb{R})$ be an optional **phase-space symmetry** acting on canonical latent coordinates $z=(q,p)\in\mathbb{R}^{2n}$ when the world model is parameterized as a symplectic/Hamiltonian system ({ref}`Section 3.3 <sec-defect-functionals-implementing-regulation>`.B).

The (candidate) total symmetry group is the direct product

$$
\mathcal{G}_{\mathbb{A}}
:=
G_{\text{obj}}
\times
G_{\text{spatial}}
\times
S_{|\mathcal{K}|}
\times
\mathrm{Symp}(2n,\mathbb{R}).

$$
**Internal vs. external symmetries.**
- **Internal (objective) gauge:** transformations of the scalar feedback scale/offset (and any potentials) that should not qualitatively change the policy update direction.
- **External (observation) gauge:** transformations of the input stream that change *pose* but not *identity*.

**Principle of covariance (engineering requirement).** The internal maps of the agent should be invariant/equivariant under $\mathcal{G}_{\mathbb{A}}$ in the following typed sense:
- **Shutter $E$**: canonicalize or quotient $G_{\text{spatial}}$ before discretization, so the macro register is approximately invariant:

  $$
  K(x)\approx K(g\cdot x)\quad (g\in G_{\text{spatial}}),

  $$
  while $z_n$ carries structured nuisance parameters (pose/basis/disturbance coordinates) and $z_{\mathrm{tex}}$ carries reconstruction-only texture ({ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>`, {ref}`Section 3.3 <sec-defect-functionals-implementing-regulation>`.A).
- **World model $S$ and policy $\pi$:** be covariant to symbol permutations $S_{|\mathcal{K}|}$ by treating $K$ only through its embedding $e_K$ (not the integer label) and by using permutation-invariant diagnostics.
- **Critic/value and dual variables:** enforce stability and constraint satisfaction in a way that is robust to re-scaling/offset of the scalar feedback ({ref}`Section 3.3 <sec-defect-functionals-implementing-regulation>`.C, {ref}`Section 3.5 <sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration>`).

These are *requirements on representations and interfaces*, not philosophical claims: if an invariance is not enforced, the corresponding failure modes (symmetry blindness, brittle scaling, uncontrolled drift) become more likely and harder to debug.

({prf:ref}`def-state-space-sensitivity-metric`) *definition* — **State-Space Sensitivity Metric**

The **state-space sensitivity metric** $G_{ij}$ at a point $z$ in the latent space is defined as the Hessian of the value function:

$$
G_{ij} = \frac{\partial^2 V}{\partial z_i \partial z_j} = \text{Hess}(V)

$$

Units: $[G_{ij}]=\mathrm{nat}\,[z]^{-2}$ if $z$ is measured in units $[z]$.

({prf:ref}`def-complete-latent-space-metric`) *definition* — **Complete Latent Space Metric**

The complete state-space sensitivity metric on $\mathcal{Z}$ is defined as:

$$
G_{ij}(z) = \underbrace{\frac{\partial^2 V(z)}{\partial z_i \partial z_j}}_{\text{Hessian (value curvature)}} + \lambda \underbrace{\mathbb{E}_{a \sim \pi} \left[ \frac{\partial \log \pi(a|z)}{\partial z_i} \frac{\partial \log \pi(a|z)}{\partial z_j} \right]}_{\text{Fisher (control sensitivity)}}

$$

Units: the Fisher term has units $[z]^{-2}$; therefore $\lambda$ carries the same units as $V$ (here $\mathrm{nat}$) so both addends match.

({prf:ref}`def-causal-enclosure-condition`) *definition* — **Causal Enclosure Condition**

**Causal Enclosure Condition (Markov sufficiency).** With the nuisance/texture split ({ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>`), let $(K_t, Z_{n,t}, Z_{\mathrm{tex},t}, A_t)$ be the internal state/action process and define the macrostate $K_t:=\Pi(Z_t)$ (projection to the discrete register). The macro-model requirement is the conditional independence

$$
K_{t+1}\ \perp\!\!\!\perp\ (Z_{n,t}, Z_{\mathrm{tex},t})\ \big|\ (K_t,A_t),

$$
equivalently the vanishing of a conditional mutual information:

$$
I(K_{t+1};Z_{n,t},Z_{\mathrm{tex},t}\mid K_t,A_t)=0.

$$

({prf:ref}`def-closure-defect`) *definition* — **Closure Defect**

**Closure Defect (kernel-level).** Write the micro-dynamics as a Markov kernel $P(dz'\mid z,a)$ and let $P_\Pi(\cdot\mid z,a)$ be the pushforward kernel on $\mathcal{K}$ induced by $\Pi$. A learned macro-dynamics kernel $\bar{P}(\cdot\mid k,a)$ is enclosure-correct iff

$$
P_\Pi(\cdot\mid z,a)=\bar{P}(\cdot\mid \Pi(z),a)
\quad\text{for }P\text{-a.e. }z.

$$
A canonical defect functional is the expected divergence

$$
\delta_{\text{CE}}
:=
\mathbb{E}_{z,a}\Big[D_{\mathrm{KL}}\big(P_\Pi(\cdot\mid z,a)\ \Vert\ \bar{P}(\cdot\mid \Pi(z),a)\big)\Big].

$$

({prf:ref}`def-local-conditioning-scale`) *definition* — **Local Conditioning Scale**

Let $(\mathcal{Z}, G)$ be the Riemannian latent manifold. Define a local scale parameter $\Theta: \mathcal{Z} \to \mathbb{R}^+$ as the trace of the inverse metric:

$$
\Theta(z) := \frac{1}{d} \operatorname{Tr}\left( G^{-1}(z) \right)

$$
where $d = \dim(\mathcal{Z})$. The corresponding **precision / coupling coefficient** is $\beta(z) = [\Theta(z)]^{-1}$.
Units: if $z$ carries units $[z]$, then $[G]=\mathrm{nat}\,[z]^{-2}$ implies $[\Theta]=[z]^2/\mathrm{nat}$ and $[\beta]=\mathrm{nat}/[z]^2$ (dimensionless when $z$ is normalized).

({prf:ref}`def-entropy-regularized-objective-functional`) *definition* — **Entropy-Regularized Objective Functional**

Let $d\mu_G:=\sqrt{|G|}\,dz$ be the Riemannian volume form on $\mathcal{Z}$ and let $p(z)$ be a probability density with respect to $d\mu_G$. For a (dimensionless) trade-off coefficient $\tau\ge 0$, define

$$
\mathcal{F}[p,\pi]
:=
\int_{\mathcal{Z}} p(z)\Big(V(z) - \tau\,H(\pi(\cdot\mid z))\Big)\,d\mu_G,

$$
where $H(\pi(\cdot\mid z)) := -\mathbb{E}_{a\sim \pi(\cdot\mid z)}[\log \pi(a\mid z)]$ is the per-state policy entropy (in nats). Because $V$ and $H$ are measured in nats ({ref}`Section 1.2 <sec-units-and-dimensional-conventions>`), $\tau$ is dimensionless.

({prf:ref}`def-belief-density`) *definition* — **Belief Density**

Let $p(z,s)\ge 0$ be a density with respect to $d\mu_G$ representing the agent's belief (or belief-weight) over latent coordinates. In closed-system idealizations one may impose $\int_{\mathcal{Z}}p(z,s)\,d\mu_G=1$; in open-system implementations with explicit projections/reweightings we track the unnormalized mass and renormalize when needed ({ref}`Section 11 <sec-intrinsic-motivation-maximum-entropy-exploration>`).

({prf:ref}`def-transport-field`) *definition* — **Transport Field**

Let $v\in\Gamma(T\mathcal{Z})$ be a vector field describing the instantaneous transport of belief mass on $\mathcal{Z}$. In a value-gradient-flow idealization (used only for intuition), one may take

$$
v^i(z) := -G^{ij}(z)\frac{\partial V}{\partial z^j},

$$
so transport points in the direction of decreasing $V$ (Riemannian steepest descent). Units: if computation time is measured in solver units, then $[v]=[z]/\mathrm{solver\ time}$ (map to $\mathrm{step}$ using the $t \leftrightarrow s$ budget in {ref}`Section 1.3 <sec-the-chronology-temporal-distinctions>`).

({prf:ref}`def-source-residual`) *definition* — **Source Residual**

In general, belief evolution may include additional update effects (e.g. approximation error, off-manifold steps, or explicit projection/reweighting). We collect these into a residual/source term $\sigma(z,s)$:

$$
\frac{\partial p}{\partial s} + \operatorname{div}_G(p v) = \sigma

$$
Interpreting $\sigma$:
1. If $\sigma>0$ on a region, belief mass is being created there beyond pure transport; this indicates an **ungrounded internal update** relative to the transport model.
2. If $\sigma<0$, belief mass is being removed beyond pure transport (aggressive forgetting or projection).
3. Integrating over any measurable region $U\subseteq\mathcal{Z}$ and applying the divergence theorem yields the exact mass balance

   $$
   \frac{d}{ds}\int_U p\,d\mu_G
   =
   -\oint_{\partial U}\langle p v,n\rangle\,dA_G
   +\int_U \sigma\,d\mu_G.

   $$
   For {math}`U=\mathcal{Z}` this relates net mass change to boundary flux and the integrated residual.

({prf:ref}`def-observation-inflow-form`) *definition* — **Observation Inflow Form**

Let $j \in \Omega^{d-1}(\partial \mathcal{Z})$ be the **observation inflow form**. This form represents the rate of information entering the model through the interface.

({prf:ref}`def-attentive-routing-law`) *definition* — **Attentive Routing Law**

$$
w_i(x) := \frac{\exp\left(\frac{\langle q_i, k(x) \rangle}{\sqrt{d}}\right)}{\sum_{j=1}^{N_c} \exp\left(\frac{\langle q_j, k(x) \rangle}{\sqrt{d}}\right)}

$$
This mechanism is **permutation invariant**: shuffling the memory order of the queries $\{q_i\}$ merely shuffles the output indices without changing the underlying topology or geometry.

({prf:ref}`def-the-macro-state-tree`) *definition* — **The Macro-State Tree**

Let $\mathcal{T}$ be a rooted tree representing the hierarchical partition of the state space.

1. The **root** represents the entire observation space $\mathcal{X}$.
2. **Level 1 nodes** correspond to charts $K_{\text{chart}} \in \{1, \dots, N_c\}$.
3. **Level 2 nodes** correspond to codes $K_{\text{code}} \in \{1, \dots, N_v\}$ within a chart.
4. Edges represent the containment relationship (refinement of the partition).

Equip the vertex set $V(\mathcal{T})$ with the graph metric $d_{\mathcal{T}}$ (shortest path length).

({prf:ref}`def-the-local-fibre-structure`) *definition* — **The Local Fibre Structure**

We model the latent space $\mathcal{Z}$ as a disjoint union of fibres over the discrete index set $\mathcal{K}$:

$$
\mathcal{Z} = \bigsqcup_{k \in \mathcal{K}} \mathcal{Z}_n^{(k)}, \qquad \mathcal{Z}_n^{(k)} \cong \mathbb{R}^{d_n}.

$$
For each macro-symbol $k \in \mathcal{K}$, the fibre $\mathcal{Z}_n^{(k)}$ represents the **structured nuisance** space (local pose/basis coordinates).

The interpolation of this discrete structure into a continuous manifold is achieved by the Attentive Atlas ({ref}`Section 7.8 <sec-tier-the-attentive-atlas>`), which provides soft transition functions (partitions of unity) $\{w_i(x)\}$ that interpolate between fibres in overlap regions.

({prf:ref}`def-the-latent-metric-tensor`) *definition* — **The Latent Metric Tensor**

Working in the upper half-space model where depth $\rho \in [0, \infty)$ corresponds to $y = e^{-\rho}$, the metric $ds^2$ on the global latent space $\mathcal{Z}$ takes the form:

$$
ds^2 = d\rho^2 + d\sigma_{\mathcal{K}}^2 + e^{-2\rho} \|dz_n\|^2

$$
where:

* $\rho$ is the resolution depth (hierarchy level), with $\rho = 0$ at the root and $\rho \to \infty$ at the boundary.
* $d\sigma_{\mathcal{K}}^2$ is the (discrete) metric on tree branches at fixed depth—operationally, it counts the number of chart/code transitions.
* $\|dz_n\|^2$ is the Euclidean metric on the structured nuisance $z_n$.
* The factor $e^{-2\rho}$ indicates that as resolution increases (deeper in the tree), the effective magnitude of nuisance variations shrinks exponentially relative to the macroscopic decision branches.

**Rigorous Interpretation of $z_n$:**
The structured nuisance $z_n$ is not stochastic noise; it is the **tangent space coordinate** on the horosphere (surface of constant depth $\rho$) determined by the active macro-symbol $K$. Horospheres in hyperbolic space are intrinsically flat (zero curvature), which is why local linear control theory (LTI approximations) applies within a single chart, even though the global geometry is hyperbolic.

({prf:ref}`def-the-peeling-step`) *definition* — **The Peeling Step**

At layer $\ell$, the input signal $x^{(\ell)}$ is decomposed into a structural component (the **Effective Theory** at scale $\ell$) and a residual component (the **High-Frequency Fluctuations**).

1. **Analysis (Encoding):** The block identifies the macro-symbol $K^{(\ell)}$ and structured nuisance $z_n^{(\ell)}$ that best approximate $x^{(\ell)}$:

$$
(K^{(\ell)}, z_n^{(\ell)}) = \mathcal{E}^{(\ell)}(x^{(\ell)})

$$
2. **Synthesis (Effective Reconstruction):** The block generates the signal explained by this structure:

$$
\hat{x}^{(\ell)} = \mathcal{D}^{(\ell)}(K^{(\ell)}, z_n^{(\ell)})

$$
3. **Residual Computation (Texture Extraction):** The unexplained signal is isolated:

$$
z_{\mathrm{tex}}^{(\ell)} = x^{(\ell)} - \hat{x}^{(\ell)}

$$

({prf:ref}`def-the-rescaling-operator-renormalization`) *definition* — **The Rescaling Operator / Renormalization**

To prevent signal decay (vanishing activations) without using skip connections, we explicitly renormalize the residual to unit variance before passing it to the next scale:

$$
x^{(\ell+1)} = \frac{z_{\mathrm{tex}}^{(\ell)}}{\sigma^{(\ell)} + \epsilon}, \qquad \sigma^{(\ell)} = \sqrt{\mathrm{Var}(z_{\mathrm{tex}}^{(\ell)}) + \epsilon}

$$
The scalar $\sigma^{(\ell)}$ is stored as a state variable (the **scale factor**) for the decoding pass.

({prf:ref}`def-total-reconstruction`) *definition* — **Total Reconstruction**

The original signal is reconstructed by summing the contributions of all scales, modulated by their respective scale factors. Define $\Pi^{(\ell)} := \prod_{j=0}^{\ell-1} \sigma^{(j)}$ with the convention $\Pi^{(0)} = 1$ (empty product). Then:

$$
\hat{x} = \sum_{\ell=0}^{L-1} \Pi^{(\ell)} \cdot \hat{x}^{(\ell)} + \Pi^{(L)} \cdot x^{(L)}

$$

({prf:ref}`def-factorized-jump-operator`) *definition* — **Factorized Jump Operator**

For each chart $i$, define:
- An **encoder** $B_i: \mathbb{R}^{d_n} \to \mathbb{R}^r$ that lifts local coordinates to the global tangent space.
- A **decoder** $A_j: \mathbb{R}^r \to \mathbb{R}^{d_n}$ that projects from the global tangent space to chart $j$'s coordinates.
- Bias terms $c_i \in \mathbb{R}^r$ and $d_j \in \mathbb{R}^{d_n}$.

The transition $L_{i \to j}$ is then:

$$
L_{i \to j}(z) = A_j(B_i z + c_i) + d_j

$$

({prf:ref}`def-overlap-consistency-loss`) *definition* — **Overlap Consistency Loss**

For a pair of charts $(i, j)$ with non-empty overlap, define the pairwise consistency loss as:

$$
\mathcal{L}_{\text{jump}}^{(i,j)} = \mathbb{E}_{x : w_i(x) > \tau, \, w_j(x) > \tau} \left[ \left\| z_n^{(j)} - L_{i \to j}(z_n^{(i)}) \right\|^2 \right]

$$
where $z_n^{(i)}$ and $z_n^{(j)}$ are the nuisance coordinates computed independently by chart $i$ and chart $j$'s encoders, and $w_i(x), w_j(x)$ are the soft router weights. The total overlap consistency loss sums over all overlapping pairs:

$$
\mathcal{L}_{\text{jump}} = \sum_{i < j} \mathcal{L}_{\text{jump}}^{(i,j)}

$$
**Intuition:** If the encoder correctly identifies that $x$ belongs to both charts, then applying the jump operator to chart $i$'s encoding should yield chart $j$'s encoding. Any discrepancy indicates that the transition functions are inconsistent with the actual data manifold.

**Implementation Details:**

1. **Overlap Detection:** A point $x$ is in the overlap $U_i \cap U_j$ if both router weights exceed a threshold:

   $$
   \mathbf{1}[x \in U_i \cap U_j] \approx \mathbf{1}[w_i(x) > \tau] \cdot \mathbf{1}[w_j(x) > \tau]

   $$
   With soft routers ({ref}`Section 7.8 <sec-tier-the-attentive-atlas>`), we use the product $w_i(x) \cdot w_j(x)$ as a soft indicator.

2. **Sampling Overlaps:** Computing all $K^2$ pairs is expensive. We sample:
   - The top-2 charts per point (from router weights).
   - Random chart pairs with probability proportional to their co-activation frequency.

3. **Symmetry Penalty (Optional):** To encourage approximate invertibility:

   $$
   \mathcal{L}_{\text{inv}} = \mathbb{E}_{x, i, j} \left[ \left\| z_n^{(i)} - L_{j \to i}(L_{i \to j}(z_n^{(i)})) \right\|^2 \right]

   $$

({prf:ref}`def-three-channel-latent`) *definition* — **The Three-Channel Latent Decomposition**

The disentangled agent's internal state at time $t$ decomposes as:

$$
Z_t = (K_t, z_{n,t}, z_{\mathrm{tex},t})

$$

where each component serves a distinct representational role:

1. **$K_t \in \mathcal{K}$ (The Macro Symbol / Law Register):** A discrete code index from a finite codebook. This is the low-frequency, causal, predictable core of the state. For downstream continuous networks, we use its embedding $z_{\text{macro}}:=e_{K_t}\in\mathbb{R}^{d_m}$, but the *information-carrying* object is the discrete symbol $K_t$.

2. **$z_{n,t} \in \mathbb{R}^{d_n}$ (Structured Nuisance / Gauge Residual):** A continuous latent for pose/basis/disturbance coordinates. This is *not* "noise": it is structured variation that may be needed for actuation and for explaining boundary-driven deviations, but it must remain disentangled from macro identity.

3. **$z_{\mathrm{tex},t} \in \mathbb{R}^{d_{\mathrm{tex}}}$ (Texture Residual):** A high-rate continuous latent for reconstruction detail. Texture is treated as an **emission residual**: it may be needed to reconstruct $x_t$ but must not be required for macro closure or for control.

({prf:ref}`def-causal-enclosure`) *definition* — **The Golden Rule of Causal Enclosure**

The macro symbol must satisfy the **causal enclosure** property:

$$
P(K_{t+1}\mid K_t,a_t)\ \text{is sharply concentrated (ideally deterministic)}

$$

and the **texture independence** property:

$$
I(K_{t+1};Z_{\mathrm{tex},t}\mid K_t,a_t)=0.

$$

Optionally, in the strongest form, nuisance independence also holds:

$$
I(K_{t+1};Z_{n,t}\mid K_t,a_t)=0.

$$

That is: nuisance should not be needed to predict the next macro symbol once action is accounted for.

({prf:ref}`def-total-disentangled-loss`) *definition* — **The Total Disentangled Loss**

The compound loss for training the split-latent agent is:

$$
\mathcal{L}_{\text{total}}
=
\lambda_{\text{recon}}\mathcal{L}_{\text{recon}}
\;+\;\lambda_{\text{vq}}\mathcal{L}_{\text{vq}}
\;+\;\lambda_{\text{closure}}\mathcal{L}_{\text{closure}}
\;+\;\lambda_{\text{slowness}}\mathcal{L}_{\text{slowness}}
\;+\;\lambda_{\text{nuis}}\mathcal{L}_{\text{nuis-KL}}
\;+\;\lambda_{\text{tex}}\mathcal{L}_{\text{tex-KL}}

$$

where:
- $\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|^2$ is the reconstruction loss
- $\mathcal{L}_{\text{vq}}$ is the codebook + commitment loss from vector quantization
- $\mathcal{L}_{\text{closure}} = -\log p_\psi(K_{t+1}\mid K_t,a_t)$ is the cross-entropy estimating $H(K_{t+1}\mid K_t,a_t)$
- $\mathcal{L}_{\text{slowness}} = \|e_{K_t} - e_{K_{t-1}}\|^2$ penalizes rapid symbol changes
- $\mathcal{L}_{\text{nuis-KL}} = D_{\mathrm{KL}}(q(z_n|x) \| \mathcal{N}(0,I))$ regularizes nuisance
- $\mathcal{L}_{\text{tex-KL}} = D_{\mathrm{KL}}(q(z_{\text{tex}}|x) \| \mathcal{N}(0,I))$ regularizes texture

({prf:ref}`def-closure-ratio`) *definition* — **Closure Ratio**

The **Closure Ratio** is defined as:

$$
\text{Closure Ratio}
=
\frac{\mathbb{E}\big[-\log p_\psi(K_{t+1}\mid K_t,a_t)\big]}{\mathbb{E}\big[-\log p_{\text{base}}(K_{t+1})\big]}.

$$

With $p_{\text{base}}$ chosen as the marginal symbol model, the numerator estimates $H(K_{t+1}\mid K_t,a_t)$ and the denominator estimates $H(K_{t+1})$, so the *gap* is a direct estimate of predictive information $I(K_{t+1};K_t,a_t)$.

({prf:ref}`def-hierarchical-latent`) *definition* — **Hierarchical Multi-Scale Latent Decomposition**

A hierarchical split-latent state has the form:

$$
Z_t = (K_t^{(1)}, K_t^{(2)}, \ldots, K_t^{(L)}, z_{\mu,t}),
\qquad
z_{\text{macro}}^{(i)} := e^{(i)}_{K_t^{(i)}}\in\mathbb{R}^{d_i}.

$$

Where $K^{(1)}$ is the slowest (most abstract) level and $K^{(L)}$ is the fastest (most detailed) macro symbol. The micro residual $z_{\mu,t}$ handles reconstruction detail below the finest macro scale.

({prf:ref}`def-instantaneous-objective`) *definition* — **Instantaneous Regularized Objective**

Define an instantaneous regularized objective:

$$
F_t
:=
V(Z_t)
+ \beta_K\big(-\log p_\psi(K_t)\big)
+ \beta_n D_{\mathrm{KL}}\!\left(q(z_{n,t}\mid x_t)\ \Vert\ p(z_n)\right)
+ \beta_{\mathrm{tex}} D_{\mathrm{KL}}\!\left(q(z_{\mathrm{tex},t}\mid x_t)\ \Vert\ p(z_{\mathrm{tex}})\right)
+ T_c D_{\mathrm{KL}}\!\left(\pi(\cdot\mid K_t)\ \Vert\ \pi_0(\cdot\mid K_t)\right),

$$

where {math}`Z_t=(K_t,z_{n,t},z_{\mathrm{tex},t})` and all terms are measured in nats.

({prf:ref}`def-macro-path-distribution`) *definition* — **Macro Path Distribution**

Fix a horizon $H\in\mathbb{N}$ and a (possibly stochastic) policy $\pi(a\mid k)$. The induced distribution over length-$H$ macro trajectories

$$
\xi := (K_{t+1},\dots,K_{t+H}) \in \mathcal{K}^H

$$
conditioned on $K_t=k$ is

$$
P_\pi(\xi\mid k)
:=
\sum_{a_{t:t+H-1}}
\prod_{h=0}^{H-1}\pi(a_{t+h}\mid K_{t+h})\ \bar{P}(K_{t+h+1}\mid K_{t+h},a_{t+h}).

$$
(For continuous $\mathcal{A}$, replace the sum by an integral with respect to the action reference measure.)

({prf:ref}`def-causal-path-entropy`) *definition* — **Causal Path Entropy**

The causal path entropy at $(k,H)$ under $\pi$ is the Shannon entropy of the path distribution:

$$
S_c(k,H;\pi) := H\!\left(P_\pi(\cdot\mid k)\right)
= -\sum_{\xi\in\mathcal{K}^H} P_\pi(\xi\mid k)\log P_\pi(\xi\mid k).

$$
This quantity is well-typed precisely because the macro register is discrete: there is no differential-entropy ambiguity.

({prf:ref}`def-exploration-gradient-metric-form`) *definition* — **Exploration Gradient, metric form**

Let $z_{\text{macro}}=e_k\in\mathbb{R}^{d_m}$ denote the code embedding of $k$ ({ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>`), and let $G$ be the relevant metric on the macro chart ({ref}`Section 2.5 <sec-second-order-sensitivity-value-defines-a-local-metric>`). Define the exploration gradient as the metric gradient of path entropy:

$$
\mathbf{g}_{\text{expl}}(e_k) := T_c\ \nabla_G S_c(k,H;\pi),

$$
where $T_c>0$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`). Operationally, gradients are taken through the continuous pre-quantization coordinates (straight-through VQ estimator); in the strictly symbolic limit, the gradient becomes a discrete preference ordering induced by $S_c(k,H;\pi)$.

**Interpretation (Exploration / Reachability).** $S_c(k,H;\pi)$ measures how many future macro-trajectories remain plausible from $k$ under $\pi$ and $\bar{P}$. Increasing $S_c$ preserves **future reachability**: the agent stays inside regions with many reachable, non-absorbing macrostates.

({prf:ref}`def-maxent-rl-objective-on-macrostates`) *definition* — **MaxEnt RL objective on macrostates**

Let $\mathcal{R}(k,a)$ be an instantaneous reward/cost-rate term ({ref}`Section 1.1.2 <sec-re-typing-standard-rl-primitives-as-interface-signals>`, {ref}`Section 2.7 <sec-the-hjb-correspondence>`) and let $\gamma\in(0,1)$ be the discount factor (dimensionless). The maximum-entropy objective is

$$
J_{T_c}(\pi)
:=
\mathbb{E}_\pi\left[\sum_{t\ge 0}\gamma^t\left(\mathcal{R}(K_t,A_t) + T_c\,\mathcal{H}(\pi(\cdot\mid K_t))\right)\right],

$$
where $\mathcal{H}$ is Shannon entropy. This is the standard "utility + entropy regularization" objective.

**Regimes.**
- $T_c\to 0$: $\pi$ collapses toward determinism; behavior can be brittle under distribution shift.
- $T_c\to\infty$: $\pi$ approaches maximal entropy; behavior becomes overly random and may degrade grounding (BarrierScat).
- The useful regime is intermediate: enough entropy to remain robust, enough utility to remain directed.

({prf:ref}`def-causal-path-space`) *definition* — **Causal Path Space**

For a macrostate $k\in\mathcal{K}$ and horizon $H$, define the future macro path space

$$
\Gamma_H(k) := \mathcal{K}^H.

$$

({prf:ref}`def-path-probability`) *definition* — **Path Probability**

$P_\pi(\xi\mid k)$ is the induced path probability from Definition 10.1.1.

({prf:ref}`def-causal-entropy`) *definition* — **Causal Entropy**

$S_c(k,H;\pi)$ is the Shannon entropy of $P_\pi(\cdot\mid k)$ (Definition 10.1.2).

({prf:ref}`def-exploration-gradient-covariant-form`) *definition* — **Exploration gradient, covariant form**

On a macro chart with metric $G$ ({ref}`Section 2.5 <sec-second-order-sensitivity-value-defines-a-local-metric>`),

$$
\mathbf{g}_{\text{expl}}(e_k) := T_c\,\nabla_G S_c(k,H;\pi).

$$

({prf:ref}`def-belief-operator`) *definition* — **Belief operator**

Let $\varrho_t\in\mathbb{C}^{d\times d}$ satisfy $\varrho_t\succeq 0$ and $\mathrm{Tr}(\varrho_t)=1$. Diagonal $\varrho_t$ reduces to a classical probability vector; non-diagonal terms can be used to encode correlations/uncertainty structure in a learned feature basis.

({prf:ref}`def-gksl-generator`) *definition* — **GKSL generator**

A continuous-time, Markovian, completely-positive trace-preserving (CPTP) evolution has a generator of the Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) form {cite}`gorini1976completely,lindblad1976generators`:

$$
\frac{d\varrho}{ds}
=
\underbrace{-i[H,\varrho]}_{\text{conservative drift}}
\;+\;
\underbrace{\sum_{j} \gamma_j\left(L_j\varrho L_j^\dagger-\frac12\{L_j^\dagger L_j,\varrho\}\right)}_{\text{dissipative update}},

$$
where {math}`H=H^\dagger` is Hermitian, {math}`\gamma_j\ge 0` are rates, and {math}`\{L_j\}` are (learned) operators.

**Operational interpretation (within this document).**
- The commutator term is a structured way to represent **reversible internal prediction** (it preserves $\mathrm{Tr}(\varrho)$ and the spectrum of $\varrho$).
- The dissipator is a structured way to represent **irreversible assimilation / disturbance** while preserving positivity and trace.

This is a modeling choice, not a claim about literal quantum physics: it is used here purely as a convenient, well-posed parametrization of CPTP belief updates.

*Note (WFR Embedding).* The GKSL generator embeds naturally into the Wasserstein-Fisher-Rao framework ({prf:ref}`def-the-wfr-action`, {ref}`Section 20.5 <sec-connection-to-gksl-master-equation>`): the commutator $-i[H, \varrho]$ corresponds to **transport** (continuous belief flow), while the dissipator $\sum_j \gamma_j(\cdot)$ corresponds to **reaction** (discrete mass creation/destruction). This provides a geometric foundation for the otherwise algebraic GKSL construction.

({prf:ref}`def-grounding-rate`) *definition* — **Grounding rate**

Let $G_t:=I(X_t;K_t)$ be the symbolic mutual information injected through the boundary (Node 13). The *grounding rate* is the average information inflow per step:

$$
\lambda_{\text{in}} := \mathbb{E}[G_t].

$$
Units: $[\lambda_{\text{in}}]=\mathrm{nat/step}$.

({prf:ref}`def-mixing-rate`) *definition* — **Mixing rate**

Let $S_t:=H(K_t)$ be the macro entropy. The *mixing rate* is the expected entropy growth not attributable to purposeful exploration:

$$
\lambda_{\text{mix}} := \mathbb{E}[(S_{t+1}-S_t)_+].

$$
Units: $[\lambda_{\text{mix}}]=\mathrm{nat/step}$.

({prf:ref}`def-dpi-boundary-capacity-constraint`) *definition* — **DPI / boundary-capacity constraint**

Consider the boundary stream $(X_t)_{t\ge 0}$ and the induced internal state process $(Z_t)_{t\ge 0}$ produced by the shutter (Definition {prf:ref}`def-bounded-rationality-controller`). Because all internal state is computed from boundary influx and internal memory, any information in the bulk must be mediated by a finite-capacity channel. Operationally, the data-processing constraint is:

$$
I_{\text{bulk}} \;\le\; C_{\partial},

$$
where $C_{\partial}$ is the effective information capacity of the boundary channel and $I_{\text{bulk}}$ is the amount of information the agent can stably maintain in $\mathcal{Z}$ without violating Causal Enclosure (no internal source term $\sigma$; Definition {prf:ref}`def-source-residual`).
Units: $[I_{\text{bulk}}]=[C_{\partial}]=\mathrm{nat}$.

({prf:ref}`def-information-density-and-bulk-information-volume`) *definition* — **Information density and bulk information volume**

Let $\rho(z,s)$ denote the probability density of the agent's belief state at position $z \in \mathcal{Z}$ and computation time $s$. The **information density** $\rho_I(z,s)\ge 0$ is defined as:

$$
\rho_I(z,s) := -\rho(z,s) \log \rho(z,s) + \frac{1}{2}\rho(z,s) \log\det G(z),

$$
with units of nats per unit Riemannian volume $d\mu_G=\sqrt{|G|}\,dz^n$ ($n=\dim\mathcal{Z}$). The first term is the local entropy contribution (Shannon density); the second term is the geometric correction accounting for the metric-induced volume distortion.

*Remark.* Integrating $\rho_I$ over $\mathcal{Z}$ yields the differential entropy $h[\rho] = -\int \rho \log \rho \, d\mu_G$ plus the expected log-volume $\frac{1}{2}\mathbb{E}_\rho[\log\det G]$. The latter term ensures that the information measure respects the intrinsic geometry: regions with curved (high-$|G|$) geometry contribute more information capacity.

({prf:ref}`def-a-bulk-information-volume`) *definition* — **a (Bulk information volume)**

Define the bulk information volume over a region $\Omega\subseteq\mathcal{Z}$ by

$$
I_{\text{bulk}}(\Omega) := \int_{\Omega} \rho_I(z,s)\, d\mu_G.

$$
When $\Omega=\mathcal{Z}$ we write $I_{\text{bulk}}:=I_{\text{bulk}}(\mathcal{Z})$. This is conceptually distinct from the probability-mass balance in {ref}`Section 2.11 <sec-variance-value-duality-and-information-conservation>`; here the integral measures grounded structure in nats.

({prf:ref}`def-boundary-capacity-area-law-at-finite-resolution`) *definition* — **Boundary capacity: area law at finite resolution**

Let $dA_G$ be the induced $(n-1)$-dimensional area form on $\partial\mathcal{Z}$. If the boundary interface has a minimal resolvable scale $\ell>0$ (pixel/token floor), then an operational capacity bound is an area law:

$$
C_{\partial}(\partial\mathcal{Z})
:=
\frac{1}{\eta_\ell}\oint_{\partial\mathcal{Z}} dA_G,

$$
where $\eta_\ell$ is the effective boundary area-per-nat at resolution $\ell$ (a resolution-dependent constant set by the interface).
Units: $[\eta_\ell]=[dA_G]/\mathrm{nat}$ and $[\ell]$ is the chosen boundary resolution length scale.

*Remark (discrete macro specialization).* For the split shutter, the most conservative computable proxy is

$$
C_{\partial}\ \approx\ \mathbb{E}[I(X_t;K_t)]\ \le\ \log|\mathcal{K}|,

$$
which is exactly Node 13 (BoundaryCheck) and Theorem {prf:ref}`thm-information-stability-window-operational`'s grounding condition.

({prf:ref}`def-extended-risk-tensor`) *definition* — **Extended Risk Tensor with Maxwell Stress**

The total Risk Tensor $T_{ij}$ decomposes into gradient and curl contributions:

$$
T_{ij} = T_{ij}^{\text{gradient}} + T_{ij}^{\text{Maxwell}},

$$
where:

1. **Gradient Stress** (from scalar potential $\Phi$):

$$
T_{ij}^{\text{gradient}} = \partial_i \Phi \, \partial_j \Phi - \frac{1}{2}G_{ij} \|\nabla\Phi\|_G^2

$$
2. **Maxwell Stress** (from {prf:ref}`def-value-curl` $\mathcal{F}$):

$$
T_{ij}^{\text{Maxwell}} = \mathcal{F}_{ik}\mathcal{F}_j^{\;k} - \frac{1}{4}G_{ij}\mathcal{F}^{kl}\mathcal{F}_{kl}

$$
*Units:* $[T_{ij}] = \mathrm{nat}^2/[z]^2$.

**Conservative Limit:** When $\mathcal{F} = 0$ (Definition {prf:ref}`def-conservative-reward-field`), the Maxwell term vanishes and we recover the standard gradient-only risk tensor.

**Non-Conservative Case:** When $\mathcal{F} \neq 0$, the Maxwell stress contributes additional terms to the curvature equation.

({prf:ref}`def-capacity-saturation-diagnostic`) *definition* — **Capacity saturation diagnostic**

Compute the capacity saturation ratio:

$$
\nu_{\text{cap}}(s) := \frac{I_{\text{bulk}}(s)}{C_{\partial}},

$$
where $I_{\text{bulk}}(s) = \int_{\mathcal{Z}} \rho_I(z,s)\, d\mu_G$ per Definition 18.1.2a.

*Interpretation:*
- $\nu_{\text{cap}} \ll 1$: Under-utilized capacity; the agent may be compressing excessively (lossy representation).
- $\nu_{\text{cap}} \approx 1$: Operating at capacity limit; geometry must regulate to prevent overflow.
- $\nu_{\text{cap}} > 1$: **Violation** of the DPI constraint (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`); indicates ungrounded structure.

*Cross-reference:* When $\nu_{\text{cap}} > 1$, the curvature correction (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) is insufficient. This triggers geometric reflow---the metric $G$ must increase $|G|$ (expand volume) to bring $I_{\text{bulk}}$ back within bounds.

({prf:ref}`def-the-wfr-action`) *definition* — **The Generalized WFR Action**

The squared WFR distance $d^2_{\mathrm{WFR}}(\rho_0, \rho_1)$ is the infimum of the generalized energy functional:

$$
\mathcal{E}[\rho, v, r] = \int_0^1 \int_{\mathcal{Z}} \left( \underbrace{\|v_s(z)\|_G^2}_{\text{Transport Cost}} + \underbrace{\lambda^2 |r_s(z)|^2}_{\text{Reaction Cost}} - \underbrace{2\langle \mathbf{A}(z), v_s(z) \rangle}_{\text{Vector Potential}} \right) d\rho_s(z) \, ds

$$
subject to the **Unbalanced Continuity Equation**:

$$
\partial_s \rho + \nabla \cdot (\rho v) = \rho r

$$
where:
- $v_s(z) \in T_z\mathcal{Z}$ is the **velocity field** (transport/flow)
- $r_s(z) \in \mathbb{R}$ is the **reaction rate** (growth/decay of mass)
- $\lambda > 0$ is the **length-scale parameter** balancing transport and reaction
- $G$ is the Riemannian metric on the continuous fibres ({ref}`Section 2.5 <sec-second-order-sensitivity-value-defines-a-local-metric>`)
- $\mathbf{A}(z)$ is the **vector potential** satisfying $d\mathbf{A} = \mathcal{F}$ (the {prf:ref}`def-value-curl`)

*Units:* $[\mathbf{A}] = \mathrm{nat}/[\text{length}]$.

**Conservative Limit:** When $\mathcal{F} = 0$ (Definition {prf:ref}`def-conservative-reward-field`), we can choose the gauge $\mathbf{A} = 0$ and recover the standard WFR action without the vector potential term.

**Non-Conservative Case:** When $\mathcal{F} \neq 0$, the vector potential term couples the transport velocity to the solenoidal component of the reward field. The Euler-Lagrange equations of this action yield the Lorentz-Langevin equation (Definition {prf:ref}`def-bulk-drift-continuous-flow`).

*Remark (Gauge Invariance).* The action is invariant under gauge transformations $\mathbf{A} \to \mathbf{A} + d\chi$ for any scalar $\chi$, since $d(d\chi) = 0$. We fix the gauge via the Coulomb condition $\delta\mathbf{A} = 0$ (divergence-free).

*Forward reference (Boundary Conditions).* {ref}`Section 23.5 <sec-wfr-boundary-conditions-waking-vs-dreaming>` specifies how boundary conditions on $\partial\mathcal{Z}$ (sensory and motor boundaries) constrain the WFR dynamics: **Waking** imposes Dirichlet (sensors) + Neumann (motors) BCs; **Dreaming** imposes reflective BCs on both, enabling recirculating flow without external input.

({prf:ref}`def-canonical-length-scale`) *definition* — **Canonical length-scale**

Let $G$ be the latent metric on $\mathcal{Z}$. The canonical choice for $\lambda$ is the **geodesic injectivity radius**:

$$
\lambda := \min_{z \in \mathcal{Z}} \text{inj}_G(z),

$$
where $\text{inj}_G(z)$ is the injectivity radius at $z$ -- the largest $r$ such that the exponential map $\exp_z: T_z\mathcal{Z} \to \mathcal{Z}$ is a diffeomorphism on $B_r(0)$.

*Default value.* If the injectivity radius is unknown or the metric is learned, a practical default is:

$$
\lambda_{\text{default}} = \sqrt{\frac{\text{tr}(G^{-1})}{n}} \approx \text{mean characteristic length of } \mathcal{Z}.

$$
This corresponds to the RMS geodesic step size in an isotropic metric.

*Cross-reference:* The screening length $\ell_{\text{screen}} = 1/\kappa$ from {ref}`Section 24.2 <sec-the-bulk-potential-screened-poisson-equation>` plays an analogous role for temporal horizons; $\lambda$ plays the corresponding role for spatial horizons in the WFR geometry.

({prf:ref}`def-wfr-world-model`) *definition* — **WFR World Model**

The policy outputs a generalized velocity field $(v, r)$ to minimize the WFR path length to the target distribution (goal).

```python
import torch
import torch.nn as nn
from typing import Tuple

class WFRWorldModel(nn.Module):
    """
    Unified World Model using Unbalanced Optimal Transport dynamics.

    Predicts the 'Generalized Velocity' (v, r) for belief particles.
    No separate 'discrete' and 'continuous' modules.
    """

    def __init__(
        self,
        macro_embed_dim: int,
        nuisance_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        # Input: particle state + action
        # State includes: macro embedding, nuisance coords, mass (weight)
        input_dim = macro_embed_dim + nuisance_dim + 1 + action_dim

        # Single MLP backbone for unified dynamics
        self.dynamics_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Head 1: Transport velocity (Riemannian motion on fibre)
        self.head_v = nn.Linear(hidden_dim, nuisance_dim)

        # Head 2: Reaction rate (Fisher-Rao mass creation/destruction)
        self.head_r = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z_t: torch.Tensor,           # [B, D] latent state (macro_embed + nuisance)
        mass_t: torch.Tensor,        # [B, 1] particle weight (belief mass)
        action_t: torch.Tensor,      # [B, A] action
        dt: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state via WFR dynamics.

        Returns:
            z_next: [B, D] next latent state
            mass_next: [B, 1] next particle mass
            v_t: [B, nuisance_dim] transport velocity
            r_t: [B, 1] reaction rate
        """
        # Unified prediction
        inp = torch.cat([z_t, mass_t, action_t], dim=-1)
        feat = self.dynamics_net(inp)

        v_t = self.head_v(feat)  # Transport velocity
        r_t = self.head_r(feat)  # Reaction rate (log-growth)

        # Integrate dynamics (Euler step)
        # Position update (Transport): z' = z + v * dt
        z_next = z_t.clone()
        z_next[..., -self.head_v.out_features:] += v_t * dt

        # Mass update (Reaction): m' = m * exp(r * dt)
        # If r > 0: hypothesis gaining probability (jumping in)
        # If r < 0: hypothesis losing probability (jumping out)
        mass_next = mass_t * torch.exp(r_t * dt)

        return z_next, mass_next, v_t, r_t
```

**How this handles the "Jump" seamlessly:**

- **Deep inside a Chart:** Model predicts $r \approx 0$ and $v \neq 0$. Particle moves normally.
- **Approaching a Boundary:** Model sees invalid description (high prediction error). Predicts $r < 0$ for current chart, $r > 0$ for neighboring chart particles.
- **Result:** Probability mass smoothly "tunnels" between charts without hard discrete switching.

({prf:ref}`def-scale-dependent-teleportation-cost`) *definition* — **Scale-Dependent Teleportation Cost**

$$
\lambda^{(\ell)} \propto \sigma^{(\ell)} \quad \text{(jump cost scales with residual variance)}

$$
where $\sigma^{(\ell)}$ is the scale factor from Definition {prf:ref}`def-the-rescaling-operator-renormalization`.

**Interpretation:**
- **Layer 0 (Bulk / IR):** High $\lambda^{(0)}$. Jumping is expensive; macro-structure is rigid. Transport dominates.
- **Layer $L$ (Texture / UV):** Low $\lambda^{(L)}$. "Mass" (texture details) can appear/disappear cheaply. Reaction dominates.

**Correspondence with Cosmological Constant:**
In the capacity-constrained metric law ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`, Theorem {prf:ref}`thm-capacity-constrained-metric-law`), the term $\Lambda G_{ij}$ plays the role of a baseline curvature. The correspondence is:

$$
\Lambda^{(\ell)} \sim \frac{1}{(\lambda^{(\ell)})^2}

$$
- Bulk (low $\Lambda$): Flat, rigid, transport-dominated
- Boundary (high $\Lambda$): Curved, fluid, reaction-dominated

({prf:ref}`def-wfr-consistency-loss-wfrcheck`) *definition* — **WFR Consistency Loss / WFRCheck**

The cone-space representation linearizes WFR locally. From $\partial_s \rho = \rho r - \nabla \cdot (\rho v)$ and $u = \sqrt{\rho}$, we have $\partial_s u = \frac{\rho r - \nabla \cdot (\rho v)}{2\sqrt{\rho}}$. Define the consistency loss:

$$
\mathcal{L}_{\mathrm{WFR}} = \left\| \sqrt{\rho_{t+1}} - \sqrt{\rho_t} - \frac{\Delta t}{2\sqrt{\rho_t}}\left(\rho_t r_t - \nabla \cdot (\rho_t v_t)\right) \right\|_{L^2}^2

$$
This penalizes deviations from the unbalanced continuity equation.

**Practical implementation:**

```python
def compute_wfr_consistency_loss(
    rho_t: torch.Tensor,       # [B, K] belief over charts at time t
    rho_t1: torch.Tensor,      # [B, K] belief over charts at time t+1
    v_t: torch.Tensor,         # [B, K, d_n] transport velocity per chart
    r_t: torch.Tensor,         # [B, K] reaction rate per chart
    dt: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute WFR consistency loss (cone-space formulation).

    Penalizes violation of unbalanced continuity equation.
    """
    sqrt_rho_t = torch.sqrt(rho_t + eps)
    sqrt_rho_t1 = torch.sqrt(rho_t1 + eps)

    # Approximate divergence term (finite difference)
    # In practice, use automatic differentiation if v is differentiable
    div_rho_v = torch.zeros_like(rho_t)  # Placeholder for nabla . (rho v)

    # Predicted change in sqrt(rho) from: d/ds sqrt(rho) = (rho*r - div(rho*v)) / (2*sqrt(rho))
    predicted_delta = (dt / (2 * sqrt_rho_t + eps)) * (rho_t * r_t - div_rho_v)

    # Actual change
    actual_delta = sqrt_rho_t1 - sqrt_rho_t

    # L2 loss
    loss = ((actual_delta - predicted_delta) ** 2).mean()

    return loss
```

({prf:ref}`def-manifold-boundary-and-interior`) *definition* — **Manifold Boundary and Interior**

Let $\mathcal{Z}$ be the latent manifold with Poincare disk model. The **boundary** is the $(n-1)$-dimensional limit set:

$$
\partial\mathcal{Z} := \{z \in \mathbb{C}^n : |z| = 1\}.

$$
The **interior** (or bulk) is the open disk:

$$
\text{int}(\mathcal{Z}) := \{z \in \mathbb{C}^n : |z| < 1\}.

$$
These are standard differential geometry terms; the boundary is the ideal boundary at infinity in the hyperbolic metric.

({prf:ref}`def-hyperbolic-volume-growth`) *definition* — **Hyperbolic Volume Growth**

With metric $G_{ij} = \frac{4\delta_{ij}}{(1-|z|^2)^2}$, the volume of a hyperbolic ball $B_r(0)$ grows exponentially:

$$
\mathrm{Vol}(B_r(0)) = 4\pi \sinh^2\!\left(\frac{r}{2}\right) \;\approx\; \pi e^r \quad \text{as } r \to \infty.

$$
Units: $[\mathrm{Vol}] = [z]^2$.

({prf:ref}`def-the-entropic-force`) *definition* — **The Entropic Force**

The "Free Energy" of a state at radius $r$ is dominated by the entropic volume term $S(r) \sim 2 \tanh^{-1}(r)$. To maximize entropy (fill the capacity), the agent experiences a radial force:

$$
F_{\text{entropy}}(z) = \nabla_G S(z) = \frac{z}{\|z\|}

$$
In normalized hyperbolic coordinates, this yields a **constant radial drift**.

Units: $[F_{\text{entropy}}] = [z]/\tau$.

({prf:ref}`def-hyperbolic-information-potential`) *definition* — **Hyperbolic Information Potential**

The **information potential** $U: \mathbb{D} \to \mathbb{R}$ is the negative hyperbolic distance from the origin:

$$
U(z) := -d_{\mathbb{D}}(0, z) = -2 \operatorname{artanh}(|z|) = -\log\!\left(\frac{1+|z|}{1-|z|}\right).

$$
Units: $[U] = \mathrm{nat}$.

*Remark (Thermodynamic Interpretation).* At origin ($z=0$): $U = 0$ (maximum potential, maximum entropy). At boundary ($|z| \to 1$): $U \to -\infty$ (minimum potential, fully specified). The depth $-U(z)$ measures the **information content** of the state.

({prf:ref}`def-the-control-field`) *definition* — **The Control Field**

The Policy $\pi_\theta(a|z)$ outputs a **control field** $u_\pi(z)$ on the tangent bundle $T\mathbb{D}$:

$$
u_\pi(z) = G^{-1}(z) \cdot \mathbb{E}_{a \sim \pi_\theta}[a]

$$
This vector field represents the **Information Preference** of the agent (or the User).

Units: $[u_\pi] = [z]/\tau$.

*Remark (Context-Conditioning).* {ref}`Section 23.6 <sec-relationship-to-the-context-conditioned-framework>` generalizes this to **context-conditioned policies** $\pi(a|z,c)$ where the context $c \in \mathcal{C}$ unifies: RL action spaces, classification label spaces, and LLM prompt spaces. The control field becomes $u_\pi(z,c) = G^{-1}(z) \cdot \nabla_z \Phi_{\text{eff}}(z,K,c)$ where the {prf:ref}`def-effective-potential` depends on task context.

({prf:ref}`def-control-field-at-origin`) *definition* — **Control Field at Origin**

At $\tau=0$, the total drift is:

$$
F_{\text{total}} = F_{\text{entropy}} + u_\pi(0)

$$
Since $F_{\text{entropy}}(0) = 0$ (isotropic), the initial trajectory is determined **entirely** by $u_\pi(0)$.

({prf:ref}`def-boundary-texture-distribution`) *definition* — **Boundary Texture Distribution**

At the terminal position $z_{\text{final}}$, texture is sampled from a **geometry-dependent** Gaussian:

$$
z_{\text{tex}} \sim \mathcal{N}\big(0,\, \Sigma(z_{\text{final}})\big),

$$
where the covariance matrix is:

$$
\Sigma(z) = \sigma_{\text{tex}}^2 \cdot G^{-1}(z) = \sigma_{\text{tex}}^2 \cdot \frac{(1-|z|^2)^2}{4} I.

$$
Units: $[\Sigma] = [z_{\text{tex}}]^2$.

({prf:ref}`def-boundary-decoder`) *definition* — **Boundary Decoder**

The Decoder $\mathcal{D}$ is the **only** component that sees texture. It performs the **boundary synthesis**:

$$
x = \mathcal{D}(z_{\text{final}}, z_{\text{tex}})

$$
where:
- $z_{\text{final}} = (e_K, z_n)$: Determines the shape, physics, and causal structure
- $z_{\text{tex}}$: "Paints" the high-frequency details onto that structure

({prf:ref}`def-stopping-criterion`) *definition* — **Stopping Criterion**

The flow terminates when the radial coordinate exceeds a cutoff:

$$
\tau_{\text{stop}} := \inf\{\tau \ge 0 : |z(\tau)| \ge R_{\text{cutoff}}\}

$$
This is equivalent to the information stopping criterion $I_{\text{bulk}}(z) \ge C_\partial$ (Theorem {prf:ref}`thm-capacity-constrained-metric-law`).

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

({prf:ref}`def-mass-tensor`) *definition* — **Mass Tensor**

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

({prf:ref}`def-extended-onsager-machlup-action`) *definition* — **Extended Onsager-Machlup Action**

Let $(\mathcal{Z}, G)$ be the latent Riemannian manifold with the capacity-constrained metric ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`). For a path $z: [0, T] \to \mathcal{Z}$, the extended Onsager-Machlup action is:

$$
S_{\mathrm{OM}}[z] = \int_0^T \left( \frac{1}{2}\mathbf{M}(z)\|\dot{z}\|^2 + \Phi_{\text{eff}}(z) + \frac{T_c}{12}\,R(z) + T_c \cdot H_{\pi}(z) \right) ds,

$$
where:
- $\mathbf{M}(z)\|\dot{z}\|^2 = G_{ij}(z)\,\dot{z}^i\,\dot{z}^j$ is the kinetic energy (mass = metric)
- $\Phi_{\text{eff}}(z)$ is the effective potential (Definition {prf:ref}`def-effective-potential`)
- $R(z)$ is the scalar curvature of the metric $G$
- $H_{\pi}(z) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|z)]$ is the policy entropy
- $T_c > 0$ is the {prf:ref}`def-cognitive-temperature` (cf. {ref}`Section 21.2 <sec-policy-control-field>`)

Units: $[S_{\mathrm{OM}}] = \mathrm{nat}$.

*Remark (Curvature Correction).* The term $\frac{T_c}{12}R(z)$ is a stochastic correction that accounts for the path-measure distortion on curved spaces. In flat space ($R = 0$), this term vanishes. The entropy term $T_c H_{\pi}$ ensures the agent prefers stochastic policies in uncertain regions.

({prf:ref}`def-bulk-drift-continuous-flow`) *definition* — **Bulk Drift - Continuous Flow (Lorentz-Langevin Equation)**

The position coordinates $z^k$ evolve according to the **Lorentz-Langevin SDE**:

$$
dz^k = \underbrace{\left( -G^{kj}\partial_j \Phi + u_\pi^k \right)}_{\text{gradient + control}} ds \;+\; \underbrace{\beta_{\text{curl}}\, G^{km} \mathcal{F}_{mj} \dot{z}^j\,ds}_{\text{Lorentz force}} \;-\; \underbrace{\Gamma^k_{ij}\dot{z}^i \dot{z}^j\,ds}_{\text{geodesic correction}} \;+\; \underbrace{\sqrt{2T_c}\,(G^{-1/2})^{kj}\,dW^j_s}_{\text{thermal noise}},

$$
where:
- $\Phi$ is the **scalar potential** from the Hodge decomposition (Theorem {prf:ref}`thm-hodge-decomposition`)
- $\mathcal{F}_{ij} = \partial_i \mathcal{R}_j - \partial_j \mathcal{R}_i$ is the **Value Curl** tensor (Definition {prf:ref}`def-value-curl`)
- $\beta_{\text{curl}} \ge 0$ is the **curl coupling strength** (dimensionless)
- $u_\pi^k$ is the **control field** from the policy (Definition {prf:ref}`prop-so-d-symmetry-at-origin`)
- $\Gamma^k_{ij}$ are the **Christoffel symbols** of the Levi-Civita connection ({ref}`Section 2.5.1 <sec-levi-civita-connection-and-parallel-transport>`, Theorem {prf:ref}`thm-capacity-constrained-metric-law`)
- $G^{-1/2}$ is the matrix square root of the inverse metric
- $W_s$ is a standard Wiener process

*Units:* $[dz] = [z]$, $[\Phi] = \mathrm{nat}$, $[\mathcal{F}_{ij}] = \mathrm{nat}/[z]^2$, $[\Gamma^k_{ij}] = [z]^{-1}$.

*Remark (Four-Force Decomposition).* The drift decomposes into:
1. **Gradient force**: $-G^{-1}\nabla\Phi$ — force proportional to scalar potential gradient
2. **Lorentz force**: $\beta_{\text{curl}} G^{-1}\mathcal{F}\dot{z}$ — velocity-dependent force from Value Curl
3. **Control field**: $u_\pi$ — policy-induced drift ({ref}`Section 21.2 <sec-policy-control-field>`)
4. **Geodesic correction**: $-\Gamma(\dot{z},\dot{z})$ — parallel transport on curved space

**Conservative Limit:** When $\mathcal{F} = 0$ (Definition {prf:ref}`def-conservative-reward-field`), the Lorentz term vanishes and we recover the standard geodesic SDE.

**Non-Conservative Dynamics:** When $\mathcal{F} \neq 0$, the Lorentz force induces rotational dynamics. Trajectories may converge to limit cycles rather than fixed points (Theorem {prf:ref}`thm-ness-existence`).

*Remark (Connection Specification).* The Christoffel symbols $\Gamma^k_{ij}$ are explicitly those of the **Levi-Civita connection** induced by the capacity-constrained metric $G$ from Theorem {prf:ref}`thm-capacity-constrained-metric-law`. This ensures metric compatibility ($\nabla G = 0$) and torsion-freeness.

({prf:ref}`def-mass-evolution-jump-process`) *definition* — **Mass Evolution - Jump Process**

The importance weight $m(s)$ evolves according to a coupled jump-diffusion:

$$
dm = m \cdot r(z, a)\,ds + m \cdot (\eta - 1)\,dN_s,

$$
where:
- $r(z, a)$ is the **reaction rate** from the WFR dynamics ({ref}`Section 20.2 <sec-the-wfr-metric>`)
- $N_s$ is a Poisson process with intensity $\lambda_{\text{jump}}(z)$
- $\eta$ is the multiplicative jump factor (typically $\eta > 1$ for jumps to higher-value charts)

*Interpretation:* Between jumps, mass evolves smoothly via the reaction term $r$. At jump times, the mass is rescaled by factor $\eta$, and the position is teleported via the chart transition operator $L_{i \to j}$.

({prf:ref}`def-effective-potential`) *definition* — **Effective Potential**

The unified effective potential is:

$$
\Phi_{\text{eff}}(z, K) = \alpha\, U(z) + (1 - \alpha)\, V_{\text{critic}}(z, K) + \gamma_{risk}\, \Psi_{\text{risk}}(z),

$$
where:
- $U(z) = -d_{\mathbb{D}}(0, z) = -2\operatorname{artanh}(|z|)$ is the **hyperbolic information potential** (Definition {prf:ref}`def-hyperbolic-volume-growth`)
- $V_{\text{critic}}(z, K)$ is the **learned value/critic function** on chart $K$ ({ref}`Section 2.7 <sec-the-hjb-correspondence>`)
- $\Psi_{\text{risk}}(z) = \frac{1}{2}\operatorname{tr}(T_{ij} G^{ij})$ is the **risk-stress contribution** (Theorem {prf:ref}`thm-capacity-constrained-metric-law`)
- $\alpha \in [0, 1]$ is the generation-vs-control hyperparameter
- $\gamma_{risk} \ge 0$ is the risk aversion coefficient

Units: $[\Phi_{\text{eff}}] = \mathrm{nat}$.

({prf:ref}`def-cognitive-temperature`) *definition* — **Cognitive Temperature**

The **cognitive temperature** $T_c > 0$ is the exploration-exploitation tradeoff parameter that controls:

1. **Diffusion magnitude:** The thermal noise term in the geodesic SDE scales as $\sqrt{2T_c}\,dW$
2. **Boltzmann policy:** The softmax temperature in $\pi(a|z) \propto \exp(Q(z,a)/T_c)$
3. **Free energy tradeoff:** The entropy-energy balance $\Phi = E - T_c S$

*Units:* nat (dimensionless in natural units where $k_B = 1$).

*Correspondence:* $T_c$ is the agent-theoretic analogue of thermodynamic temperature $k_B T$ in statistical mechanics.

({prf:ref}`def-baoab-splitting`) *definition* — **Boris-BAOAB Splitting**

The Boris-BAOAB integrator splits the Lorentz-Langevin dynamics into five substeps per time step $h$:

1. **B** (half kick + Boris rotation):
   - Half-kick from gradient: $p^- \leftarrow p - \frac{h}{2}\nabla\Phi(z)$
   - Boris rotation (if $\mathcal{F} \neq 0$):
     - $t \leftarrow \frac{h}{2}\beta_{\text{curl}} G^{-1}\mathcal{F}$ (rotation vector)
     - $p' \leftarrow p^- + p^- \times t$
     - $s \leftarrow \frac{2t}{1 + |t|^2}$
     - $p^+ \leftarrow p^- + p' \times s$
   - Half-kick from gradient: $p \leftarrow p^+ - \frac{h}{2}\nabla\Phi(z)$

2. **A** (half drift): $z \leftarrow \operatorname{Exp}_z\left(\frac{h}{2} G^{-1}(z)\, p\right)$

3. **O** (thermostat): $p \leftarrow c_1 p + c_2\, G^{1/2}(z)\, \xi$, where $\xi \sim \mathcal{N}(0, I)$

4. **A** (half drift): $z \leftarrow \operatorname{Exp}_z\left(\frac{h}{2} G^{-1}(z)\, p\right)$

5. **B** (half kick + Boris rotation): Same as step 1

where $c_1 = e^{-\gamma h}$ and $c_2 = \sqrt{(1 - c_1^2) T_c}$.

**Conservative Limit:** When $\mathcal{F} = 0$, the Boris rotation is identity and we recover standard BAOAB.

*Remark (Boris Rotation).* The Boris algorithm is a volume-preserving integrator for magnetic-like forces. It rotates the momentum around the local Value Curl axis, preserving the norm $|p|$ while changing direction. This ensures the Lorentz force does no net work, consistent with physics.

*Remark (O-step).* The O-step implements the **Ornstein-Uhlenbeck thermostat**, which exactly preserves the Maxwell-Boltzmann momentum distribution $p \sim \mathcal{N}(0, T_c G)$.

({prf:ref}`def-agent-lifecycle-phases`) *definition* — **Agent Lifecycle Phases**

| Phase           | Time Interval                | Dynamics                         | Texture      | Key Operations                                                                         |
|-----------------|------------------------------|----------------------------------|--------------|----------------------------------------------------------------------------------------|
| **1. Init**     | $\tau = 0$                   | $z(0) = 0$                       | None         | Initialize at origin; $p(0) \sim \mathcal{N}(0, T_c G(0))$                             |
| **2. Kick**     | $[0, \tau_{kick}]$           | Langevin at origin               | None         | Apply symmetry-breaking control $u_\pi$ (Def. {prf:ref}`prop-so-d-symmetry-at-origin`) |
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
    # Apply symmetry-breaking control at origin
    u_pi = policy.symmetry_breaking_kick(z, mode='generation')

    # ===== Phase 3: Bulk (with Texture Firewall) =====
    for step in range(max_steps):
        # Compute effective potential gradient
        grad_Phi = compute_effective_potential_gradient(
            state.z, state.K, policy.value_fn, alpha=0.5
        )

        # Update control field
        u_pi = policy.control_field(state.z, state.K)

        # BAOAB step (texture is invisible here)
        state = geodesic_baoab_step(
            state, grad_Phi, u_pi, T_c, gamma, h,
            jump_rate_fn=policy.jump_rate,
            chart_transition_fn=policy.chart_transition
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

({prf:ref}`def-einstein-relation-on-manifolds`) *definition* — **Einstein Relation on Manifolds**

The fluctuation-dissipation relation requires:

$$
\sigma^2(z) = \frac{2\gamma(z)\, T_c}{G(z)},

$$
where $\sigma^2$ is the noise variance. This ensures the correct equilibrium distribution.

({prf:ref}`def-fisher-covariance-duality`) *definition* — **Fisher-Covariance Duality**

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

    This maintains constant effective noise: sigma^2 * G = 2 * gamma * T_c
    """
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    # Conformal factor inverse: G^{-1} = (1-|z|^2)^2 / 4
    inv_conformal = (1.0 - r_sq + 1e-8) ** 2 / 4.0
    return base_T * inv_conformal * certainty_scale
```

({prf:ref}`def-symplectic-boundary-manifold`) *definition* — **Symplectic Boundary Manifold**

The agent's interface is a symplectic manifold $(\partial\mathcal{Z}, \omega)$ with canonical coordinates $(q, p) \in T^*\mathcal{M}$ where:
- $q \in \mathcal{Q}$ is the **position bundle** (sensory configuration)
- $p \in T^*_q\mathcal{Q}$ is the **momentum bundle** (motor flux)

The symplectic form is:

$$
\omega = \sum_{i=1}^n dq^i \wedge dp_i.

$$
Units: $[\omega] = [q][p] = \mathrm{nat}$.

*Remark (Causal Structure).* The symplectic structure encodes causality: observations fix "where" the belief state is (position), while actions fix "how" it flows outward (momentum/flux). These cannot be treated symmetrically as static fields.

({prf:ref}`def-dirichlet-boundary-condition-sensors`) *definition* — **Dirichlet Boundary Condition --- Sensors**

The sensory input stream $\phi(x)$ imposes a **Dirichlet** (position-clamping) condition on the belief density:

$$
\rho_{\partial}^{\text{sense}}(q, t) = \delta(q - q_{\text{obs}}(t)),

$$
where $q_{\text{obs}}(t) = E_\phi(x_t)$ is the encoded observation. This clamps the *configuration* of the belief state.

*Interpretation:* Information flow from environment to agent (observation).

({prf:ref}`def-neumann-boundary-condition-motors`) *definition* — **Neumann Boundary Condition --- Motors**

The motor output stream $A(x)$ imposes a **Neumann** (flux-clamping) condition:

$$
\nabla_n \rho \cdot \mathbf{n} \big|_{\partial\mathcal{Z}_{\text{motor}}} = j_{\text{motor}}(p, t),

$$
where $j_{\text{motor}}$ is the motor current density determined by the policy:

$$
j_{\text{motor}} = D_A(u_\pi) = \text{Decoder}(z, u_\pi, z_{\text{tex,motor}}).

$$
*Interpretation:* Information flow from agent to environment (action).

Units: $[j_{\text{motor}}] = \mathrm{nat}/\text{step}$.

({prf:ref}`def-visual-atlas-perception`) *definition* — **Visual Atlas — Perception**

The Visual Atlas $\mathcal{A}_{\text{vis}} = \{(U_\alpha, \phi_\alpha, e_\alpha^{\text{vis}})\}_{\alpha \in \mathcal{K}_{\text{vis}}}$ is a chart atlas on the sensory manifold $\mathcal{Q}$ with:
- **Charts** $U_\alpha \subset \mathcal{Q}$: Objects, Scenes, Viewpoints
- **Chart maps** $\phi_\alpha: U_\alpha \to \mathbb{R}^{d_{\text{vis}}}$: Local coordinates
- **Codebook embeddings** $e_\alpha^{\text{vis}} \in \mathbb{R}^{d_m}$: Discrete macro codes

*Input:* Raw observations $\phi_{\text{raw}}$ (pixels, sensors).
*Output:* Latent state $z \in \mathcal{Z}$ (configuration).

({prf:ref}`def-action-atlas-actuation`) *definition* — **Action Atlas --- Actuation**

The Action Atlas $\mathcal{A}_{\text{act}} = \{(V_\beta, \psi_\beta, e_\beta^{\text{act}})\}_{\beta \in \mathcal{K}_{\text{act}}}$ is a chart atlas on the motor manifold $T^*\mathcal{Q}$ with:
- **Charts** $V_\beta \subset T^*\mathcal{Q}$: Gaits, Grasps, Tool Affordances (topologically distinct control regimes)
- **Chart maps** $\psi_\beta: V_\beta \to \mathbb{R}^{d_{\text{act}}}$: Local motor coordinates
- **Codebook embeddings** $e_\beta^{\text{act}} \in \mathbb{R}^{d_m}$: Action primitive codes

*Input:* Intention $u_{\text{intent}} \in T_z\mathcal{Z}$ (from Policy, {ref}`Section 21.2 <sec-policy-control-field>`).
*Output:* Actuation $a_{\text{raw}}$ (torques, voltages).

*Remark (Jump Operator in Action Atlas).* The **Jump Operator** $L_{\beta \to \beta'}$ in the Action Atlas represents **Task Switching**: transitioning from one control primitive to another (e.g., "Walk" $\to$ "Jump", "Grasp" $\to$ "Release"). This mirrors the chart transition operator in the Visual Atlas ({ref}`Section 20.6 <sec-the-unified-world-model>`).

({prf:ref}`def-the-holographic-shutter-unified-interface`) *definition* — **The Holographic Shutter — Unified Interface**

The Shutter is extended from {ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>` to a symmetric tuple:

$$
\mathbb{S} = (\mathcal{A}_{\text{vis}}, \mathcal{A}_{\text{act}}),

$$
where:
- **Ingress (Perception):** $E_\phi: \mathcal{Q} \to \mathcal{Z}$ via Visual Atlas
- **Egress (Actuation):** $D_A: T_z\mathcal{Z} \times \mathcal{Z} \to T^*\mathcal{Q}$ via Action Atlas
- **Proprioception (Inverse Model):** $E_A: T^*\mathcal{Q} \to T_z\mathcal{Z}$ maps realized actions back to intentions

**Cross-references:** {ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>` (VQ-VAE Shutter), {ref}`Section 7.8 <sec-tier-the-attentive-atlas>` (AttentiveAtlasEncoder), {ref}`Section 7.10 <sec-decoder-architecture-overview-topological-decoder>` (TopologicalDecoder).

({prf:ref}`def-motor-texture-decomposition`) *definition* — **Motor Texture Decomposition**

The motor output decomposes as:

$$
a_t = (A_t, z_{n,\text{motor}}, z_{\text{tex,motor}}),

$$
where:
- $A_t \in \mathcal{K}_{\text{act}}$ is the **discrete motor macro** (action primitive/chart index)
- $z_{n,\text{motor}} \in \mathbb{R}^{d_{\text{motor},n}}$ is **motor nuisance** (impedance, compliance, force distribution)
- $z_{\text{tex,motor}} \in \mathbb{R}^{d_{\text{motor,tex}}}$ is **motor texture** (tremor, fine-grained noise, micro-corrections)

*Remark (Parallel to Visual Decomposition).* This mirrors the visual decomposition $(K_t, z_{n,t}, z_{\text{tex},t})$ from {ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>`:

| Component                 | Visual Domain                 | Motor Domain                              |
|---------------------------|-------------------------------|-------------------------------------------|
| **Macro (discrete)**      | Object/Scene chart $K$        | Action primitive $A$                      |
| **Nuisance (continuous)** | Pose/viewpoint $z_n$          | Compliance/impedance $z_{n,\text{motor}}$ |
| **Texture (residual)**    | Pixel detail $z_{\text{tex}}$ | Tremor/noise $z_{\text{tex,motor}}$       |

({prf:ref}`def-compliance-tensor`) *definition* — **Compliance Tensor**

The motor nuisance encodes the **compliance tensor**:

$$
C_{ij}(z_{n,\text{motor}}) = \frac{\partial a^i}{\partial f^j},

$$
where $f$ is the external force/feedback. This determines how the motor output responds to perturbations:
- **High compliance** ($C$ large): Soft, yielding response (safe interaction)
- **Low compliance** ($C$ small): Stiff, precise response (accurate positioning)

Units: $[C_{ij}] = [a]/[f]$.

({prf:ref}`def-motor-texture-distribution`) *definition* — **Motor Texture Distribution**

At the motor boundary, texture is sampled from a geometry-dependent Gaussian:

$$
z_{\text{tex,motor}} \sim \mathcal{N}(0, \Sigma_{\text{motor}}(z)),

$$
where:

$$
\Sigma_{\text{motor}}(z) = \sigma_{\text{motor}}^2 \cdot G_{\text{motor}}^{-1}(z) = \sigma_{\text{motor}}^2 \cdot \frac{(1-|z|^2)^2}{4} I_{d_{\text{motor,tex}}}.

$$
This follows the same conformal scaling as visual texture (Definition {prf:ref}`def-boundary-texture-distribution`), ensuring consistent thermodynamic behavior.

({prf:ref}`def-cycle-phases`) *definition* — **Cycle Phases**

| Phase             | Process            | Information Flow                      | Entropy Change               |
|-------------------|--------------------|---------------------------------------|------------------------------|
| **I. Perception** | Compression        | Mutual information $I(X;K)$ extracted | $\Delta S_{\text{bulk}} < 0$ |
| **II. Dreaming**  | Internal evolution | No external exchange                  | $\Delta S = 0$ (isentropic)  |
| **III. Action**   | Expansion          | Mutual information $I(A;K)$ injected  | $\Delta S_{\text{bulk}} > 0$ |

*Remark (Statistical mechanics analogy).* This cycle is structurally analogous to a Stirling cycle in thermodynamics.

({prf:ref}`def-dreaming-as-unitary-evolution`) *definition* — **Dreaming as Unitary Evolution**

In the dreaming phase, the internal dynamics are approximately unitary (energy-conserving):

$$
\partial_s \rho + [H_{\text{internal}}, \rho]_{\text{Poisson}} = 0,

$$
where $H_{\text{internal}}$ is the effective Hamiltonian:

$$
H_{\text{internal}}(z, p) = \frac{1}{2}\|p\|_{G^{-1}}^2 + V_{\text{critic}}(z).

$$
*Mechanism:* The agent is decoupled from the boundary (adiabatic/isolated). The Bulk evolves under Hamiltonian dynamics (BAOAB integrator with $\gamma \to 0$).

*Information-theoretic interpretation:* Isentropic ($\Delta S = 0$). Internal planning proceeds without information exchange with the environment.

({prf:ref}`def-waking-boundary-clamping`) *definition* — **Waking: Boundary Clamping**

During waking ($u_\pi \neq 0$), the sensory stream creates a high-mass source at the encoded location:

$$
\rho_{\partial}^{\text{sense}}(z, t) = \delta(z - z_{\text{obs}}(t)) \quad \text{(Dirichlet)},

$$
and the motor stream creates a flux sink:

$$
\nabla_n \rho \cdot \mathbf{n} = j_{\text{motor}}(u_\pi) \quad \text{(Neumann)}.

$$
The internal belief $\rho_{\text{bulk}}$ evolves to minimize the **WFR Geodesic Distance** to $\rho_{\partial}$:
- **Small Error** ($d_{\text{WFR}} < \lambda$): Transport dominates ($v$ term). The agent smoothly tracks the observation.
- **Large Error** ($d_{\text{WFR}} > \lambda$): Reaction dominates ($r$ term). The agent "teleports" (Surprise/Saccade) to the new reality via chart jump.

({prf:ref}`def-dreaming-reflective-boundary`) *definition* — **Dreaming: Reflective Boundary**

During dreaming ($u_\pi = 0$), the sensory stream is cut. The boundary condition becomes **Reflective**:

$$
\nabla_n \rho \cdot \mathbf{n} = 0 \quad \text{(Reflective/Neumann-zero)}.

$$
The system is closed:
- Total mass is conserved: $\int_{\mathcal{Z}} \rho\, r\, d\mu_G = 0$
- Dynamics are driven purely by the internal potential $V_{\text{critic}}(z)$
- No information enters or leaves the boundary

({prf:ref}`def-context-space`) *definition* — **Context Space**

The **Context Space** $\mathcal{C}$ is a manifold parameterizing the control/conditioning signal for the agent:

$$
\mathcal{C} := \{c : c \text{ specifies a boundary condition on } \partial\mathcal{Z}\}.

$$
The context determines the target distribution at the motor boundary via the effective potential:

$$
\pi(a | z, c) \propto \exp\left(-\frac{1}{T_c} \Phi_{\text{eff}}(z, K, c)\right).

$$
Units: $[\mathcal{C}]$ inherits from the task domain.

({prf:ref}`def-context-instantiation-functor`) *definition* — **Context Instantiation Functor**

The Context Space admits a functor $\mathcal{I}: \mathbf{Task} \to \mathcal{C}$ with three canonical instantiations:

| Task Domain        | Context $c \in \mathcal{C}$ | Motor Output $a$           | Effective Potential $\Phi_{\text{eff}}$      |
|--------------------|-----------------------------|----------------------------|----------------------------------------------|
| **RL**             | Action space $\mathcal{A}$  | Motor command (torques)    | $V_{\text{critic}}(z, K)$                    |
| **Classification** | Label space $\mathcal{Y}$   | Class prediction $\hat{y}$ | $-\log p(y\mid z)$ (cross-entropy)           |
| **LLM**            | Prompt space $\mathcal{P}$  | Token sequence             | $-\log p(\text{token}\mid z, \text{prompt})$ |

*Key Insight:* In all cases, the context $c$ functions as the **symmetry-breaking boundary condition** that determines which direction the holographic expansion takes at the origin.

({prf:ref}`def-context-conditioned-wfr`) *definition* — **Context-Conditioned WFR**

The WFR dynamics ({ref}`Section 20.2 <sec-the-wfr-metric>`) generalize to context-conditioned form:

$$
\partial_s \rho + \nabla \cdot (\rho\, v_c) = \rho\, r_c,

$$
where:
- $v_c(z) = -G^{-1}(z) \nabla_z \Phi_{\text{eff}}(z, K, c) + u_\pi(z, c)$ is the context-conditioned velocity
- $r_c(z)$ is the context-conditioned reaction rate (chart jumps influenced by context)

({prf:ref}`def-reward-1-form`) *definition* — **The Reward 1-Form**

Let $\mathcal{R}$ be a differential 1-form on the latent manifold $(\mathcal{Z}, G)$. The **instantaneous reward rate** received by the agent moving with velocity $v \in T_z\mathcal{Z}$ is:

$$
r_t = \langle \mathcal{R}(z), v \rangle_G = \mathcal{R}_i(z) \dot{z}^i.

$$

*Units:* $[\mathcal{R}] = \mathrm{nat}/[\text{length}]$.

The cumulative reward along a trajectory $\gamma: [0,T] \to \mathcal{Z}$ is the **line integral**:

$$
R_{\text{cumulative}} = \int_\gamma \mathcal{R} = \int_0^T \mathcal{R}_i(\gamma(t)) \dot{\gamma}^i(t) \, dt.

$$

*Remark.* Instantaneous reward depends on both position $z$ and velocity $\dot{z}$. A stationary agent ($\dot{z} = 0$) receives zero instantaneous reward.

({prf:ref}`def-the-reward-flux`) *definition* — **The Reward Flux (Boundary Form)**

The environment provides reward via a flux form $J_r$ on the boundary $\partial\Omega$:

$$
\int_{\partial\Omega} J_r = \text{Cumulative Boundary Reward}.

$$
In the discrete limit, this manifests as point charges $r_t$ deposited at the boundary coordinates $(t, z_{\text{boundary}})$.

*Units:* $[J_r] = \mathrm{nat}/\mathrm{area}$, $[r_t] = \mathrm{nat}$.

*Relation to 1-form:* The boundary reward flux $J_r$ and the bulk 1-form $\mathcal{R}$ are related by Stokes' theorem: the boundary integral of $J_r$ equals the bulk integral of $d\mathcal{R}$ plus boundary terms.

({prf:ref}`def-value-curl`) *definition* — **The Value Curl (Vorticity Tensor)**

The **Value Curl** is the exterior derivative of the reward form:

$$
\mathcal{F} := d\mathcal{R} = d\delta\Psi.

$$
In coordinates: $\mathcal{F}_{ij} = \partial_i \mathcal{R}_j - \partial_j \mathcal{R}_i$.

*Units:* $[\mathcal{F}] = \mathrm{nat}/[\text{length}]^2$ (curvature of value).

**Properties:**
1. $\mathcal{F}$ is antisymmetric: $\mathcal{F}_{ij} = -\mathcal{F}_{ji}$
2. $\mathcal{F}$ satisfies the Bianchi identity: $d\mathcal{F} = 0$
3. $\mathcal{F}$ is gauge-invariant: if $\mathcal{R} \to \mathcal{R} + d\chi$, then $\mathcal{F} \to \mathcal{F}$

({prf:ref}`def-conservative-reward-field`) *definition* — **Conservative Reward Field**

The reward field $\mathcal{R}$ is **conservative** if and only if:

$$
\mathcal{F} = d\mathcal{R} = 0 \quad \text{(curl-free)}.

$$
Equivalently, $\mathcal{R} = d\Phi$ for some scalar potential $\Phi$ (the solenoidal and harmonic components vanish).

**Conservative Special Case:** When $\mathcal{F} = 0$ everywhere, we recover standard scalar value functions with $V(z) = \Phi(z)$. The cumulative reward around any closed loop vanishes:

$$
\oint_\gamma \mathcal{R} = \int_\Sigma d\mathcal{R} = \int_\Sigma \mathcal{F} = 0.

$$

*Remark.* Standard RL assumes conservative reward fields. The scalar value function $V(s)$ exists precisely because path-independence holds.

({prf:ref}`def-canonical-ensemble`) *definition* — **Canonical Ensemble {cite}`sutton2018rl`**

This potential induces a probability measure on the manifold via the **Canonical Ensemble**:

$$
P_{\text{stationary}}(z) = \frac{1}{Z} \exp\left(\frac{V(z)}{T_c}\right),

$$
where $Z = \int_{\mathcal{Z}} \exp(V(z)/T_c) \, d\mu_G(z)$ is the partition function.

*Sign Convention:* If $V$ is "Reward" (higher is better), use $+V/T_c$. If $V$ is "Cost" (lower is better), use $-V/T_c$. Throughout this document we use the **Reward convention** unless otherwise noted.

({prf:ref}`def-value-metric-conformal-coupling`) *definition* — **Value-Metric Conformal Coupling**

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

({prf:ref}`def-holographic-coefficient`) *definition* — **Holographic Coefficient**

The **Holographic Coefficient** $\nu_D$ for a $D$-dimensional latent manifold with $(D-1)$-sphere boundary is:

$$
\nu_D := \frac{(D-1)\,\Omega_{D-1}}{8\pi}

$$

where $\Omega_{D-1} = \frac{2\pi^{D/2}}{\Gamma(D/2)}$ is the surface area of the unit $(D-1)$-sphere.

| $D$ | Boundary | $\Omega_{D-1}$ | $\nu_D$ | Numerical |
|-----|----------|----------------|---------|-----------|
| 2   | Circle ($S^1$) | $2\pi$ | $1/4$ | 0.250 |
| 3   | Sphere ($S^2$) | $4\pi$ | $1$ | 1.000 |
| 4   | Glome ($S^3$) | $2\pi^2$ | $3\pi/4$ | 2.356 |
| 5   | 4-sphere ($S^4$) | $8\pi^2/3$ | $4\pi/3$ | 4.189 |
| 6   | 5-sphere ($S^5$) | $\pi^3$ | $5\pi^2/8$ | 6.169 |
| $D \gg 1$ | Hyper-sphere | $\to 0$ | $\to 0$ | Capacity collapse |

*Remark (Dimensional pressure).* The coefficient $\nu_D \to 0$ as $D \to \infty$ (curse of dimensionality). High-dimensional agents are **less efficient** at boundary information storage. This creates pressure for dimensional reduction—$D \approx 3$ maximizes holographic efficiency near the sweet spot.

*Remark (Physics correspondence).* For $D=2$, we recover the Bekenstein-Hawking coefficient $\nu_2 = 1/4$, making the Causal Information Bound $I_{\max} = \text{Area}/(4\ell_L^2)$ directly analogous to black hole entropy $S = A/(4\ell_P^2)$.

*Units:* $[\nu_D] = \text{dimensionless}$.

({prf:ref}`def-levin-length`) *definition* — **Levin Length**

Let $\eta_\ell$ be the boundary area-per-nat at resolution $\ell$ (Definition {prf:ref}`def-boundary-capacity-area-law-at-finite-resolution`). The **Levin Length** $\ell_L$ is the characteristic length scale of a single unit of distinction:

$$
\ell_L := \sqrt{\eta_\ell}.

$$
Units: $[\ell_L] = [z]$ (latent coordinate length).

*Interpretation.* A cell of area $\ell_L^2$ in the latent manifold corresponds to one nat of information capacity. The Levin Length is the information-geometric analog of a minimal resolvable element—the "pixel size" of the agent's internal representation.

*Remark (Naming).* The name honors Leonid Levin's foundational work on algorithmic information theory and the universal distribution {cite}`levin1973universal`. The Levin Length represents the floor below which distinctions cannot be computationally meaningful.

({prf:ref}`def-saturation-limit`) *definition* — **Saturation Limit**

The agent is at the **Saturation Limit** when the bulk information volume (Definition {prf:ref}`def-a-bulk-information-volume`) equals the boundary capacity (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`):

$$
I_{\text{bulk}} = C_\partial.

$$
At this limit, the DPI constraint $I_{\text{bulk}} \le C_\partial$ is satisfied with equality.

({prf:ref}`def-capacity-horizon-diagnostic`) *definition* — **Capacity Horizon Diagnostic**

Compute the **Saturation Ratio**:

$$
\eta_{\text{Sch}}(s) := \frac{I_{\text{bulk}}(s)}{I_{\max}} = \frac{I_{\text{bulk}}(s)}{\nu_D \cdot \text{Area}(\partial\mathcal{Z}) / \ell_L^{D-1}},

$$
where:
- $I_{\text{bulk}}(s) = \int_{\mathcal{Z}} \rho_I(z,s) \, d\mu_G$ per Definition {prf:ref}`def-a-bulk-information-volume`
- $\nu_D$ is the Holographic Coefficient (Definition {prf:ref}`def-holographic-coefficient`)
- $D$ is the latent manifold dimension

*Special case (Poincare disk, $D=2$):* $\eta_{\text{Sch}} = 4\ell_L^2 \cdot I_{\text{bulk}} / \text{Area}(\partial\mathcal{Z})$.

*Interpretation:*
- $\eta_{\text{Sch}} < 0.5$: Safe operating regime. Ample capacity headroom.
- $0.5 \le \eta_{\text{Sch}} < 0.9$: Elevated utilization. Monitor for growth trends.
- $0.9 \le \eta_{\text{Sch}} < 0.99$: **Warning.** Update velocity degraded (Corollary {prf:ref}`cor-saturation-velocity-tradeoff`). Prepare for ontological intervention.
- $\eta_{\text{Sch}} \ge 0.99$: **Critical.** Causal Stasis imminent. Halt exploration and trigger emergency fusion.

*Cross-reference:* Complements CapacitySaturationCheck (Node 40, {ref}`Section 18.3 <sec-diagnostic-node-capacity-saturation>`) by providing the velocity-degradation interpretation and connecting to ontological remediation.

({prf:ref}`def-semantic-partition`) *definition* — **Semantic Partition**

Let $\mathcal{Y} = \{1, \ldots, C\}$ be the set of class labels and $\mathcal{K}$ the macro-state register (Definition 2.2.1). A labeling $Y: \mathcal{X} \to \mathcal{Y}$ induces a **soft partition** of the chart atlas:

$$
\mathcal{A}_y := \{k \in \mathcal{K} : P(Y=y \mid K=k) > 1 - \epsilon_{\text{purity}}\},

$$
where $\epsilon_{\text{purity}} \in (0, 0.5)$ is the purity threshold.

*Interpretation:* $\mathcal{A}_y$ is the **sub-atlas** of charts predominantly associated with class $y$. A chart $k$ belongs to $\mathcal{A}_y$ if, given that a sample routes to chart $k$, the probability of class $y$ exceeds $1 - \epsilon_{\text{purity}}$.

({prf:ref}`def-class-conditioned-potential`) *definition* — **Class-Conditioned Potential**

Given a target class $y \in \mathcal{Y}$, define the semantic potential:

$$
V_y(z, K) := -\beta_{\text{class}} \log P(Y=y \mid K) + V_{\text{base}}(z, K),

$$
where:
- $P(Y=y \mid K) = \text{softmax}(\Theta_{K,:})_y$ with learnable parameters $\Theta \in \mathbb{R}^{N_c \times C}$
- $V_{\text{base}}(z, K)$ is the unconditioned critic ({ref}`Section 2.7 <sec-the-hjb-correspondence>`)
- $\beta_{\text{class}} > 0$ is the **class temperature** (inverse of semantic diffusion)
- Units: $[V_y] = \mathrm{nat}$

*Remark (Chart-to-Class Mapping).* The learnable parameter $\Theta_{k,y}$ represents the log-affinity of chart $k$ for class $y$. After training, $P(Y=y \mid K=k) = \text{softmax}(\Theta_{k,:})_y$ approximates the empirical conditional distribution.

*Remark (Alternative: Empirical Estimation).* Instead of learnable parameters, one may estimate $P(Y|K)$ empirically via exponential moving average:

$$
\hat{P}(Y=y \mid K=k) = \frac{\text{EMA}[\mathbb{I}[Y=y, K=k]]}{\text{EMA}[\mathbb{I}[K=k]]}.

$$
This is non-differentiable w.r.t. chart assignment but more grounded in observations. A hybrid approach initializes learnable $\Theta$ from empirical estimates after warmup.

({prf:ref}`def-region-of-attraction`) *definition* — **Region of Attraction**

The **region of attraction** for class $y$ is:

$$
\mathcal{B}_y := \{z \in \mathcal{Z} : \lim_{t \to \infty} \phi_t(z) \in \mathcal{A}_y\},

$$
where $\phi_t$ denotes the flow of the gradient dynamical system $\dot{z} = -G^{-1}(z)\nabla V_y(z)$.

*Interpretation:* $\mathcal{B}_y$ is the set of initial conditions from which the deterministic gradient flow on $V_y$ converges to the class-$y$ region.

({prf:ref}`def-class-consistent-jump-rate`) *definition* — **Class-Consistent Jump Rate**

For the WFR reaction term (Definition {prf:ref}`def-the-wfr-action`), modulate the inter-chart transition rate:

$$
\lambda_{i \to j}^{\text{sup}} := \lambda_{i \to j}^{(0)} \cdot \exp\left(-\gamma_{\text{sep}} \cdot D_{\text{class}}(i, j)\right),

$$
where:
- $\lambda^{(0)}_{i \to j}$ is the **base transition rate** from the GKSL master equation ({prf:ref}`def-gksl-generator`, {cite}`lindblad1976gksl,gorini1976gksl`, {ref}`Section 20.5 <sec-connection-to-gksl-master-equation>`), derived from the overlap consistency of jump operators (Section 7.13)
- $\gamma_{\text{sep}} \geq 0$ is the **separation strength** (hyperparameter)
- $D_{\text{class}}(i, j) = \mathbb{I}[\text{Class}(i) \neq \text{Class}(j)]$ is the class disagreement indicator
- $\text{Class}(k) := \arg\max_y P(Y=y \mid K=k)$ is the dominant class of chart $k$

*Remark (Rate vs Operator).* {ref}`Section 7.13 <sec-factorized-jump-operators-efficient-chart-transitions>` defines the **transition function** $L_{i \to j}$ (the coordinate change map). The **transition rate** $\lambda_{i \to j}$ is a separate quantity from the GKSL/master equation framework ({ref}`Section 20.5 <sec-connection-to-gksl-master-equation>`, Equation 20.5.2) that governs *how often* jumps occur, not *where* they go. The rate is typically derived from the overlap structure: $\lambda_{i \to j}^{(0)} \propto \mathbb{E}_{x}[w_i(x) w_j(x)]$, measuring how much probability mass lies in the overlap $U_i \cap U_j$.

*Interpretation:* Transitions between charts of the same class proceed at the base rate $\lambda^{(0)}$. Transitions between charts of different classes are exponentially suppressed by factor $e^{-\gamma_{\text{sep}}}$.

({prf:ref}`def-class-modulated-jump-operator`) *definition* — **Class-Modulated Jump Operator**

Modify the jump operator (Definition {prf:ref}`def-factorized-jump-operator`) to incorporate class consistency:

```python
def class_modulated_jump_rate(
    lambda_base: torch.Tensor,    # [N_c, N_c] base jump rates
    chart_to_class: torch.Tensor, # [N_c, C] learnable logits
    gamma_sep: float = 5.0,       # Separation strength
) -> torch.Tensor:
    """
    Compute class-modulated jump rates.

    Cross-ref:
        - Definition 25.3.1 (Class-Consistent Jump Rate)
        - Definition 7.13.1 (Jump Operator)
    """
    # Get dominant class per chart
    p_y_given_k = F.softmax(chart_to_class, dim=1)  # [N_c, C]
    dominant_class = p_y_given_k.argmax(dim=1)       # [N_c]

    # Compute class disagreement matrix
    class_match = (dominant_class.unsqueeze(1) == dominant_class.unsqueeze(0)).float()  # [N_c, N_c]
    D_class = 1.0 - class_match  # 1 if classes differ, 0 if same

    # Modulate rates
    lambda_sup = lambda_base * torch.exp(-gamma_sep * D_class)

    return lambda_sup
```

**Cross-references:** {ref}`Section 20.2 <sec-the-wfr-metric>` (WFR Metric), Definition {prf:ref}`def-factorized-jump-operator`, {ref}`Section 20.5 <sec-connection-to-gksl-master-equation>` (GKSL Connection).

({prf:ref}`def-purity-loss`) *definition* — **Purity Loss**

The purity loss measures how well charts separate classes:

$$
\mathcal{L}_{\text{purity}} = \sum_{k=1}^{N_c} P(K=k) \cdot H(Y \mid K=k),

$$
where:
- $P(K=k) = \mathbb{E}_{x \sim \mathcal{D}}[w_k(x)]$ is the marginal chart probability
- $H(Y \mid K=k) = -\sum_y P(Y=y \mid K=k) \log P(Y=y \mid K=k)$ is the class entropy within chart $k$

*Interpretation:* $\mathcal{L}_{\text{purity}} = H(Y \mid K)$, the conditional entropy of class given chart. Minimizing this encourages each chart to be associated with a single class.

({prf:ref}`def-balance-loss`) *definition* — **Balance Loss**

Prevent degenerate solutions where all samples route to few charts:

$$
\mathcal{L}_{\text{balance}} = D_{\text{KL}}\left(\bar{w} \;\|\; \text{Uniform}(N_c)\right),

$$
where $\bar{w} = \mathbb{E}_{x \sim \mathcal{D}}[w(x)]$ is the average router weight vector.

*Interpretation:* Encourages all charts to be used, preventing "dead charts" and ensuring the atlas covers the label space.

({prf:ref}`def-contrastive-loss`) *definition* — **Contrastive Loss**

Enforce that different-class samples are geometrically separated:

$$
\mathcal{L}_{\text{metric}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}: y_i \neq y_j} w_i^\top w_j \cdot \max(0, m - d_{\text{jump}}(z_i, z_j))^2,

$$
where:
- $\mathcal{P}$ is the set of sample pairs in the batch
- $w_i, w_j$ are router weight vectors
- $m > 0$ is the margin (minimum desired separation)
- $d_{\text{jump}}(z_i, z_j)$ is the minimum jump cost ({ref}`Section 7.13 <sec-factorized-jump-operators-efficient-chart-transitions>`)

*Interpretation:* If two samples have different labels but high router overlap ($w_i^\top w_j$ large), they must be separated by at least margin $m$ in jump distance. Otherwise, the loss penalizes the configuration.

({prf:ref}`def-route-alignment-loss`) *definition* — **Route Alignment Loss**

The primary classification loss:

$$
\mathcal{L}_{\text{route}} = \mathbb{E}_{x, y_{\text{true}}}\left[\text{CE}\left(\sum_k w_k(x) \cdot P(Y=\cdot \mid K=k), \; y_{\text{true}}\right)\right],

$$
where $\text{CE}$ denotes cross-entropy.

*Interpretation:* The predicted class distribution is the router-weighted average of per-chart class distributions. This must match the true label.

({prf:ref}`def-total-loss`) *definition* — **Total Loss**

The full supervised topology loss:

$$
\mathcal{L}_{\text{sup-topo}} = \mathcal{L}_{\text{route}} + \lambda_{\text{pur}} \mathcal{L}_{\text{purity}} + \lambda_{\text{bal}} \mathcal{L}_{\text{balance}} + \lambda_{\text{met}} \mathcal{L}_{\text{metric}}.

$$
Typical hyperparameters: $\lambda_{\text{pur}} = 0.1$, $\lambda_{\text{bal}} = 0.01$, $\lambda_{\text{met}} = 0.01$.

**Algorithm 25.4.7 (SupervisedTopologyLoss Implementation).**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SupervisedTopologyLoss(nn.Module):
    """
    Supervised topology loss enforcing chart purity, balance, and separation.

    Cross-ref:
        - Definition 25.4.6 (Total Loss)
        - {ref}`Section 7.8 <sec-tier-the-attentive-atlas>` (Router Weights)
    """

    def __init__(
        self,
        num_charts: int,
        num_classes: int,
        lambda_purity: float = 0.1,
        lambda_balance: float = 0.01,
        lambda_metric: float = 0.01,
        margin: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_charts = num_charts
        self.num_classes = num_classes
        self.lambda_purity = lambda_purity
        self.lambda_balance = lambda_balance
        self.lambda_metric = lambda_metric
        self.margin = margin

        # Learnable chart-to-class mapping (Definition 25.2.1)
        self.chart_to_class = nn.Parameter(
            torch.randn(num_charts, num_classes) * 0.01
        )
        self.temperature = temperature

    @property
    def p_y_given_k(self) -> torch.Tensor:
        """P(Y|K) distribution [N_c, C]."""
        return F.softmax(self.chart_to_class / self.temperature, dim=1)

    def forward(
        self,
        router_weights: torch.Tensor,  # [B, N_c]
        y_true: torch.Tensor,          # [B] class labels
        z_latent: torch.Tensor = None, # [B, D] optional for metric loss
    ) -> Dict[str, torch.Tensor]:
        """
        Compute supervised topology losses.

        Returns dict with individual losses and total.
        """
        B = router_weights.shape[0]
        p_y_k = self.p_y_given_k  # [N_c, C]

        # === Route Alignment Loss (Definition 25.4.5) ===
        # P(Y|x) = sum_k w_k(x) * P(Y|K=k)
        p_y_x = torch.matmul(router_weights, p_y_k)  # [B, C]
        loss_route = F.cross_entropy(
            torch.log(p_y_x + 1e-8), y_true
        )

        # === Purity Loss (Definition 25.4.1) ===
        # H(Y|K=k) for each chart
        entropy_per_chart = -(p_y_k * torch.log(p_y_k + 1e-8)).sum(dim=1)  # [N_c]
        # P(K=k) = average router weight
        p_k = router_weights.mean(dim=0)  # [N_c]
        # L_purity = sum_k P(K=k) * H(Y|K=k)
        loss_purity = (p_k * entropy_per_chart).sum()

        # === Balance Loss (Definition 25.4.3) ===
        # KL(p_k || Uniform) = sum_k p_k * log(p_k / (1/N_c)) = sum_k p_k * (log(p_k) + log(N_c))
        uniform = torch.ones_like(p_k) / self.num_charts
        # Manual KL computation: KL(P||Q) = sum P * log(P/Q)
        loss_balance = (p_k * (torch.log(p_k + 1e-8) - torch.log(uniform))).sum()

        # === Metric Contrastive Loss (Definition 25.4.4) ===
        loss_metric = torch.tensor(0.0, device=router_weights.device)
        if self.lambda_metric > 0 and B > 1:
            # Router overlap as proxy for proximity
            # w_i^T w_j measures routing similarity
            overlap = torch.matmul(router_weights, router_weights.t())  # [B, B]

            # Class disagreement mask
            y_match = (y_true.unsqueeze(1) == y_true.unsqueeze(0)).float()
            y_diff = 1.0 - y_match  # 1 if different classes

            # Penalize high overlap for different-class pairs
            # Using overlap as proxy for d_jump (lower overlap ~ larger distance)
            pseudo_dist = 1.0 - overlap  # Rough proxy
            hinge = F.relu(self.margin - pseudo_dist)
            loss_metric = (y_diff * overlap * hinge ** 2).sum() / (y_diff.sum() + 1e-8)

        # === Total Loss ===
        loss_total = (
            loss_route
            + self.lambda_purity * loss_purity
            + self.lambda_balance * loss_balance
            + self.lambda_metric * loss_metric
        )

        return {
            'loss_total': loss_total,
            'loss_route': loss_route,
            'loss_purity': loss_purity,
            'loss_balance': loss_balance,
            'loss_metric': loss_metric,
        }
```

**Cross-references:** {ref}`Section 7.8 <sec-tier-the-attentive-atlas>` (Router Weights), Section 7.13 (Jump Operators), {ref}`Section 3 <sec-diagnostics-stability-checks>` (Diagnostic Nodes).

({prf:ref}`def-class-centroid-in-poincar-disk`) *definition* — **Class Centroid in Poincare Disk**

For the Poincare disk embedding {cite}`nickel2017poincare,ganea2018hnn`, define the class centroid using the **Frechet mean** {cite}`lou2020frechet`:

$$
c_y := \arg\min_{c \in \mathbb{D}} \sum_{x: Y(x)=y} d_{\mathbb{D}}(c, \text{Enc}(x))^2.

$$
This is well-defined since the Poincare disk has negative curvature (unique Frechet means).

**Cross-references:** {ref}`Section 21.2 <sec-policy-control-field>` (Langevin Dynamics), {ref}`Section 21.3 <sec-the-retrieval-texture-firewall>` (Mobius Re-centering), Definition {prf:ref}`prop-so-d-symmetry-at-origin`.

({prf:ref}`def-hierarchical-labels`) *definition* — **Hierarchical Labels**

A **label hierarchy** is a sequence of label spaces:

$$
\mathcal{Y}_0 \twoheadrightarrow \mathcal{Y}_1 \twoheadrightarrow \cdots \twoheadrightarrow \mathcal{Y}_L,

$$
where $\twoheadrightarrow$ denotes a surjection (coarsening). $\mathcal{Y}_0$ are coarse labels (super-categories), $\mathcal{Y}_L$ are fine labels (leaf categories).

*Example:* $\mathcal{Y}_0 = \{\text{Animal}, \text{Vehicle}\}$, $\mathcal{Y}_1 = \{\text{Dog}, \text{Cat}, \text{Car}, \text{Bike}\}$, $\mathcal{Y}_2 = \{\text{Terrier}, \text{Poodle}, \ldots\}$.

({prf:ref}`def-hierarchical-supervised-loss`) *definition* — **Hierarchical Supervised Loss**

The total hierarchical loss:

$$
\mathcal{L}_{\text{hier}} = \sum_{\ell=0}^{L} \alpha_\ell \left(\mathcal{L}_{\text{route}}^{(\ell)} + \lambda_{\text{pur}} \mathcal{L}_{\text{purity}}^{(\ell)}\right),

$$
where $\alpha_\ell$ weights the contribution of each scale (typically $\alpha_\ell = 1$ or decaying with $\ell$).

**Cross-references:** {ref}`Section 7.12 <sec-stacked-topoencoders-deep-renormalization-group-flow>` (Stacked TopoEncoder), Definition {prf:ref}`def-the-peeling-step`, {ref}`Section 7.12.3 <sec-rigorous-interpretation-renormalization-group-flow>` (RG Interpretation).

({prf:ref}`def-the-meta-control-problem`) *definition* — **The Meta-Control Problem**

Let $\theta_t \in \mathcal{M}_\Theta$ be the agent parameters at training step $t$. The meta-control problem is: find a policy $\pi_{\mathfrak{G}}$ that selects hyperparameters $\Lambda_t$ to minimize task loss while satisfying the Sieve constraints.

**Cross-references:** {ref}`Section 3.5 <sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration>` (Adaptive Multipliers), Section 3.4 (Joint Optimization).

({prf:ref}`def-uncontrolled-dynamics`) *definition* — **Uncontrolled Dynamics**

Standard gradient descent defines a discrete flow on $\mathcal{M}_\Theta$:

$$
\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}_{\text{task}}(\theta_t),

$$
where $\eta > 0$ is the step size.

Units: $[\theta] = \text{parameter units}$, $[\eta] = \text{step}^{-1}$, $[\nabla\mathcal{L}] = \text{nat} \cdot [\theta]^{-1}$.

({prf:ref}`def-constrained-dynamics`) *definition* — **Constrained Dynamics**

The Fragile Agent imposes $K$ constraints $\{C_k(\theta) \leq 0\}_{k=1}^K$ defined by the Sieve ({ref}`Section 3.1 <sec-theory-thin-interfaces>`). Each $C_k$ corresponds to a diagnostic node:

$$
C_k(\theta) = \text{Node}_k(\theta) - \epsilon_k,

$$
where $\epsilon_k$ is the tolerance threshold. The learning dynamics must satisfy these constraints throughout training.

({prf:ref}`def-controlled-update-law`) *definition* — **Controlled Update Law**

The controlled update with adaptive multipliers is:

$$
\theta_{t+1} = \theta_t - \eta_t \left( G^{-1}(\theta_t) \nabla \mathcal{L}_{\text{task}}(\theta_t) + \sum_{k=1}^K \lambda_{k,t} \nabla C_k(\theta_t) \right),

$$
where:
- $G(\theta)$ is the parameter-space metric (cf. natural gradient, {ref}`Section 2.5 <sec-second-order-sensitivity-value-defines-a-local-metric>`)
- $\eta_t$ is the adaptive learning rate
- $\lambda_{k,t} \geq 0$ are the constraint multipliers

Units: $[\lambda_k] = \text{dimensionless}$.

*Remark (Natural Gradient Connection).* The factor $G^{-1}$ applies preconditioning analogous to Fisher Information in natural gradient methods {cite}`amari1998natural`. This ensures updates are measured in information-geometric units rather than Euclidean units.

**Cross-references:** {ref}`Section 2.5 <sec-second-order-sensitivity-value-defines-a-local-metric>` (State-Space Metric), Section 3.1 (Diagnostic Nodes).

({prf:ref}`def-diagnostic-state-space`) *definition* — **Diagnostic State Space**

The Governor observes the **Sieve Residuals** via the constraint evaluation map $\Psi: \mathcal{M}_\Theta \to \mathbb{R}^K$:

$$
s_t = \Psi(\theta_t) = [C_1(\theta_t), \ldots, C_K(\theta_t)]^\top.

$$
The components of $s_t$ are the normalized defect functionals corresponding to diagnostic nodes 1–41 ({ref}`Section 3.1 <sec-theory-thin-interfaces>`). Positive values indicate constraint violation.

Units: $[s_t] = \text{nat}$ (for entropy-based nodes) or dimensionless (for normalized defects).

({prf:ref}`def-the-universal-governor`) *definition* — **The Universal Governor**

The Governor is a policy $\pi_{\mathfrak{G}}: \mathbb{R}^{K \times H} \to \mathbb{R}_+^{K+2}$ mapping the history of Sieve residuals to control inputs:

$$
\Lambda_t = \pi_{\mathfrak{G}}(s_t, s_{t-1}, \ldots, s_{t-H}; \phi),

$$
where:
- $\Lambda_t = (\eta_t, \lambda_{1,t}, \ldots, \lambda_{K,t}, T_{c,t}) \in \mathbb{R}_+^{K+2}$, where $T_c$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`)
- $\phi$ are the learnable parameters of the Governor
- $H$ is the history horizon (temporal context)

Units: $[\eta_t] = \text{step}^{-1}$, $[\lambda_{k,t}] = \text{dimensionless}$, $[T_{c,t}] = \text{nat}$.

*Remark (Temporal Processing).* The Governor processes a window of $H$ diagnostic snapshots. This enables detection of first and second differences $\Delta s_t$, $\Delta^2 s_t$, which are required for PID-like control (Proposition {prf:ref}`prop-subsumption-of-section`).

({prf:ref}`def-inner-problem-agent-optimization`) *definition* — **Inner Problem: Agent Optimization**

Given fixed control $\Lambda$, the agent minimizes the regularized objective:

$$
\theta^*(\Lambda) = \arg\min_{\theta} \left[ \mathcal{L}_{\text{task}}(\theta) + \sum_{k=1}^K \lambda_k C_k(\theta) \right].

$$

({prf:ref}`def-outer-problem-governor-optimization`) *definition* — **Outer Problem: Governor Optimization**

The Governor minimizes the **Training Regret** over the distribution of tasks $\mathcal{T}$:

$$
J(\phi) = \mathbb{E}_{\mathcal{T} \sim P(\mathcal{T})} \left[ \sum_{t=0}^T \left( \mathcal{L}_{\text{task}}(\theta_t) + \gamma_{\text{viol}} \sum_{k=1}^K \text{ReLU}(C_k(\theta_t))^2 \right) \right],

$$
subject to: $\theta_{t+1} = \Phi(\theta_t, \pi_{\mathfrak{G}}(\Psi(\theta_t); \phi))$.

Units: $[J] = \text{nat}$, $[\gamma_{\text{viol}}] = \text{dimensionless}$.

The outer objective penalizes cumulative task loss (convergence speed) and squared constraint violations (feasibility). The weight $\gamma_{\text{viol}}$ trades off these two objectives.

({prf:ref}`def-training-lyapunov-function`) *definition* — **Training Lyapunov Function**

Define the candidate Lyapunov function for the training dynamics:

$$
V_{\mathfrak{L}}(\theta) = \mathcal{L}_{\text{task}}(\theta) + \sum_{k=1}^K \frac{\mu_k}{2} \max(0, C_k(\theta))^2,

$$
where $\mu_k > 0$ are penalty weights for constraint violations.

Units: $[V_{\mathfrak{L}}] = \text{nat}$, $[\mu_k] = \text{dimensionless}$.

$V_{\mathfrak{L}}$ is the augmented Lagrangian with quadratic penalty. If $\Delta V_{\mathfrak{L}} < 0$ along the training trajectory, training converges (Theorem {prf:ref}`thm-stable-training-trajectory`).

({prf:ref}`def-canonical-obstruction-suite`) *definition* — **Canonical Obstruction Suite**

A distribution of synthetic optimization landscapes $\{\mathcal{L}_{\text{syn}}^{(i)}\}$ constructed to elicit specific failure modes:

| Obstruction            | Hessian Property                          | Failure Mode            | Diagnostic Signal                              | Required Correction                        |
|------------------------|-------------------------------------------|-------------------------|------------------------------------------------|--------------------------------------------|
| **Rosenbrock Valley**  | $\kappa(\nabla^2\mathcal{L}) \gg 1$       | Oscillation             | High $\lVert\nabla\mathcal{L}\rVert$ variance  | Reduce $\eta$ (gain scheduling)            |
| **Saddle Point**       | $\lambda_{\min}(\nabla^2\mathcal{L}) < 0$ | Stagnation              | Low $\lVert\nabla\mathcal{L}\rVert$, flat loss | Increase $T_c$ (entropy injection)         |
| **Disconnected Modes** | Multimodal landscape                      | Mode collapse           | $H(K) \to 0$                                   | Increase jump rate $\lambda_{\text{jump}}$ |
| **Noise Floor**        | High aleatoric uncertainty                | Overfitting             | $I(K; Z_{\text{tex}}) > 0$                     | Texture firewalling                        |
| **Constraint Cliff**   | Sharp constraint boundary                 | Oscillation at boundary | $C_k$ sign changes                             | Increase $\mu_k$ (barrier strength)        |

*Remark (Training Protocol).* The Governor is trained via reinforcement learning on this suite, with reward $r_t = -\Delta V_{\mathfrak{L}}$. Episodes terminate when $V_{\mathfrak{L}}$ plateaus or diverges.

({prf:ref}`def-historical-record`) *definition* — **Historical Record**

Let $\gamma: [0, T] \to \mathcal{Z}$ be the agent's trajectory on the latent manifold $(\mathcal{Z}, G)$ over time interval $[0, T]$. The *historical record* is the pair $(\gamma, \alpha)$ where $\alpha: [0, T] \to \mathbb{R}$ is the reward flux along the trajectory (Definition {prf:ref}`def-the-reward-flux`).

*Units:* $[\gamma(t)] = [z]$, $[\alpha(t)] = \text{nat}/[s]$.

*Cross-reference:* This connects to Memory Time $t' < t$ (Definition 1.3.4).

({prf:ref}`def-memory-screen`) *definition* — **Memory Screen**

The *memory screen* is the signed measure on $\mathcal{Z}$ defined by

$$
\Xi_T := \int_0^T \alpha(t') \, \delta_{\gamma(t')} \, dt',

$$
where:
- $\delta_{\gamma(t')}$ is the Dirac measure concentrated at $\gamma(t') \in \mathcal{Z}$,
- $\alpha(t') = J_r(t')$ is the (signed) reward flux at time $t'$ (Definition {prf:ref}`def-the-reward-flux`).

*Units:* $[\Xi_T] = \text{nat}$ (total signed measure), $[\alpha] = \text{nat}/[s]$ (reward flux rate).

*Interpretation:* $\Xi_T$ encodes where the agent has been, weighted by the sign and magnitude of reward received. Positive rewards contribute positive measure (attractive memory); negative rewards contribute negative measure (repulsive memory).

*Cross-reference (Relativistic Multi-Agent):* In Chapter 29, the Memory Screen is elevated from an auxiliary construct to a **primary state variable**. The Causal Bundle $\mathcal{Z}_{\text{causal}} := \mathcal{Z}^{(N)} \times \Xi_{<t}$ restores the Markov property in relativistic multi-agent settings where finite information speed creates non-Markovian dynamics. See Definition {prf:ref}`def-causal-bundle`.

({prf:ref}`def-memory-kernel-via-heat-equation`) *definition* — **Memory Kernel via Heat Equation {cite}`grigoryan2009heat,rosenberg1997laplacian`**

The canonical memory kernel is the *Heat Kernel* $H_\tau(z, z')$ on $(\mathcal{Z}, G)$, defined as the fundamental solution to the heat equation:

$$
(\partial_\tau - \Delta_G) H_\tau(z, z') = 0, \quad H_0(z, z') = \delta(z - z'),

$$
where:
- $\tau > 0$ is the *diffusion time* (memory smoothing scale),
- $\Delta_G = G^{ij}\nabla_i\nabla_j$ is the Laplace-Beltrami operator on $(\mathcal{Z}, G)$ (Definition 2.5.3).

*Units:* $[H_\tau] = [z]^{-d}$ (probability density), $[\tau] = [z]^2$ (diffusion time in geometric units).

*Interpretation:* $H_\tau(z, z')$ measures how much influence a memory at $z'$ has on the current position $z$ after diffusion time $\tau$. Larger $\tau$ yields smoother, more diffuse memory influence. For compact manifolds, $H_\tau$ admits an eigenfunction expansion; for non-compact manifolds with bounded geometry, Gaussian upper bounds hold {cite}`grigoryan2009heat`.

({prf:ref}`def-memory-potential`) *definition* — **Memory Potential**

The *memory potential* is defined by

$$
\Psi_{\text{mem}}(z) := -\int_{\mathcal{Z}} H_\tau(z, z') \, d\Xi_T(z').

$$
Expanding using Definition {prf:ref}`def-memory-screen`:

$$
\Psi_{\text{mem}}(z) = -\int_0^T \alpha(t') H_\tau(z, \gamma(t')) \, dt'.

$$
*Units:* $[\Psi_{\text{mem}}] = \text{nat}$.

*Interpretation:* The memory potential is the convolution of the heat kernel with the signed reward-weighted trajectory measure. Since $\Xi_T$ is a signed measure:
- Near high-reward past positions ($\alpha > 0$): $\Psi_{\text{mem}} < 0$, creating a potential well. The force $-\nabla_G \Psi_{\text{mem}}$ points toward the memory (attractive).
- Near high-penalty past positions ($\alpha < 0$): $\Psi_{\text{mem}} > 0$, creating a potential barrier. The force $-\nabla_G \Psi_{\text{mem}}$ points away from the memory (repulsive).

The sign convention ensures that the drift $-G^{-1}\nabla \Psi_{\text{mem}}$ moves toward rewarding experiences and away from penalizing ones.

({prf:ref}`def-memory-augmented-geodesic-sde`) *definition* — **Memory-Augmented Geodesic SDE**

The memory-augmented dynamics on $(\mathcal{Z}, G)$ are:

$$
dz^k = \left[ -G^{kj}\partial_j\bigl(\Phi_{\text{eff}} + \Psi_{\text{mem}}\bigr) + u_\pi^k \right] ds - \Gamma^k_{ij}\dot{z}^i\dot{z}^j\,ds + \sqrt{2T_c}\,(G^{-1/2})^{kj}\,dW^j_s,

$$
where:
- $\Phi_{\text{eff}}$ is the effective potential (Definition {prf:ref}`def-effective-potential`),
- $\Psi_{\text{mem}}$ is the memory potential (Definition {prf:ref}`def-memory-potential`),
- $\Gamma^k_{ij}$ are the Christoffel symbols of $G$ (Definition 2.5.1),
- $u_\pi^k$ is the policy control field (Definition {prf:ref}`def-the-control-field`),
- $T_c$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`, {ref}`Section 22.4 <sec-the-geodesic-baoab-integrator>`),
- $W^j_s$ is a standard Wiener process.

*Cross-reference:* Definition {prf:ref}`def-bulk-drift-continuous-flow`.

*Units:* All terms have units $[z]/[s]$.

({prf:ref}`def-memory-augmented-reaction-diffusion`) *definition* — **Memory-Augmented Reaction-Diffusion**

The WFR dynamics with memory are:

$$
\partial_s \rho + \nabla \cdot (\rho \mathbf{v}) = \rho \left(\frac{\Phi_{\text{eff}} + \Psi_{\text{mem}} - \bar{\Phi}_{\text{aug}}}{T_c}\right),

$$
where:
- $\rho(z, s)$ is the belief density,
- $\mathbf{v} = -G^{-1}\nabla(\Phi_{\text{eff}} + \Psi_{\text{mem}}) + u_\pi$ is the augmented drift,
- $\bar{\Phi}_{\text{aug}} = \int_{\mathcal{Z}} (\Phi_{\text{eff}} + \Psi_{\text{mem}}) \rho \, d\mu_G$ is the mean augmented potential.

*Cross-reference:* Definition {prf:ref}`def-the-wfr-action`, Theorem {prf:ref}`thm-wfr-consistency-value-creates-mass`.

*Units:* $[\partial_s \rho] = [z]^{-d}/[s]$, all terms balance.

({prf:ref}`def-non-locality-ratio`) *definition* — **Non-Locality Ratio**

The *non-locality ratio* at position $z$ is:

$$
\Omega_{\text{mem}}(z) := \frac{\|\nabla_G \Psi_{\text{mem}}(z)\|_G}{\|\nabla_G \Phi_{\text{eff}}(z)\|_G + \epsilon},

$$
where $\epsilon > 0$ is a regularization constant preventing division by zero.

*Units:* $[\Omega_{\text{mem}}] = \text{dimensionless}$.

**Heuristic 27.5.2 (Homeostatic Bound on Memory).** For stable operation, the non-locality ratio should satisfy:

$$
\Omega_{\text{mem}} \in [\Omega_{\min}, \Omega_{\max}],

$$
with empirically recommended bounds $\Omega_{\min} \approx 0.01$, $\Omega_{\max} \approx 10$. These bounds are task-dependent and should be tuned based on the environment's stationarity.

*Boundary cases:*
- $\Omega_{\text{mem}} \to 0$: Pure Markovian dynamics; agent exhibits catastrophic forgetting.
- $\Omega_{\text{mem}} \to \infty$: Pure memory-driven dynamics; agent overfits to historical experience and fails to respond to current environmental gradients.

*Cross-reference:* The Governor ({ref}`Section 26 <sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller>`) can regulate $\Omega_{\text{mem}}$ by adjusting the memory smoothing scale $\tau$ or the reward flux weighting in $\alpha(t')$.

({prf:ref}`def-external-knowledge-manifold`) *definition* — **External Knowledge Manifold**

Let $\mathcal{Z}_{\text{ext}}$ denote the external knowledge manifold equipped with metric $G_{\text{ext}}$, structured as a fiber bundle:

$$
\mathcal{Z}_{\text{ext}} = \mathcal{K} \times \mathcal{Z}_n \times \mathcal{Z}_{\text{tex}},

$$
where $\mathcal{K}$ is the macro-concept space, $\mathcal{Z}_n$ the nuisance coordinates, and $\mathcal{Z}_{\text{tex}}$ the texture fiber.

*Units:* $[G_{\text{ext},ij}] = [z]^{-2}$ (matching the internal metric).

*Cross-reference:* This decomposition mirrors {ref}`Section 2.8 <sec-conditional-independence-and-sufficiency>`'s latent structure $(K, z_n, z_{\text{tex}})$ and {ref}`Section 7.8 <sec-tier-the-attentive-atlas>`'s Atlas architecture.

({prf:ref}`def-knowledge-atom`) *definition* — **Knowledge Atom**

A *knowledge atom* is a triple $\xi = (K, z_n, z_{\text{tex}}) \in \mathcal{Z}_{\text{ext}}$ where:
- $K \in \mathcal{K}$: macro-concept (topic, entity class, logical category)
- $z_n \in \mathcal{Z}_n$: nuisance coordinates (style, formatting, source metadata)
- $z_{\text{tex}} \in \mathcal{Z}_{\text{tex}}$: high-frequency texture (specific wording, surface form)

*Cross-reference:* Compare {ref}`Section 2.8 <sec-conditional-independence-and-sufficiency>`'s decomposition. The macro closure mechanism (Definition 2.8.1) applies equally to external atoms.

({prf:ref}`def-hyperbolic-geodesic-distance`) *definition* — **Hyperbolic Geodesic Distance**

For points $z, \xi \in \mathbb{D}^d$ (the Poincare disk), the geodesic distance is:

$$
d_{\mathbb{D}}(z, \xi) = \operatorname{acosh}\left(1 + \frac{2\|z - \xi\|^2}{(1 - \|z\|^2)(1 - \|\xi\|^2)}\right).

$$
*Units:* $[d_{\mathbb{D}}] = [z]$ (dimensionless in Poincare coordinates).

*Cross-reference:* This is the distance function induced by the Poincare metric $G_{ij}$ (Definition {prf:ref}`def-hyperbolic-volume-growth`). See also Definition {prf:ref}`prop-isotropic-radial-expansion` for the hyperbolic potential $U(z) = -2\operatorname{artanh}(\|z\|)$.

({prf:ref}`def-retrieval-measure-via-geodesic-functional`) *definition* — **Retrieval Measure via Geodesic Functional**

Given a query position $z \in \mathcal{Z}_{\text{int}}$ and archive prior $\mu_{\mathcal{E}} \in \mathcal{P}(\mathcal{Z}_{\text{ext}})$, the *retrieval measure* is:

$$
\nu_\omega = \arg\min_{\nu \in \mathcal{P}(\mathcal{Z}_{\text{ext}})} \left\{ \int d_{\mathbb{D}}(z, \xi) \, d\nu(\xi) + T_{\text{ret}} D_{\text{KL}}(\nu \| \mu_{\mathcal{E}}) \right\},

$$
where $T_{\text{ret}} > 0$ is the *retrieval temperature*.

*Units:* $[T_{\text{ret}}] = \text{nat}$.

*Interpretation:* This variational problem balances semantic proximity (first term) against prior plausibility (KL term). At $T_{\text{ret}} \to 0$, retrieval concentrates on the nearest neighbor; at $T_{\text{ret}} \to \infty$, it reverts to the archive prior.

({prf:ref}`def-bulk-projection-operator`) *definition* — **Bulk Projection Operator**

The *bulk projection* $\Pi_{\text{bulk}}: \mathcal{Z}_{\text{ext}} \to \mathcal{K} \times \mathcal{Z}_n$ is defined by:

$$
\Pi_{\text{bulk}}(\xi) = \Pi_{\text{bulk}}(K, z_n, z_{\text{tex}}) := (K, z_n).

$$
*Interpretation:* This projection discards texture, retaining only control-relevant coordinates.

*Cross-reference:* This extends the internal texture exclusion of {ref}`Section 2.8 <sec-conditional-independence-and-sufficiency>` to external retrieval.

({prf:ref}`def-bulk-filtered-retrieval-potential`) *definition* — **Bulk-Filtered Retrieval Potential**

The *retrieval potential* is:

$$
\Psi_{\text{ret}}(z) = -\Lambda_{\text{ret}} \int_{\mathcal{Z}_{\text{ext}}} \exp\left(-\lambda \, d_{\mathbb{D}}(z, \Pi_{\text{bulk}}(\xi))\right) d\nu_\omega(\xi),

$$
with the firewall constraint:

$$
\frac{\partial \Psi_{\text{ret}}}{\partial z_{\text{tex,ext}}} \equiv 0.

$$
*Units:* $[\Psi_{\text{ret}}] = \text{nat}$, $[\Lambda_{\text{ret}}] = \text{nat}$, $[\lambda] = [z]^{-1}$.

*Cross-reference:* Compare the memory potential $\Psi_{\text{mem}}$ (Definition {prf:ref}`def-memory-potential`), which uses heat kernel rather than geodesic exponential. Both generate conservative forces.

({prf:ref}`def-retrieval-augmented-geodesic-sde`) *definition* — **Retrieval-Augmented Geodesic SDE**

The equations of motion with retrieval are:

$$
dz^k = \left[ -G^{kj}\partial_j(\Phi_{\text{eff}} + \Psi_{\text{mem}} + \Psi_{\text{ret}}) + u_\pi^k \right] ds - \Gamma^k_{ij}\dot{z}^i\dot{z}^j\,ds + \sqrt{2T_c}(G^{-1/2})^{kj}dW^j_s,

$$
where:
- $\Phi_{\text{eff}}$: effective potential (Definition {prf:ref}`def-effective-potential`)
- $\Psi_{\text{mem}}$: memory potential (Definition {prf:ref}`def-memory-potential`)
- $\Psi_{\text{ret}}$: retrieval potential (Definition {prf:ref}`def-bulk-filtered-retrieval-potential`)
- $\Gamma^k_{ij}$: Christoffel symbols (Definition 2.5.1, Definition 22.2.1a)
- $u_\pi^k$: policy control field (Definition {prf:ref}`def-the-control-field`)
- $T_c$: cognitive temperature ({ref}`Section 22.4 <sec-the-geodesic-baoab-integrator>`)

*Cross-reference:* This extends the memory-augmented SDE (Definition {prf:ref}`def-memory-augmented-geodesic-sde`) with the retrieval term $\Psi_{\text{ret}}$.

({prf:ref}`def-retrieval-source-term`) *definition* — **Retrieval Source Term**

The Wasserstein–Fisher–Rao continuity equation with retrieval is:

$$
\partial_s \rho + \nabla \cdot (\rho \mathbf{v}) = \rho \, r_{\text{local}}(z) + \sigma_{\text{ret}}(z),

$$
where:
- $r_{\text{local}}(z)$: local mass creation rate (reward-driven, Definition {prf:ref}`def-the-wfr-action`)
- $\sigma_{\text{ret}}(z)$: retrieval source term

The retrieval source is:

$$
\sigma_{\text{ret}}(z) = \eta_{\text{ret}} \cdot \Psi_{\text{ret}}(z) \cdot \mathbf{1}[\Psi_{\text{ret}}(z) > \Psi_{\text{threshold}}],

$$
with $[\sigma_{\text{ret}}] = \text{nat}/[z]^d/\text{step}$.

*Cross-reference:* Compare {ref}`Section 27.4 <sec-wfr-dynamics-with-memory-sources>`'s memory mass creation. Both mechanisms inject mass at non-local locations.

({prf:ref}`def-semantic-vacuum`) *definition* — **Semantic Vacuum**

Let $(\mathbb{D}, G)$ be the Poincare disk with metric $G_{ij}(z) = 4\delta_{ij}/(1-|z|^2)^2$ (Definition {prf:ref}`def-hyperbolic-volume-growth`). The **Semantic Vacuum** is the fiber

$$
\emptyset := \{z \in \mathcal{Z} : |z| = 0\} = \{0\} \times \mathcal{Z}_{\text{tex}},

$$
equipped with the following properties:

1. **$SO(D)$ Symmetry:** At $z=0$, the metric is isotropic $G(0) = 4I$ (Proposition {prf:ref}`prop-so-d-symmetry-at-origin`), and the entropic force vanishes: $F_{\text{entropy}}(0) = 0$. The system has full rotational symmetry $SO(D)$.

2. **Infrared Limit:** For any TopoEncoder scale $\tau$ ({ref}`Section 7.12.3 <sec-rigorous-interpretation-renormalization-group-flow>`), $\lim_{\tau \to 0} z(\tau) = \emptyset$. The vacuum is the coarsest resolution.

3. **Reference Measure:** The vacuum carries the Dirac reference measure $\delta_0$ on the bulk coordinates $(K, z_n)$:

   $$
   \mu_{\emptyset} := \delta_0 \otimes \mathcal{N}(0, \sigma_{\text{tex}}^2 I),

   $$
   where the texture component is drawn from the isotropic prior (Definition {prf:ref}`def-boundary-texture-distribution` with $G^{-1}(0) = I/4$).

4. **Information Content:** At the vacuum, $U(0) = 0$ (Definition {prf:ref}`prop-isotropic-radial-expansion`), corresponding to zero information content (maximum entropy).

*Units:* $[\mu_{\emptyset}]$ is a probability measure; $[U] = \mathrm{nat}$.

*Remark (Unstable Equilibrium).* The vacuum is an **unstable fixed point** of the radial dynamics. From Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`, the parameter $\mu = 1/2 > 0$ implies the origin is unstable: any perturbation grows exponentially until noise or policy breaks the symmetry.

({prf:ref}`def-ontological-stress`) *definition* — **Ontological Stress**

Let $(K_t, z_{n,t}, z_{\text{tex},t})$ be the agent's state at time $t$ (Definition {prf:ref}`def-bounded-rationality-controller`). The **Ontological Stress** is the conditional mutual information:

$$
\Xi := I(z_{\text{tex},t}; z_{\text{tex},t+1} \mid K_t, z_{n,t}, A_t),

$$
where $I(\cdot;\cdot|\cdot)$ denotes conditional mutual information in nats.

*Units:* $[\Xi] = \mathrm{nat}$ (dimensionless information).

*Interpretation.* By Axiom {prf:ref}`ax-bulk-boundary-decoupling` (Bulk-Boundary Decoupling), texture should be unpredictable -- a white-noise residual. If $\Xi > 0$, then texture at time $t$ predicts texture at time $t+1$, conditional on the macro-state and action. This violates the partition condition: the texture channel contains structure that should have been captured by $(K, z_n)$ but was not. The agent's ontology is **too coarse**.

*Cross-reference.* Compare with the closure defect $I(K_{t+1}; Z_t \mid K_t, A_t)$ ({ref}`Section 2.8 <sec-conditional-independence-and-sufficiency>`). Ontological Stress is the dual: predictability *within* texture rather than *from* texture to macro.

({prf:ref}`def-query-fission`) *definition* — **Query Fission**

Let $q_i \in \mathbb{R}^d$ be a chart query vector ({ref}`Section 7.8 <sec-tier-the-attentive-atlas>`) with associated codebook $\{e_{i,c}\}_{c=1}^{N_v}$. A **query fission** replaces $q_i$ with two daughter queries:

$$
q_i \mapsto \{q_i^+, q_i^-\} := \{q_i + \epsilon u, q_i - \epsilon u\},

$$
where $u \in \mathbb{R}^d$ is the **fission direction** (unit vector) and $\epsilon > 0$ is the **fission amplitude**.

The daughter codebooks are initialized as copies:

$$
e_{i^\pm, c} := e_{i, c} \quad \forall c \in \{1, \ldots, N_v\}.

$$
*Selection of fission direction.* The optimal $u$ maximizes the variance of router assignments under the new queries:

$$
u^* = \arg\max_{\|u\|=1} \text{Var}_{x \sim \mathcal{D}}\left[\langle k(x), u \rangle \mid w_i(x) > 1/N_c\right],

$$
i.e., the principal component of keys within the chart's Voronoi cell.

({prf:ref}`def-ontological-ricci-flow`) *definition* — **Ontological Ricci Flow**

Let $G_{ij}(z, s)$ be the capacity-constrained metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) parameterized by flow time $s$. Define the **local stress field** $\Xi(z) := \mathbb{E}[\Xi \mid K = k(z)]$, where $k(z)$ is the chart containing $z$. The **Ontological Ricci Flow** is:

$$
\frac{\partial G_{ij}}{\partial s} = -2\left(R_{ij} - \frac{1}{2}R\, G_{ij} + \Lambda G_{ij} - \kappa T_{ij}\right) + \nu \nabla_i \nabla_j \Xi(z),

$$
where:
- $R_{ij}$ is the Ricci curvature tensor, $R = G^{ij}R_{ij}$ the scalar curvature
- $\Lambda, \kappa$ are constants from Theorem {prf:ref}`thm-capacity-constrained-metric-law`
- $T_{ij}$ is the risk tensor
- $\nu > 0$ is the stress-curvature coupling constant

*Units:* $[\partial G / \partial s] = [z]^{-2}$; $[\Xi] = \text{nat}$; $[\nabla_i \nabla_j \Xi] = \text{nat}/[z]^2$.

*Interpretation.* The first term drives the metric toward the capacity-constrained fixed point. The second term $\nu \nabla_i \nabla_j \Xi$ introduces curvature in regions of high stress gradient, expanding the metric where new distinctions are needed.

({prf:ref}`def-ontological-redundancy`) *definition* — **Ontological Redundancy**

Let $K_i$ and $K_j$ be two charts with associated belief distributions $\mu_i, \mu_j$, transition models $\bar{P}_i, \bar{P}_j$, and value functions $V_i, V_j$. Their **ontological redundancy** is:

$$
\Upsilon_{ij} := \exp\left(-\left[ d_{\text{WFR}}(\mu_i, \mu_j) + D_{\mathrm{KL}}(\bar{P}_i \| \bar{P}_j) + \|V_i - V_j\|_G^2 \right]\right)

$$
where:
- $d_{\text{WFR}}(\mu_i, \mu_j)$ is the Wasserstein-Fisher-Rao distance ({prf:ref}`def-the-wfr-action`) between belief distributions,
- $D_{\mathrm{KL}}(\bar{P}_i \| \bar{P}_j) := \mathbb{E}_{k \sim \mu_i}\left[ D_{\mathrm{KL}}(\bar{P}(\cdot|k, a) \| \bar{P}_j(\cdot|k, a)) \right]$ is the mean predictive divergence,
- $\|V_i - V_j\|_G^2 := \mathbb{E}_{z \sim \mu_i}\left[ (V_i(z) - V_j(z))^2 \right]$ is the mean squared value divergence.

*Units:* Dimensionless; $\Upsilon_{ij} \in [0, 1]$.

*Interpretation:* $\Upsilon_{ij} \to 1$ implies the charts are functionally redundant: they occupy similar regions of belief space, predict similar futures, and assign similar values. $\Upsilon_{ij} \to 0$ implies they are functionally distinct.

({prf:ref}`def-discrimination-gain`) *definition* — **Discrimination Gain**

The **Discrimination Gain** $G_\Delta(i, j)$ is the mutual information the agent loses about observations by merging charts $i$ and $j$:

$$
G_\Delta(i, j) := I(X; \{K_i, K_j\}) - I(X; K_{i \cup j})

$$
where $K_{i \cup j}$ is the merged chart that routes observations previously assigned to $K_i$ or $K_j$ to a single index.

*Units:* nat.

*MDL interpretation:* $G_\Delta$ is the increase in **distortion** (description length) resulting from the merge. If $G_\Delta \approx 0$, the distinction between $K_i$ and $K_j$ carries negligible information about the observation stream.

({prf:ref}`def-query-coalescence`) *definition* — **Query Coalescence**

Given charts $i, j$ satisfying the Fusion Criterion ({prf:ref}`thm-fusion-criterion`), the merged query is the **usage-weighted barycenter**:

$$
q_{\text{merged}} := \frac{\bar{w}_i q_i + \bar{w}_j q_j}{\bar{w}_i + \bar{w}_j}

$$
where $\bar{w}_k := \mathbb{E}[w_k(x)]$ is the historical routing weight from the Attentive Atlas ({prf:ref}`def-attentive-routing-law`).

*Interpretation:* The more frequently used chart contributes more to the merged query position. This preserves the routing behavior for the majority of observations.

({prf:ref}`def-fiber-reconciliation`) *definition* — **Fiber Reconciliation**

Let $L_{j \to i}: \mathcal{F}_j \to \mathcal{F}_i$ be the factorized jump operator ({prf:ref}`def-factorized-jump-operator`). For an observation $x$ previously assigned to chart $j$ with nuisance coordinates $z_n^{(j)}$, the reconciled coordinates in chart $i$ are:

$$
z_n^{(i, \text{reconciled})} := L_{j \to i}(z_n^{(j)}) = A_i(B_j z_n^{(j)} + c_j) + d_i

$$
where $B_j$ is the chart-to-global encoder and $A_i$ is the global-to-chart decoder.

*Codebook reconciliation:* The codebook entries of chart $j$ are projected into chart $i$'s Voronoi structure. Entries that fall within existing Voronoi cells of chart $i$ are absorbed; entries that create new structure may be retained if codebook capacity permits.

({prf:ref}`node-fusion-readiness-check`) *definition* — **Node 54 --- FusionReadinessCheck**

**Component:** Atlas (Chart Router)

**Type:** Metabolic Efficiency

**Interpretation:** Are any two charts functionally redundant?

**Proxy:**

$$
\text{FusionReady} := \mathbb{I}\left[ \max_{i \neq j} \Upsilon_{ij} > \Upsilon_{\text{crit}} \right]

$$
**Computational cost:** $O(N_c^2)$ pairwise comparisons.

**Trigger condition:** Two or more charts have redundancy exceeding threshold.

**Remediation:**
1. Identify most redundant pair $(i^*, j^*) = \arg\max_{i \neq j} \Upsilon_{ij}$.
2. Verify Fusion Criterion ({prf:ref}`thm-fusion-criterion`).
3. If satisfied, initiate subcritical bifurcation dynamics.
4. Execute Query Coalescence and Fiber Reconciliation.
5. Decrement chart count: $N_c \to N_c - 1$.

({prf:ref}`node-codebook-liveness-check`) *definition* — **Node 55 --- CodebookLivenessCheck**

**Component:** Codebook (VQ Layer)

**Type:** Dead Code Detection

**Interpretation:** Are any code indices unused?

**Proxy:**

$$
\text{DeadCodeDetected} := \mathbb{I}\left[ \min_k P(K = k) < \epsilon_{\text{dead}} \right]

$$
where $P(K = k)$ is the empirical usage frequency of code $k$ over a trailing window.

**Computational cost:** $O(|\mathcal{K}|)$.

**Trigger condition:** Code usage falls below minimum threshold (default $\epsilon_{\text{dead}} = 10^{-4}$).

**Remediation:** Execute Lazarus Protocol ({prf:ref}`alg-lazarus`).

*Connection to existing diagnostics:* This node operationalizes the dead-code tolerance constraint from {ref}`Section 3.5.5 <sec-calibrating-tolerances>`: $H(K) \geq \log((1 - \rho_{\text{dead}})|\mathcal{K}|)$.

({prf:ref}`def-intra-symbol-variance`) *definition* — **Intra-Symbol Variance (Geometric Tension)**

For code $e_k$ in chart $i$, the **geometric tension** is:

$$
\sigma_k^2 := \mathbb{E}\left[ \|z_e - e_k\|^2 \;\Big|\; \text{VQ}(z_e) = k \right]

$$
where $z_e$ is the pre-quantized encoder output.

*Units:* $[z]^2$ (squared latent units).

*Interpretation:* High $\sigma_k^2$ indicates the symbol is overloaded---its Voronoi cell contains multiple distinct clusters that should be separated.

({prf:ref}`def-functional-indistinguishability`) *definition* — **Functional Indistinguishability**

Two symbols $k_1, k_2$ within the same chart are fusion candidates if the **policy divergence** and **value gap** are negligible:

$$
\mathcal{D}_f(k_1, k_2) := D_{\mathrm{KL}}\left( \pi(\cdot | k_1) \| \pi(\cdot | k_2) \right) + |V(k_1) - V(k_2)|

$$
If $\mathcal{D}_f(k_1, k_2) < \epsilon_{\text{indist}}$, the distinction provides no **control authority**.

*Units:* nat.

*Interpretation:* Symbols are functionally indistinguishable when the policy and value function treat them identically.

({prf:ref}`def-voronoi-partition`) *definition* — **Symbolic Voronoi Partition**

Let $\mathcal{Z}_i$ be the continuous fiber associated with chart $i$. The codebook $\mathcal{C}_i = \{e_{i,k}\}_{k=1}^{N_v}$ induces a partition $\{\mathcal{V}_k\}$ of $\mathcal{Z}_i$ via:

$$
\mathcal{V}_k := \left\{ z \in \mathcal{Z}_i : d_G(z, e_k) \leq d_G(z, e_j) \;\forall j \neq k \right\}

$$
The probability mass of symbol $k$ is the measure of its Voronoi cell:

$$
P(k) := \int_{\mathcal{V}_k} p(z)\, d\mu_G(z)

$$
where $d\mu_G = \sqrt{\det G}\, dz$ is the Riemannian volume form.

({prf:ref}`def-local-distortion`) *definition* — **Local Distortion Functional**

The **local distortion** of symbol $k$ quantifies the representational error within its Voronoi cell:

$$
\mathcal{D}_k := \int_{\mathcal{V}_k} d_G(z, e_k)^2\, p(z)\, d\mu_G(z)

$$
*Units:* $[z]^2$ (weighted squared geodesic distance).

*Relation to geometric tension:* $\mathcal{D}_k = P(k) \cdot \sigma_k^2$, where $\sigma_k^2$ is the intra-symbol variance ({prf:ref}`def-intra-symbol-variance`).

({prf:ref}`def-symbol-utility`) *definition* — **Symbol Utility Functional**

The **utility** $U_k$ of symbol $k$ measures its contribution to control authority and predictive accuracy:

$$
U_k := P(k) \cdot I(K=k; A) + P(k) \cdot I(K=k; K_{t+1})

$$
where:
- $I(K=k; A)$ is the mutual information between symbol activation and action selection,
- $I(K=k; K_{t+1})$ is the mutual information between symbol activation and next-state prediction.

*Units:* nat.

*Interpretation:* A symbol with $U_k \approx 0$ neither influences actions nor aids prediction---it is **semantically dead** regardless of its usage frequency.

({prf:ref}`def-hyperbolic-frechet-coalescence`) *definition* — **Hyperbolic Frechet Mean for Query Coalescence**

Let $\{q_i\}_{i=1}^k \subset \mathbb{D}$ be a set of chart query vectors with associated usage weights $\bar{w}_i := \mathbb{E}[w_i(x)]$ from the Attentive Atlas ({prf:ref}`def-attentive-routing-law`). The **Intrinsic Merged Query** is:

$$
q_{\text{merged}} := \operatorname*{arg\,min}_{q \in \mathbb{D}} \sum_{i=1}^k \bar{w}_i \cdot d^2_{\mathbb{D}}(q, q_i),

$$
where $d_{\mathbb{D}}(x, y) = \operatorname{arccosh}\left(1 + \frac{2\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\right)$ is the hyperbolic distance.

*Units:* $[q_{\text{merged}}] = [q_i]$ (dimensionless in the unit disk).

*Cross-reference:* This definition supersedes {prf:ref}`def-query-coalescence` for hyperbolic embeddings.

({prf:ref}`def-metabolic-flux`) *definition* — **Metabolic Flux**

Let $\rho(s, z)$ be the belief density evolving in computation time $s$ according to the WFR continuity equation (Definition {prf:ref}`def-the-wfr-action`):

$$
\partial_s \rho + \nabla \cdot (\rho v) = \rho r.

$$
We define the **Metabolic Flux** $\dot{\mathcal{M}}: \mathbb{R}_{\ge 0} \to \mathbb{R}_{\ge 0}$ as:

$$
\dot{\mathcal{M}}(s) := \sigma_{\text{met}} \int_{\mathcal{Z}} \left( \|v_s(z)\|_G^2 + \lambda^2 |r_s(z)|^2 \right) \rho(s, z) \, d\mu_G,

$$
where:
- $\sigma_{\text{met}} > 0$ is the **metabolic resistance coefficient** (units: nat$\cdot$step)
- $v_s(z)$ is the velocity field at computation time $s$
- $r_s(z)$ is the reaction rate (mass creation/destruction)
- $\lambda$ is the WFR length-scale (Definition {prf:ref}`def-the-wfr-action`)
- $d\mu_G = \sqrt{\det G} \, dz$ is the Riemannian volume form

*Physical interpretation:* The metabolic flux measures the instantaneous rate of energy dissipation required to update the belief distribution. Transport ($\|v\|_G^2$) represents the cost of moving probability mass; reaction ($|r|^2$) represents the cost of creating or destroying mass. The WFR action is the kinetic energy of the belief flow.

({prf:ref}`def-metabolic-potential`) *definition* — **Metabolic Potential**

We define $\Psi_{\text{met}}(s) := \int_0^s \dot{\mathcal{M}}(u) \, du$ as the cumulative metabolic energy dissipated during a single interaction step $t$ for an internal rollout of duration $s$. Units: $[\Psi_{\text{met}}] = \text{nat}$.

({prf:ref}`def-cognitive-carnot-efficiency`) *definition* — **Cognitive Carnot Efficiency**

The **Carnot limit** for cognitive systems is $\eta_{\text{thought}} = 1$, achieved when the belief update is a reversible isothermal process. Real agents operate at $\eta_{\text{thought}} < 1$ due to:
1. **Friction:** Non-optimal transport paths (geodesic deviation)
2. **Irreversibility:** Finite-rate updates (non-quasi-static processes)
3. **Dissipation:** Exploration noise ($T_c > 0$)

({prf:ref}`def-the-interventional-surgery`) *definition* — **The Interventional Surgery**

Let $P(z_{t+1} | z_t, a_t)$ be the transition kernel on the latent manifold $\mathcal{Z}$. We define the **Interventional Operator** $\mathfrak{I}: \mathcal{P}(\mathcal{Z} \times \mathcal{A} \times \mathcal{Z}) \to \mathcal{P}(\mathcal{Z} \times \mathcal{A} \times \mathcal{Z})$—equivalent to Pearl's $do(a_t)$ {cite}`pearl2009causality`—as a surgery on the joint distribution that cuts the incoming edges to the action variable.

Geometrically, $\mathfrak{I}$ transforms the symplectic interface ({ref}`Section 23.1 <sec-the-symplectic-interface-position-momentum-duality>`) from a **Coupled Dirichlet state** (where $z_t$ is clamped by the observation $x_t$) to a **Forced Neumann state** (where $z_{t+1}$ is driven purely by the agent's internal motor impulse $u_\pi$).

Formally, the operator acts by truncated factorization:

$$
P(z' | z, do(a)) := P(z' | z, a),

$$
where the structural mechanism $P(z' | z, a)$ is preserved but $a$ is no longer a function of $z$. For marginal interventional queries:

$$
P(z' | do(a)) = \int_{\mathcal{Z}} P(z' | \tilde{z}, a) P_{\text{pre}}(\tilde{z}) \, d\mu_G(\tilde{z}),

$$
where $P_{\text{pre}}(\tilde{z})$ is the pre-intervention distribution over latent states.

({prf:ref}`def-causal-information-potential`) *definition* — **Causal Information Potential**

Recall the World Model scaling coefficient $\gamma$ ({ref}`Section 3.2 <sec-scaling-exponents-characterizing-the-agent>`). We define the **Causal Information Potential** $\Psi_{\text{causal}}: \mathcal{Z} \times \mathcal{A} \to \mathbb{R}_{\ge 0}$ as the Expected Information Gain (EIG) {cite}`lindley1956measure` regarding the transition parameters $\theta_W$ at state-action pair $(z, a)$:

$$
\Psi_{\text{causal}}(z, a) := \mathbb{E}_{z' \sim \bar{P}(\cdot | z, a)} \left[ D_{\text{KL}} \left( p(\theta_W | z, a, z') \| p(\theta_W | z, a) \right) \right].

$$
Units: $[\Psi_{\text{causal}}] = \text{nat}$.

*Physical interpretation:* $\Psi_{\text{causal}}(z, a)$ measures how much the agent expects to learn about the World Model parameters $\theta_W$ by executing action $a$ from state $z$. High $\Psi_{\text{causal}}$ indicates that the outcome $z'$ is highly informative about the transition dynamics—the agent is uncertain about what will happen, and observing the outcome will resolve significant uncertainty. This is the foundation of Bayesian experimental design {cite}`chaloner1995bayesian`.

({prf:ref}`def-reward-flux-harvesting`) *definition* — **The Reward Flux**

The **Reward Flux** $J_r(t)$ is the instantaneous rate of reward accumulation (Definition {prf:ref}`def-the-reward-flux`):

$$
J_r(t) = \langle \mathcal{R}(z_t), v_t \rangle_G = r_t

$$

where $\mathcal{R}$ is the reward 1-form ({ref}`Section 24.1 <sec-the-reward-field-value-forms-and-hodge-geometry>`) and $v_t = \dot{z}_t$ is the velocity in latent space.

*Units:* $[J_r] = \text{nats/step}$ (information-theoretic) or $[\text{utility/step}]$ (decision-theoretic).

*Interpretation:* A positive reward $r_t > 0$ indicates the agent has navigated to a state with lower environmental entropy—a configuration where resources (food, fuel, safety) are localized and accessible.

({prf:ref}`def-information-utility`) *definition* — **Information Utility**

The **Information Utility** $\mathcal{I}_{\text{util}}(r_t)$ quantifies the actionable information content of the reward signal:

$$
\mathcal{I}_{\text{util}}(r_t) := I(Z_t; R_t) = H[R_t] - H[R_t \mid Z_t]

$$

where $I(Z_t; R_t)$ is the mutual information between the agent's state $Z_t$ and the reward $R_t$.

*Operational interpretation:* This is the reduction in uncertainty about environmental resources achieved by navigating to state $z_t$ and observing reward $r_t$.

*Units:* $[\mathcal{I}_{\text{util}}] = \text{nats}$ (or bits if using $\log_2$).

*Simplification:* When the reward signal is deterministic given state, $H[R_t \mid Z_t] = 0$, so $\mathcal{I}_{\text{util}}(r_t) = H[R_t]$. In practice, we often use the approximation $\mathcal{I}_{\text{util}}(r_t) \approx |r_t|$ for rewards measured in natural units.

({prf:ref}`def-metabolic-transducer`) *definition* — **The Metabolic Transducer Operator**

The **Metabolic Transducer** $\mathfrak{T}_{\text{harvest}}$ is the operator converting the reward flux to free energy flux:

$$
\dot{E}_{\text{in}}(t) = \mathfrak{T}_{\text{harvest}}(r_t) := \eta \cdot k_B T_{\text{env}} \cdot \mathcal{I}_{\text{util}}(r_t)

$$

where:
- $k_B \approx 1.38 \times 10^{-23}$ J/K is **Boltzmann's constant**
- $T_{\text{env}}$ is the **environmental temperature** (Kelvin)
- The product $k_B T_{\text{env}}$ is the **energy-per-nat conversion factor** (Joules/nat)
- $\eta \in [0, 1]$ is the **transduction efficiency** (Carnot-bounded, see Theorem {prf:ref}`thm-carnot-transduction-bound`)
- $\mathcal{I}_{\text{util}}(r_t)$ is the **information utility** of the reward signal (Definition {prf:ref}`def-information-utility`)

*Units:* $[\mathfrak{T}] = \text{Joules/step}$ (power).

*Simplified form:* For dimensionless analysis with $k_B = 1$, we write:

$$
\mathfrak{T}_{\text{harvest}}(r_t) = \eta \cdot T_{\text{env}} \cdot r_t

$$

where $r_t$ is measured in nats.

({prf:ref}`def-internal-battery`) *definition* — **The Internal Battery**

The **Internal Battery** $B(t)$ is a scalar state variable representing the agent's stored free energy:

$$
B: [0, \infty) \to [0, B_{\max}]

$$

where:
- $B_{\max}$ is the maximum storage capacity (Joules)
- $B(0) = B_0$ is the initial endowment

*Units:* $[B] = \text{Joules}$ (energy).

*Interpretation:* The battery represents the agent's capacity for future computation. In biological systems, this corresponds to ATP/glucose reserves; in artificial systems, to available compute budget.

({prf:ref}`def-homeostatic-potential`) *definition* — **The Homeostatic Potential**

The battery level $B(t)$ induces a scalar potential field acting on the policy:

$$
\Phi_{\text{homeo}}(z, B) = \frac{\lambda_{\text{surv}}}{B + \epsilon} \cdot \mathbb{1}[z \in \mathcal{Z}_{\text{food}}]

$$

where:
- $\lambda_{\text{surv}} > 0$ is the **survival weight** (dimensionless priority)
- $\epsilon > 0$ is a regularization constant preventing singularity
- $\mathcal{Z}_{\text{food}} \subset \mathcal{Z}$ is the **food region** (states where $\mathfrak{T}(r) > 0$)

*Units:* $[\Phi_{\text{homeo}}] = [\Phi_{\text{task}}] = \text{nats}$ (log-probability scale).

({prf:ref}`def-waste-heat-flux`) *definition* — **The Waste Heat Flux**

The **Waste Heat Flux** is the rate at which the agent must dump entropy to the environment:

$$
\dot{Q}_{\text{waste}} = (1 - \eta) \cdot \mathfrak{T}_{\text{gross}}(r_t) + \dot{\mathcal{M}}(t)

$$

where $\mathfrak{T}_{\text{gross}} = k_B T_{\text{env}} \cdot \mathcal{I}_{\text{util}}(r_t)$ is the gross transduction before efficiency losses.

*Units:* $[\dot{Q}_{\text{waste}}] = \text{Watts}$ (power).

*Interpretation:* All non-useful energy becomes waste heat that must be radiated to maintain thermal equilibrium.

({prf:ref}`def-thermal-operating-envelope`) *definition* — **The Thermal Operating Envelope**

The agent is **thermally viable** if there exists a steady-state solution to:

$$
\dot{Q}_{\text{waste}}(T_c) = \dot{Q}_{\text{radiate}}(T_c)

$$

with $T_c < T_{\text{env}}$ and $\eta(T_c) > \eta_{\min}$ where $\eta_{\min}$ is the minimum efficiency for survival (from Theorem {prf:ref}`thm-autopoietic-inequality`).

The **Thermal Operating Envelope** is the region in $(T_c, \dot{\mathcal{M}}, \dot{Q}_{\text{radiate}})$ space where this condition holds.

({prf:ref}`def-metric-friction`) *definition* — **Metric Friction**

Let $\phi_{A \to B}: \mathcal{Z}_A \to \mathcal{Z}_B$ be the best-fit map between agent ontologies (the correspondence minimizing distortion). **Metric Friction** is the squared Frobenius norm of the pullback metric distortion:

$$
\mathcal{F}_{AB}(z) := \| G_A(z) - \phi_{A \to B}^* G_B(\phi(z)) \|_F^2

$$

where $\phi^* G_B$ denotes the pullback metric and $\|\cdot\|_F$ is the Frobenius norm.

*Interpretation:* If $\mathcal{F}_{AB} > 0$, the agents disagree on the fundamental geometry of the world—distances, angles, and causal structure. Cooperation becomes impossible because "gradients" point in different directions.

*Units:* $[\mathcal{F}_{AB}] = \text{dimensionless}$ (ratio of metric tensors).

({prf:ref}`def-inter-agent-connection`) *definition* — **The Inter-Agent Connection**

Let agents $A$ and $B$ each possess a nuisance bundle with gauge connection $A_\mu^{(A)}$ and $A_\mu^{(B)}$ respectively (Definition {prf:ref}`def-strategic-connection`). The **Inter-Agent Connection** on the product manifold $\mathcal{Z}_A \times \mathcal{Z}_B$ is:

$$
\mathcal{A}_{AB}^\mu(z_A, z_B) := A_\mu^{(A)}(z_A) \otimes \mathbb{1}_B + \mathbb{1}_A \otimes A_\mu^{(B)}(z_B) + \lambda_{\text{lock}} \mathcal{C}_{AB}^\mu

$$

where:
- $\mathbb{1}_A, \mathbb{1}_B$ are identity operators on the respective bundles
- $\mathcal{C}_{AB}^\mu$ is the **Coupling Connection** encoding the interaction
- $\lambda_{\text{lock}} \geq 0$ is the **Locking Strength**

*Interpretation:* The first two terms represent independent gauge evolution. The third term, proportional to $\lambda_{\text{lock}}$, couples the agents' internal gauges via communication.

({prf:ref}`def-locking-curvature`) *definition* — **The Locking Curvature**

The **Locking Curvature** tensor measuring gauge mismatch between agents is:

$$
\mathcal{F}_{AB}^{\mu\nu} := \partial^\mu \mathcal{A}_{AB}^\nu - \partial^\nu \mathcal{A}_{AB}^\mu - ig_{\text{lock}}[\mathcal{A}_{AB}^\mu, \mathcal{A}_{AB}^\nu]

$$

where $g_{\text{lock}}$ is the inter-agent coupling constant. The **Integrated Friction** (gauge-invariant scalar) is:

$$
\Psi_{\text{sync}} := \int_{\mathcal{Z}_{\text{shared}}} \text{Tr}(\mathcal{F}_{AB}^{\mu\nu} \mathcal{F}_{AB,\mu\nu}) \sqrt{|G_{\text{shared}}|} \, d^D z

$$

*Interpretation:* When $\mathcal{F}_{AB}^{\mu\nu} = 0$, the inter-agent connection is flat—parallel transport is path-independent, meaning the agents' gauge choices are compatible. When $\mathcal{F}_{AB}^{\mu\nu} \neq 0$, the agents disagree on how to "translate" internal states.

({prf:ref}`def-gauge-alignment-order-parameter`) *definition* — **The Gauge Alignment Order Parameter**

The **Gauge Alignment Order Parameter** measuring the relative orientation of agents' internal gauges is:

$$
\phi_{AB}(z) := \text{Tr}(U_A(z) U_B^\dagger(z)) \in \mathbb{C}

$$

where $U_A, U_B \in G_{\text{Fragile}}$ are the local gauge transformations. The **Locking Potential** governing its dynamics is:

$$
\mathcal{V}_{\text{lock}}(\phi_{AB}) = -\mu_{\text{lock}}^2 |\phi_{AB}|^2 + \lambda_{\text{lock}} |\phi_{AB}|^4

$$

where:
- $\mu_{\text{lock}}^2 = \beta - \beta_c$ is the effective mass parameter
- $\beta$ is the interaction coupling strength
- $\beta_c$ is the critical coupling

({prf:ref}`def-message-lie-algebra`) *definition* — **Message as Lie Algebra Element**

A **Message** $m_{A \to B}$ from Agent $A$ to Agent $B$ is an element of the Lie algebra $\mathfrak{g}$ of the gauge group:

$$
m_{A \to B} \in \mathfrak{g} = \text{Lie}(G_{\text{Fragile}}), \quad m = m^a T_a

$$

where $\{T_a\}$ are the generators satisfying $[T_a, T_b] = i f^{abc} T_c$.

*Interpretation:* A message is an **instruction** to apply an infinitesimal gauge transformation. The symbol sequence encodes the coefficients $m^a$. "Understanding" a message means successfully applying $e^{im}$ to one's internal manifold.

({prf:ref}`def-language-channel`) *definition* — **The Language Channel**

The **Language Channel** $\mathcal{L}$ is a low-bandwidth projection of the full gauge algebra:

$$
\mathcal{L}: \mathfrak{g} \to \mathfrak{g}_{\mathcal{L}} \subset \mathfrak{g}

$$

where $\dim(\mathfrak{g}_{\mathcal{L}}) \ll \dim(\mathfrak{g})$. The channel satisfies the bandwidth constraint of Axiom {prf:ref}`ax-finite-communication-bandwidth`.

*Interpretation:* Language cannot transmit the full metric tensor. It projects onto a finite-dimensional subspace—the "expressible" portion of experience.

({prf:ref}`def-translation-operator`) *definition* — **Gauge-Covariant Translation Operator**

The **Translation Operator** $\mathcal{T}_{A \to B}(m)$ induced by message $m$ along path $\gamma_{AB}$ is:

$$
\mathcal{T}_{A \to B}(m) := \exp\left(-ig \int_{\gamma_{AB}} m^a A_\mu^a \, dz^\mu\right) \cdot \mathcal{P}\exp\left(-ig \int_{\gamma_{AB}} A_\mu \, dz^\mu\right)

$$

where:
- The first factor encodes the **message content**
- The second factor is the **Wilson line** (parallel transport)
- $\mathcal{P}$ denotes path-ordering

*Properties:*
1. **Gauge Covariance:** $\mathcal{T}_{A \to B}$ transforms as $U_A \mathcal{T}_{A \to B} U_B^\dagger$
2. **Composition:** $\mathcal{T}_{A \to C} = \mathcal{T}_{B \to C} \circ \mathcal{T}_{A \to B}$
3. **Identity at Locking:** When $A^{(A)} = A^{(B)}$, reduces to pure message action

({prf:ref}`def-semantic-alignment`) *definition* — **Semantic Alignment**

**Understanding** occurs when the message reduces metric friction:

$$
\text{Understanding}(m) \iff \mathcal{F}_{AB}(z; t+\Delta t) < \mathcal{F}_{AB}(z; t)

$$

after Agent $B$ receives and processes message $m$.

*Interpretation:* "Meaning" is not in the symbol $m$, but in the **metric update** $\Delta G_B = G_B(e^{im} \cdot) - G_B(\cdot)$ triggered by $m$. A symbol "means" the geometric transformation it induces in the listener.

({prf:ref}`def-metric-eigendecomposition`) *definition* — **Metric Eigendecomposition**

Decompose the metric tensor into its principal components:

$$
G_A = \sum_{k=1}^{D} \sigma_k^{(A)} v_k^{(A)} \otimes v_k^{(A)}

$$

where $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_D > 0$ are eigenvalues (principal curvatures) and $v_k^{(A)}$ are eigenvectors.

- **Core Concepts:** Components with $\sigma_k > \sigma_{\text{thresh}}$ (high information density)
- **Nuance:** Components with $\sigma_k \leq \sigma_{\text{thresh}}$ (low information density)

({prf:ref}`def-institutional-manifold`) *definition* — **The Institutional Manifold**

The **Institutional Manifold** $\mathcal{Z}_{\text{Inst}}$ is a **Static Reference Manifold** encoding shared conventions (Laws, Dictionaries, Money). Agents lock to the Institution rather than each other:

$$
\mathcal{F}_{A,\text{Inst}} + \mathcal{F}_{B,\text{Inst}} \quad \text{replaces} \quad \mathcal{F}_{AB}

$$

*Scaling:* Institution-mediated locking is $O(N)$ instead of $O(N^2)$.

({prf:ref}`def-n-agent-product-manifold`) *definition* — **N-Agent Product Manifold**

The global configuration space is the product manifold:

$$
\mathcal{Z}^{(N)} := \mathcal{Z}^{(1)} \times \mathcal{Z}^{(2)} \times \cdots \times \mathcal{Z}^{(N)}.

$$
The metric on $\mathcal{Z}^{(N)}$ is the direct sum of individual metrics:

$$
G^{(N)} := \bigoplus_{i=1}^N G^{(i)},

$$
where each $G^{(i)}$ is the capacity-constrained metric from Theorem {prf:ref}`thm-capacity-constrained-metric-law`. In coordinates, this is block-diagonal: if $\mathbf{z} = (z^{(1)}, \ldots, z^{(N)})$ with $z^{(i)} \in \mathbb{R}^{d_i}$, then $G^{(N)}_{\mu\nu}(\mathbf{z}) = G^{(i)}_{ab}(z^{(i)})$ when indices $\mu, \nu$ both lie in agent $i$'s block, and $G^{(N)}_{\mu\nu} = 0$ otherwise.

*Units:* $[G^{(N)}] = [z]^{-2}$.

*Remark (Isolated Agents).* The product metric $G^{(N)}$ describes agents in **isolation**—there is no cross-coupling between $\mathcal{Z}^{(i)}$ and $\mathcal{Z}^{(j)}$. Strategic coupling modifies this to $\tilde{G}^{(N)}$ via the Game Tensor ({ref}`Section 29.6 <sec-the-game-tensor-deriving-adversarial-geometry>`).

({prf:ref}`def-agent-specific-boundary-interface`) *definition* — **Agent-Specific Boundary Interface**

Each agent $i$ possesses its own symplectic boundary $(\partial\mathcal{Z}^{(i)}, \omega^{(i)})$ with:
- **Dirichlet component** (sensors): $\phi^{(i)}(x) = $ observation stream
- **Neumann component** (motors): $j^{(i)}_{\text{motor}}(x) = $ action flux

The boundary conditions follow the structure of Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`–23.1.3, applied per-agent.

*Cross-reference:* {ref}`Section 23.1 <sec-the-symplectic-interface-position-momentum-duality>` (Symplectic Boundary Manifold), Definition {prf:ref}`def-mass-tensor`.

({prf:ref}`def-environment-distance`) *definition* — **Environment Distance**

Let $d_{\mathcal{E}}^{ij}$ denote the **environment distance** between agents $i$ and $j$—the geodesic length in the environment manifold $\mathcal{E}$ that information must traverse. This may differ from the latent distance $d_G(z^{(i)}, z^{(j)})$.

*Examples:*
- **Physical agents:** $d_{\mathcal{E}}^{ij} = $ spatial separation in meters
- **Networked agents:** $d_{\mathcal{E}}^{ij} = $ network hop distance or latency
- **Co-located agents:** $d_{\mathcal{E}}^{ij} = 0$ (shared boundary)

*Units:* $[d_{\mathcal{E}}^{ij}] = $ meters or equivalent environment-specific units.

({prf:ref}`def-causal-interval`) *definition* — **Causal Interval**

The **Causal Interval** between spacetime events $(z^{(i)}, t_i)$ and $(z^{(j)}, t_j)$ is:

$$
\Delta s^2_{ij} := -c_{\text{info}}^2 (t_j - t_i)^2 + (d_{\mathcal{E}}^{ij})^2.

$$
The events are classified as:
- **Timelike** ($\Delta s^2_{ij} < 0$): $|t_j - t_i| > \tau_{ij}$. Causal influence is possible.
- **Spacelike** ($\Delta s^2_{ij} > 0$): $|t_j - t_i| < \tau_{ij}$. No causal influence is possible.
- **Lightlike** ($\Delta s^2_{ij} = 0$): $|t_j - t_i| = \tau_{ij}$. Boundary case.

*Consequence:* If agents $i$ and $j$ are spacelike separated at time $t$, no instantaneous Hamiltonian $H(z^{(i)}_t, z^{(j)}_t)$ can couple their states. Coupling must occur via retarded potentials.

({prf:ref}`def-past-light-cone`) *definition* — **Past Light Cone**

The **Past Light Cone** of Agent $i$ at time $t$ is the set of all agent-time pairs that can causally influence Agent $i$:

$$
\mathcal{C}^-_i(t) := \left\{ (j, t') \in \{1,\ldots,N\} \times \mathbb{R} : t' \leq t - \tau_{ij} \right\}.

$$
The **Future Light Cone** is defined symmetrically:

$$
\mathcal{C}^+_i(t) := \left\{ (j, t') : t' \geq t + \tau_{ij} \right\}.

$$
*Physical interpretation:* Agent $i$ at time $t$ can only receive information from events in $\mathcal{C}^-_i(t)$ and can only influence events in $\mathcal{C}^+_i(t)$. The region outside both cones is causally disconnected.

({prf:ref}`def-retarded-potential`) *definition* — **Retarded Potential (Memory Screen)**

Let $\rho^{(j)}(t, z)$ be the reward/action flux emitted by Agent $j$. The potential perceived by Agent $i$ at position $z$ and time $t$ is the **Retarded Potential**:

$$
\Psi_{\text{ret}}^{(i)}(t, z) = \sum_{j \neq i} \int_{-\infty}^{t} \int_{\mathcal{Z}^{(j)}} G_{\text{ret}}(z, t; \zeta, \tau) \rho^{(j)}(\tau, \zeta) \, d\mu_G(\zeta) \, d\tau,

$$
where $G_{\text{ret}}$ is the **Retarded Green's Function** for the wave operator on the manifold:

$$
G_{\text{ret}}(z, t; \zeta, \tau) \propto \frac{\delta\left((t-\tau) - d_{\mathcal{E}}(z, \zeta)/c_{\text{info}}\right)}{d_{\mathcal{E}}(z, \zeta)^{(D-2)/2}}.

$$

*Interpretation:* Agent $i$ does not perceive Agent $j$'s current state. It perceives the "ghost" of Agent $j$ from time $\tau_{ij} = d_{\mathcal{E}}^{ij}/c_{\text{info}}$ ago.

*Units:* $[\Psi_{\text{ret}}] = \text{nat}$, $[G_{\text{ret}}] = [z]^{-(D-2)/2}[\text{time}]^{-1}$.

({prf:ref}`def-causal-bundle`) *definition* — **Causal Bundle**

The **Causal Bundle** is the augmented state space:

$$
\mathcal{Z}_{\text{causal}} := \mathcal{Z}^{(N)} \times \Xi_{<t},

$$
where:
- $\mathcal{Z}^{(N)} = \prod_i \mathcal{Z}^{(i)}$ is the product configuration space (Definition {prf:ref}`def-n-agent-product-manifold`)
- $\Xi_{<t}$ is the **Memory Screen** restricted to the causal past (Definition {prf:ref}`def-memory-screen`): $\Xi_{<t} = \{(\gamma(t'), \alpha(t')) : t' < t\}$

The **Relativistic State** for Agent $i$ at time $t$ is:

$$
\mathcal{S}^{(i)}_t := \left( z^{(i)}_t, \Xi^{(i)}_{<t} \right),

$$
where $\Xi^{(i)}_{<t}$ stores the history of received retarded potentials over the interval $[t - \tau_{\text{horizon}}, t)$.

*Operational validity:* $\Xi^{(i)}_{<t}$ is locally observable at time $t$. The "true" global state of all agents is hidden, but $\mathcal{S}^{(i)}_t$ is a sufficient statistic for Agent $i$'s optimal policy within its future light cone.

({prf:ref}`def-ghost-interface`) *definition* — **Ghost Interface**

The **Ghost Interface** $\mathcal{G}_{ij}(t)$ between agents $i$ and $j$ at time $t$ is:

$$
\mathcal{G}_{ij}(t) := \partial\mathcal{Z}^{(i)}(t) \times \partial\mathcal{Z}^{(j)}(t - \tau_{ij}),

$$
coupling Agent $i$'s current boundary to Agent $j$'s past boundary, where $\tau_{ij} = d_{\mathcal{E}}^{ij}/c_{\text{info}}$ is the causal delay.

The **Ghost Symplectic Structure** is:

$$
\omega_{\mathcal{G},ij} := \omega^{(i)}(t) \oplus \omega^{(j)}(t - \tau_{ij})\big|_{\mathcal{G}_{ij}}.

$$

*Mechanism:* Agent $i$ couples not to $z^{(j)}_t$, but to the **Ghost State** $\hat{z}^{(j)}_t := z^{(j)}_{t-\tau_{ij}}$—the state of Agent $j$ when the signal was emitted.

*Units:* $[\tau_{ij}] = [\text{time}]$.

({prf:ref}`def-retarded-interaction-potential`) *definition* — **Retarded Interaction Potential**

The **Retarded Interaction Source Density** from Agent $j$ to Agent $i$ is:

$$
\rho^{\text{ret}}_{ij}(\zeta, \tau) := \alpha_{ij} \cdot \rho^{(j)}_r(\zeta, \tau),

$$
where $\rho^{(j)}_r$ is the conservative reward source density for Agent $j$ and $\alpha_{ij} \in \{-1,0,+1\}$ encodes the
strategic relationship (cooperative, independent, adversarial).

The induced **Retarded Interaction Potential** is:

$$
\Phi^{\text{ret}}_{ij}(z^{(i)}, t) = \int_{-\infty}^{t} \int_{\mathcal{Z}^{(j)}} G_{\text{ret}}(z^{(i)}, t; \zeta, \tau)\,
\rho^{\text{ret}}_{ij}(\zeta, \tau)\, d\mu_G(\zeta)\, d\tau,

$$
with $G_{\text{ret}}$ from Definition {prf:ref}`def-retarded-potential`.

*Remark (Ghost limit).* For point sources, this reduces to evaluation at the retarded time (ghost state). The static
ghost-kernel form is a quasi-static approximation of this retarded convolution.

({prf:ref}`def-the-game-tensor`) *definition* — **The Game Tensor**

We define the **Game Tensor** $\mathcal{G}_{ij}^{kl}$ as the cross-Hessian of Agent $i$'s value with respect to Agent $j$'s position:

$$
\mathcal{G}_{ij}^{kl}(z^{(i)}, z^{(j)}) := \frac{\partial^2 V^{(i)}}{\partial z^{(j)}_k \partial z^{(j)}_l}\bigg|_{z^{(j)} = z^{(j)*}},

$$
where $z^{(j)*}$ is Agent $j$'s current position (or expected position under their policy). This tensor measures how sensitive Agent $i$'s value landscape is to Agent $j$'s location.

*Units:* $[\mathcal{G}_{ij}^{kl}] = \text{nat}/[z]^2$.

**Derivation 29.4.2 (The Strategic Metric).** Recall the **Capacity-Constrained Metric Law** (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), where curvature is driven by the Risk Tensor $T_{ab}$. See **{ref}`Appendix E.16 <sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws>`** for the formal derivation of the Strategic Jacobian and Game Tensor using the implicit function theorem.

For Agent $i$, the "risk" includes the **Predictive Volatility** of the adversary $j$. If Agent $i$ updates its state by $\delta z^{(i)}$, and the adversary $j$ responds with $\delta z^{(j)} \approx \mathcal{J}_{ji} \delta z^{(i)}$ (where $\mathcal{J}_{ji}$ is the **Strategic Jacobian**—the best-response derivative, see Definition {prf:ref}`def-strategic-jacobian`), the second-order variation of Agent $i$'s value is:

$$
\delta^2 V^{(i)} = (\delta z^{(i)})^\top \left( \nabla_{z^{(i)}}^2 V^{(i)} + \underbrace{(\nabla_{z^{(j)}} \nabla_{z^{(i)}} V^{(i)}) \mathcal{J}_{ji}}_{\text{Strategic back-reaction}} \right) \delta z^{(i)}.

$$
**Agent $i$'s perceived geometry** is modified by adversarial presence as follows:

1. **Effective metric inflation.** In regions where the strategic back-reaction has positive eigenvalues (adversarial curvature), Agent $i$ perceives an inflated metric:

   $$
   \tilde{G}^{(i)}_{kl}(z) = G^{(i)}_{kl}(z) + \sum_{j \neq i} \beta_{ij} \cdot \mathcal{G}_{ij,kl}(z),

   $$
   where $\mathcal{G}_{ij,kl} = G^{(i)}_{km} G^{(i)}_{ln} \mathcal{G}_{ij}^{mn}$ is the Game Tensor with lowered indices, and $\beta_{ij} > 0$ for adversarial agents, $\beta_{ij} = 0$ for neutral, $\beta_{ij} < 0$ for cooperative.

2. **Geodesic deflection.** The Christoffel symbols acquire correction terms from the metric perturbation:

   $$
   \tilde{\Gamma}^{(i),m}_{kl} = \Gamma^{(i),m}_{kl} + \frac{1}{2}(G^{(i)})^{mn}\left(\nabla_k (\beta \mathcal{G})_{nl} + \nabla_l (\beta \mathcal{G})_{nk} - \nabla_n (\beta \mathcal{G})_{kl}\right),

   $$
   where $(\beta\mathcal{G})_{kl} := \sum_{j \neq i} \beta_{ij} \mathcal{G}_{ij,kl}$.

3. **Risk amplification.** High $\|\mathcal{G}_{ij}\|$ regions correspond to strategic uncertainty. This contributes to the Risk Tensor (Theorem {prf:ref}`thm-capacity-constrained-metric-law`):

   $$
   T^{(i)}_{kl} \to T^{(i)}_{kl} + \gamma_{\text{game}} \sum_{j \neq i} |\beta_{ij}| \cdot \mathcal{G}_{ij,kl}.

   $$
*Physical interpretation:* Adversarial agents effectively "curve" each other's latent space. An agent approaching a contested region experiences increased geodesic resistance (higher mass), making aggressive maneuvers more costly.

**The sign structure** of the Game Tensor $\mathcal{G}_{ij}$ determines the strategic relationship:

| Eigenvalue Structure | $\text{sgn}(\det \mathcal{G}_{ij})$ | Interpretation                                              |
|----------------------|-------------------------------------|-------------------------------------------------------------|
| All positive         | $+$                                 | Adversarial: $j$'s presence increases $i$'s value curvature |
| All negative         | $(-1)^d$                            | Cooperative: $j$'s presence smooths $i$'s value landscape   |
| Mixed signs          | varies                              | Mixed-motive game                                           |
| Near-zero            | $\approx 0$                         | Weakly coupled (near-independent)                           |

The trace $\operatorname{tr}(\mathcal{G}_{ij}) = \sum_k \mathcal{G}_{ij}^{kk}$ measures **total strategic sensitivity**: how much Agent $i$'s value curvature depends on Agent $j$'s position. Large $|\operatorname{tr}(\mathcal{G}_{ij})|$ indicates high strategic coupling; small trace indicates approximate independence.

*Cross-reference:* The Game Tensor generalizes the conformal factor $\Omega$ (Definition {prf:ref}`def-value-metric-conformal-coupling`) to the multi-agent setting. Where $\Omega$ captured self-induced value curvature, $\mathcal{G}_{ij}$ captures cross-agent value curvature.

*Cross-reference (Gauge-Covariant Version):* When local gauge invariance is imposed ({ref}`Section 29.13 <sec-local-gauge-symmetry-nuisance-bundle>`), the Game Tensor acquires a gauge-covariant form $\tilde{\mathcal{G}}_{ij}^{kl} := D_k D_l V^{(i)}|_{z^{(j)}}$ using covariant derivatives. Under gauge transformation $U(z)$, the covariant Game Tensor transforms homogeneously: $\tilde{\mathcal{G}}'_{ij} = U \tilde{\mathcal{G}}_{ij} U^\dagger$. See Definition {prf:ref}`def-gauge-covariant-game-tensor`.

({prf:ref}`def-retarded-game-tensor`) *definition* — **Retarded Game Tensor**

Under finite information speed $c_{\text{info}}$, the Game Tensor acquires a **retarded component**. The **Retarded Game Tensor** is:

$$
\mathcal{G}_{ij}^{kl,\text{ret}}(z^{(i)}, t) := \frac{\partial^2 V^{(i)}}{\partial z^{(j)}_k \partial z^{(j)}_l}\bigg|_{z^{(j)} = \hat{z}^{(j)}_t},

$$
where $\hat{z}^{(j)}_t = z^{(j)}_{t - \tau_{ij}}$ is the ghost state of Agent $j$ at the retarded time.

The **total effective metric** including retardation is:

$$
\tilde{G}^{(i)}_{kl}(z, t) = G^{(i)}_{kl}(z) + \sum_{j \neq i} \beta_{ij} \cdot \mathcal{G}_{ij,kl}^{\text{ret}}(z, t).

$$

*Consequence (Strategic Hysteresis):* The metric inflation Agent $i$ experiences depends on Agent $j$'s position at the retarded time, not the current time. An agent may enter a region expecting low resistance, only to encounter a "delayed wall" of metric inflation arriving from the opponent's past position.

({prf:ref}`def-joint-wfr-action`) *definition* — **Joint WFR Action (Relativistic)**

The N-agent WFR action on the product space with retarded interactions is:

$$
\mathcal{A}^{(N)}[\boldsymbol{\rho}, \mathbf{v}, \mathbf{r}] = \int_0^T \left[ \sum_{i=1}^N \int_{\mathcal{Z}^{(i)}} \left(\|v^{(i)}\|_{\tilde{G}^{(i)}}^2 + \lambda_i^2 |r^{(i)}|^2 \right) d\rho^{(i)} + \mathcal{V}_{\text{int}}^{\text{ret}}(\boldsymbol{\rho}, t) \right] dt,

$$
where:
- $v^{(i)}$ is the velocity field for Agent $i$'s belief flow
- $r^{(i)}$ is the reaction term (mass creation/destruction)
- $\tilde{G}^{(i)}$ is the game-augmented metric with retarded components (Definition {prf:ref}`def-retarded-game-tensor`)
- $\mathcal{V}_{\text{int}}^{\text{ret}}(\boldsymbol{\rho}, t) = \sum_{i < j} \int \Phi^{\text{ret}}_{ij}(z^{(i)}, t) \, d\rho^{(i)}(z^{(i)}) d\rho^{(j)}(z^{(j)})$ is the retarded interaction energy

*Cross-reference:* Definition {prf:ref}`def-the-wfr-action`, Definition {prf:ref}`def-retarded-interaction-potential`.

({prf:ref}`def-local-gauge-group`) *definition* — **Local Gauge Group**

The **Local Gauge Group** is a compact Lie group $G$ with:

1. **Lie algebra $\mathfrak{g}$:** The tangent space at identity, with generators $\{T_a\}_{a=1}^{\dim(G)}$ satisfying $[T_a, T_b] = if^{abc}T_c$ where $f^{abc}$ are the **structure constants**.

2. **Representation:** The matter fields $\psi^{(i)}$ transform in a representation $\rho: G \to GL(V)$ where $V$ is the representation space.

3. **Position-dependent element:** $U(z) \in G$ for each $z \in \mathcal{Z}$, forming the infinite-dimensional group of gauge transformations $\mathcal{G} := C^\infty(\mathcal{Z}, G)$.

*Standard choices:*
- $G = SO(D)$: Rotations of $D$-dimensional nuisance space
- $G = SU(N)$: Unitary transformations (for complex representations)
- $G = U(1)$: Abelian phase rotations (electromagnetic limit)

*Cross-reference:* The $SO(D)$ symmetry at the origin (Proposition {prf:ref}`prop-so-d-symmetry-at-origin`) is the special case where the stabilizer is trivial.

({prf:ref}`def-matter-field-belief-amplitude`) *definition* — **Matter Field (Belief Amplitude)**

The **Matter Field** for agent $i$ is the complex-valued section

$$
\psi^{(i)}: \mathcal{Z}^{(i)} \times \mathbb{R} \to V

$$

where $V$ is the representation space of $G$. The matter field is related to the belief wave-function by:

$$
\psi^{(i)}(z, t) = \sqrt{\rho^{(i)}(z, t)} \exp\left(\frac{iV^{(i)}(z, t)}{\sigma}\right) \cdot \xi^{(i)}(z)

$$

where:
- $\rho^{(i)}$ is the belief density
- $V^{(i)}$ is the value function
- $\sigma > 0$ is the **cognitive action scale**, $\sigma := T_c \cdot \tau_{\text{update}}$, the information-theoretic analog of Planck's constant (full definition: {prf:ref}`def-cognitive-action-scale` in {ref}`Section 29.21 <sec-the-belief-wave-function-schrodinger-representation>`)
- $\xi^{(i)}(z) \in V$ is the **internal state vector** encoding nuisance orientation

*Units:* $[\psi] = [\text{length}]^{-D/2}$ (probability amplitude density).

*Transformation law:* Under gauge transformation $U(z)$:

$$
\psi'^{(i)}(z, t) = \rho(U(z))\psi^{(i)}(z, t)

$$

where $\rho: G \to GL(V)$ is the representation.

({prf:ref}`def-strategic-connection`) *definition* — **Strategic Connection (Gauge Potential)**

The **Strategic Connection** is a $\mathfrak{g}$-valued 1-form on $\mathcal{Z}$:

$$
A = A_\mu^a T_a \, dz^\mu

$$

where:
- $A_\mu^a(z, t)$ are the **connection coefficients** (real-valued functions)
- $\{T_a\}_{a=1}^{\dim(\mathfrak{g})}$ are the generators of the Lie algebra $\mathfrak{g}$
- $\mu$ indexes spacetime/latent coordinates $(t, z^1, \ldots, z^D)$

*Units:* $[A_\mu] = [\text{length}]^{-1}$ (inverse length, like momentum).

*Interpretation:* The connection $A_\mu$ tells agent $i$ how to "translate" the nuisance interpretation from point $z$ to point $z + dz$. It is the **strategic context** required to compare internal states at different locations.

({prf:ref}`def-covariant-derivative`) *definition* — **Covariant Derivative**

The **Covariant Derivative** acting on matter fields is:

$$
D_\mu = \partial_\mu - igA_\mu

$$

For a matter field $\psi$ in representation $\rho$:

$$
D_\mu\psi = \partial_\mu\psi - igA_\mu^a \rho(T_a)\psi

$$

*Properties:*
1. **Covariant transformation:** $(D_\mu\psi)' = U(D_\mu\psi)$
2. **Leibniz rule:** $D_\mu(\psi\chi) = (D_\mu\psi)\chi + \psi(D_\mu\chi)$
3. **Reduces to partial derivative** when $A_\mu = 0$ (trivial connection)

*Units:* $[D_\mu\psi] = [\psi]/[\text{length}]$.

({prf:ref}`def-gauge-covariant-game-tensor`) *definition* — **Gauge-Covariant Game Tensor**

The **Gauge-Covariant Game Tensor** is defined using covariant derivatives:

$$
\tilde{\mathcal{G}}_{ij}^{kl}(z) := D_k D_l V^{(i)}\big|_{z^{(j)}}

$$

Explicitly:

$$
\tilde{\mathcal{G}}_{ij}^{kl} = \partial_k\partial_l V^{(i)} - ig(\partial_k A_l + \partial_l A_k)V^{(i)} - g^2[A_k, A_l]V^{(i)} + \Gamma^m_{kl}\partial_m V^{(i)}

$$

where $\Gamma^m_{kl}$ are the Christoffel symbols of the strategic metric.

*Properties:*
1. Transforms covariantly: $\tilde{\mathcal{G}}'_{ij} = U\tilde{\mathcal{G}}_{ij}U^\dagger$
2. Reduces to ordinary Game Tensor when $A_\mu = 0$
3. The trace $\text{Tr}(\tilde{\mathcal{G}}_{ij})$ is gauge-invariant

({prf:ref}`def-field-strength-tensor`) *definition* — **Field Strength Tensor (Yang-Mills Curvature)**

The **Field Strength Tensor** is the $\mathfrak{g}$-valued 2-form:

$$
\mathcal{F}_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu]

$$

In components with Lie algebra generators:

$$
\mathcal{F}_{\mu\nu}^a = \partial_\mu A_\nu^a - \partial_\nu A_\mu^a + gf^{abc}A_\mu^b A_\nu^c

$$

where $f^{abc}$ are the structure constants of $\mathfrak{g}$.

*Units:* $[\mathcal{F}_{\mu\nu}] = [\text{length}]^{-2}$ (curvature).

*Special cases:*
- **Abelian ($[A_\mu, A_\nu] = 0$):** $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu$ (electromagnetic field tensor)
- **Non-Abelian:** The commutator term generates **self-interaction** of the gauge field

({prf:ref}`def-strategic-curvature-scalar`) *definition* — **Strategic Curvature Scalar**

The **Strategic Curvature Scalar** is the gauge-invariant contraction:

$$
\mathcal{R}_{\text{strat}} := \text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu}) = \mathcal{F}_{\mu\nu}^a \mathcal{F}^{\mu\nu,a}

$$

where indices are raised with the Lorentzian metric $\eta^{\mu\nu} = \text{diag}(-1, +1, \ldots, +1)$ or the strategic metric $\tilde{G}^{\mu\nu}$.

*Properties:*
- $\mathcal{R}_{\text{strat}} \geq 0$ for compact gauge groups
- $\mathcal{R}_{\text{strat}} = 0$ if and only if $\mathcal{F}_{\mu\nu} = 0$ (flat connection)
- Provides a measure of total strategic tension in a region

({prf:ref}`def-yang-mills-action`) *definition* — **Yang-Mills Action**

The **Yang-Mills Action** for the strategic gauge field is:

$$
S_{\text{YM}}[A] = -\frac{1}{4g^2}\int_{\mathcal{Z} \times \mathbb{R}} \text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})\sqrt{|\tilde{G}|}\,d^{D+1}x

$$

where:
- $\mathcal{F}_{\mu\nu}$ is the field strength tensor (Definition {prf:ref}`def-field-strength-tensor`)
- $\tilde{G}$ is the strategic metric with determinant $|\tilde{G}|$
- $g$ is the coupling constant
- The trace is over Lie algebra indices: $\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu}) = \mathcal{F}_{\mu\nu}^a\mathcal{F}^{\mu\nu,a}$

*Units:* $[S_{\text{YM}}] = \text{nat}$ (action).

*Properties:*
1. **Gauge-invariant:** $S_{\text{YM}}[A'] = S_{\text{YM}}[A]$ under $A \to A'$
2. **Lorentz-invariant:** Covariant under coordinate transformations
3. **Positive semi-definite:** $S_{\text{YM}} \geq 0$ for compact gauge groups

({prf:ref}`def-complete-lagrangian`) *definition* — **Complete Multi-Agent Lagrangian**

The **Complete Multi-Agent Lagrangian** is:

$$
\mathcal{L}_{\text{SMFT}} = \mathcal{L}_{\text{YM}} + \mathcal{L}_{\text{Dirac}} + \mathcal{L}_{\text{Higgs}} + \mathcal{L}_{\text{Yukawa}}

$$

where each sector contributes:

**(i) Yang-Mills Sector (Strategic Gauge Field):**

$$
\mathcal{L}_{\text{YM}} = -\frac{1}{4}\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})

$$

This governs the dynamics of the strategic connection $A_\mu$.

**(ii) Dirac Sector (Belief Matter Field):**

$$
\mathcal{L}_{\text{Dirac}} = \sum_{i=1}^N \bar{\psi}^{(i)}(i\gamma^\mu D_\mu - m_i)\psi^{(i)}

$$

where:
- $\psi^{(i)}$ is the belief spinor for agent $i$
- $\bar{\psi}^{(i)} = \psi^{(i)\dagger}\gamma^0$ is the Dirac adjoint
- $D_\mu = \partial_\mu - igA_\mu$ is the covariant derivative
- $m_i$ is the "bare mass" (intrinsic inertia) of agent $i$

**(iii) Higgs Sector (Value Order Parameter):**

$$
\mathcal{L}_{\text{Higgs}} = |D_\mu\Phi|^2 - V(\Phi)

$$

with the Higgs potential:

$$
V(\Phi) = \mu^2|\Phi|^2 + \lambda|\Phi|^4

$$

where $\Phi$ is the **value order parameter** (a scalar field in a representation of $G$).

**(iv) Yukawa Sector (Strategic Coupling):**

$$
\mathcal{L}_{\text{Yukawa}} = -\sum_{i,j=1}^N y_{ij}\bar{\psi}^{(i)}\Phi\psi^{(j)}

$$

where $y_{ij}$ are the **Yukawa coupling constants** determining how strongly agents couple through the value field.

*Units:* $[\mathcal{L}] = \text{nat}/[\text{length}]^{D+1}$ (Lagrangian density).

({prf:ref}`def-mass-gap`) *definition* — **Mass Gap**

The **Mass Gap** of the strategic Hamiltonian $\hat{H}_{\text{strat}}$ is:

$$
\Delta := \inf\left\{\text{spec}(\hat{H}_{\text{strat}}) \setminus \{E_0\}\right\} - E_0

$$

where $E_0$ is the ground state energy.

*Properties:*
- $\Delta > 0$: **Gapped** spectrum (isolated ground state)
- $\Delta = 0$: **Gapless** spectrum (continuous above ground state)

*Units:* $[\Delta] = \text{nat}/[\text{time}]$ (energy).

({prf:ref}`def-computational-swampland`) *definition* — **The Computational Swampland**

The **Computational Swampland** $\mathcal{S}_{\text{swamp}}$ is the set of all field theories that violate the Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`) at some finite scale:

$$
\mathcal{S}_{\text{swamp}} := \left\{ \mathcal{T} : \exists R < \infty \text{ such that } I_{\text{bulk}}(R) > C_\partial(R) \right\}

$$

Equivalently, $\mathcal{S}_{\text{swamp}}$ consists of theories with Levin Length $\ell_L \to 0$ (infinite information density).

*Properties of Swampland theories:*
1. **Mathematically consistent:** They satisfy internal field-theoretic axioms (Wightman, etc.)
2. **Computationally unrealizable:** No bounded observer can simulate or represent them
3. **Physically pathological:** They require infinite information storage for any finite region

*Landscape vs. Swampland:* Theories with $\ell_L > 0$ and $I_{\text{bulk}} \leq C_\partial$ at all scales constitute the **Computational Landscape**—the set of physically realizable theories.

({prf:ref}`def-inference-hilbert-space`) *definition* — **Inference Hilbert Space**

Let $(\mathcal{Z}, G)$ be the latent manifold with capacity-constrained metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`). The **Inference Hilbert Space** is:

$$
\mathcal{H} := L^2(\mathcal{Z}, d\mu_G), \quad d\mu_G := \sqrt{\det G(z)}\, d^n z,

$$
with inner product:

$$
\langle \psi_1 | \psi_2 \rangle := \int_{\mathcal{Z}} \overline{\psi_1(z)} \psi_2(z)\, d\mu_G(z).

$$
The measure $d\mu_G$ is the **Riemannian volume form**, ensuring coordinate invariance of the inner product.

*Units:* $[\psi] = [z]^{-d/2}$ (probability amplitude density).

*Remark (Coordinate Invariance).* Under a coordinate transformation $z \to z'$, the Jacobian factor $|\partial z/\partial z'|$ cancels with $\sqrt{\det G}$, leaving $\langle \psi_1 | \psi_2 \rangle$ invariant.

({prf:ref}`def-belief-wave-function`) *definition* — **Belief Wave-Function**

Let $\rho(z, s)$ be the belief density from the WFR dynamics (Definition {prf:ref}`def-the-wfr-action`) and $V(z, s)$ be the value function (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`). The **Belief Wave-Function** is the complex amplitude:

$$
\psi(z, s) := \sqrt{\rho(z, s)} \exp\left(\frac{i V(z, s)}{\sigma}\right),

$$
where $\sigma > 0$ is the **Cognitive Action Scale** (Definition {prf:ref}`def-cognitive-action-scale`).

**Decomposition:**
- **Amplitude:** $R(z, s) := \sqrt{\rho(z, s)} = |\psi(z, s)|$
- **Phase:** $\phi(z, s) := V(z, s)/\sigma = \arg(\psi(z, s))$

**Probability Recovery:**

$$
|\psi(z, s)|^2 = \rho(z, s), \quad \int_{\mathcal{Z}} |\psi|^2 d\mu_G = 1.

$$
*Physical interpretation:* The amplitude $R$ encodes "how much" belief mass is at $z$; the phase $\phi$ encodes "which direction" the belief is flowing (via $\nabla V$).

({prf:ref}`def-cognitive-action-scale`) *definition* — **Cognitive Action Scale**

The **Cognitive Action Scale** $\sigma$ is the information-theoretic analog of Planck's constant $\hbar$:

$$
\sigma := T_c \cdot \tau_{\text{update}},

$$
where:
- $T_c$ is the Cognitive Temperature ({prf:ref}`def-cognitive-temperature`, {ref}`Section 22.4 <sec-the-geodesic-baoab-integrator>`), setting the scale of stochastic exploration
- $\tau_{\text{update}}$ is the characteristic belief update timescale

**Equivalent characterizations:**
1. **Entropy-Action Duality:** $\sigma$ relates entropy production to "cognitive action" via $\Delta S = \mathcal{A}/\sigma$
2. **Resolution Limit:** $\sigma \sim \ell_L^2$ where $\ell_L$ is the Levin Length ({ref}`Section 33.2 <sec-saturation-limit>`)
3. **Uncertainty Scale:** $\sigma$ sets the minimum uncertainty product $\Delta z \cdot \Delta p \geq \sigma/2$

*Units:* $[\sigma] = \text{nat} \cdot \text{step} = \text{bit} \cdot \text{step} / \ln 2$.

*Cross-reference:* In the limit $\sigma \to 0$ (zero temperature, infinite precision), the wave-function becomes a delta function concentrated on the optimal trajectory—recovering classical gradient flow.

({prf:ref}`def-bohm-quantum-potential`) *definition* — **Bohm Quantum Potential (Information Resolution Limit)**

The **Bohm Quantum Potential** is:

$$
Q_B(z, s) := -\frac{\sigma^2}{2} \frac{\Delta_G \sqrt{\rho}}{\sqrt{\rho}} = -\frac{\sigma^2}{2} \frac{\Delta_G R}{R},

$$
where $R = \sqrt{\rho}$ is the amplitude.

**Explicit form in terms of $\rho$:**

$$
Q_B = -\frac{\sigma^2}{8\rho^2} \|\nabla_G \rho\|_G^2 + \frac{\sigma^2}{4\rho} \Delta_G \rho.

$$
**Physical interpretation:** $Q_B$ represents the **energetic cost of belief localization**. Regions where $\rho$ has high curvature (sharp belief features) incur an effective potential energy penalty. This prevents the belief from concentrating to delta functions.

**Information-theoretic interpretation:** $Q_B$ enforces the **Levin Length** ({ref}`Section 33.2 <sec-saturation-limit>`) as a resolution limit. The agent cannot represent distinctions finer than $\ell_L \sim \sqrt{\sigma}$.

*Units:* $[Q_B] = \text{nat}$ (same as potential).

*Cross-reference:* In standard quantum mechanics, $Q_B$ is called the "quantum potential" or "Bohm potential." Here it emerges from the information geometry, not fundamental physics.

({prf:ref}`def-joint-inference-hilbert-space`) *definition* — **Joint Inference Hilbert Space**

For $N$ agents with individual Hilbert spaces $\mathcal{H}^{(i)} = L^2(\mathcal{Z}^{(i)}, d\mu_{G^{(i)}})$, the **Joint Inference Hilbert Space** is the tensor product:

$$
\mathcal{H}^{(N)} := \bigotimes_{i=1}^N \mathcal{H}^{(i)} = L^2\left(\mathcal{Z}^{(N)}, d\mu_{G^{(N)}}\right),

$$
where:
- $\mathcal{Z}^{(N)} = \prod_{i=1}^N \mathcal{Z}^{(i)}$ is the product manifold (Definition {prf:ref}`def-n-agent-product-manifold`)
- $d\mu_{G^{(N)}} = \prod_{i=1}^N d\mu_{G^{(i)}}$ is the product measure

Elements $\Psi \in \mathcal{H}^{(N)}$ are functions $\Psi: \mathcal{Z}^{(N)} \to \mathbb{C}$ with:

$$
\|\Psi\|^2 = \int_{\mathcal{Z}^{(N)}} |\Psi(\mathbf{z})|^2 d\mu_{G^{(N)}}(\mathbf{z}) < \infty.

$$
*Notation:* We use uppercase $\Psi$ for joint wave-functions and lowercase $\psi^{(i)}$ for single-agent wave-functions.

({prf:ref}`def-strategic-entanglement`) *definition* — **Strategic Entanglement**

A joint wave-function $\Psi \in \mathcal{H}^{(N)}$ exhibits **Strategic Entanglement** if it cannot be written as a product:

$$
\Psi(z^{(1)}, \ldots, z^{(N)}) \neq \prod_{i=1}^N \psi^{(i)}(z^{(i)}) \quad \text{for any choice of } \psi^{(i)} \in \mathcal{H}^{(i)}.

$$
**Entanglement Entropy:** For a bipartition $\{i\} \cup \{j \neq i\}$, the **Strategic Entanglement Entropy** is:

$$
S_{\text{ent}}(i) := -\text{Tr}\left[\hat{\rho}^{(i)} \ln \hat{\rho}^{(i)}\right],

$$
where $\hat{\rho}^{(i)} = \text{Tr}_{j \neq i}[|\Psi\rangle\langle\Psi|]$ is the **reduced density operator** obtained by partial trace over all agents except $i$.

**Physical interpretation:**
- $S_{\text{ent}}(i) = 0$: Agent $i$ is **disentangled** (can be modeled independently)
- $S_{\text{ent}}(i) > 0$: Agent $i$ is **entangled** with others (cannot be modeled in isolation)
- $S_{\text{ent}}(i) = \ln N$: **Maximal entanglement** (all agents maximally correlated)

*Cross-reference:* The partial trace operation corresponds to the **Information Bottleneck** (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`)—marginalizing over opponents discards strategic correlations.

({prf:ref}`def-strategic-hamiltonian`) *definition* — **Strategic Hamiltonian**

The **Strategic Hamiltonian** on $\mathcal{H}^{(N)}$ is:

$$
\hat{H}_{\text{strat}} := \sum_{i=1}^N \hat{H}^{(i)}_{\text{kin}} + \sum_{i=1}^N \hat{\Phi}^{(i)}_{\text{eff}} + \sum_{i < j} \hat{V}_{ij},

$$
where:
1. **Kinetic terms:** $\hat{H}^{(i)}_{\text{kin}} = -\frac{\sigma_i^2}{2} \Delta_{G^{(i)}}$ (acting on $\mathcal{Z}^{(i)}$ coordinates)
2. **Individual potentials:** $\hat{\Phi}^{(i)}_{\text{eff}}$ (local reward landscape for agent $i$)
3. **Interaction potentials:** $\hat{V}_{ij} = \Phi_{ij}(z^{(i)}, z^{(j)})$ (strategic coupling)

*Notation (Per-Agent Action Scale):* Here $\sigma_i := T_{c,i} \cdot \tau_{\text{update},i}$ is the cognitive action scale for agent $i$, generalizing Definition {prf:ref}`def-cognitive-action-scale`. For **homogeneous** agents with identical cognitive properties, $\sigma_i = \sigma$ for all $i$. For **heterogeneous** agents (e.g., different computation rates), $\sigma_i$ may vary.

*Remark (Separability).* If all $\hat{V}_{ij} = 0$, the Hamiltonian is **separable**: $\hat{H}_{\text{strat}} = \sum_i \hat{H}^{(i)}$, and the ground state is a product $\Psi_0 = \prod_i \psi^{(i)}_0$. Non-zero interaction creates entanglement.

({prf:ref}`def-pareto-barrier`) *definition* — **Pareto Barrier**

A **Pareto Barrier** $\mathcal{B}_P \subset \mathcal{Z}^{(N)}$ is a region where:
1. **Local value decrease:** $\Phi^{(i)}_{\text{eff}}(\mathbf{z}) > \Phi^{(i)}_{\text{eff}}(\mathbf{z}^*)$ for at least one agent $i$ and some starting point $\mathbf{z}^*$
2. **No Nash within:** There exists no Nash equilibrium $\mathbf{z}' \in \mathcal{B}_P$
3. **Separates basins:** $\mathcal{B}_P$ lies between distinct Nash equilibria $\mathbf{z}^*_A$ and $\mathbf{z}^*_B$

The **barrier height** is:

$$
\Delta \Phi_P := \max_{\mathbf{z} \in \mathcal{B}_P} \left[ \sum_{i=1}^N \Phi^{(i)}_{\text{eff}}(\mathbf{z}) - \sum_{i=1}^N \Phi^{(i)}_{\text{eff}}(\mathbf{z}^*_A) \right].

$$
*Mathematical characterization:* A Pareto barrier is a region where the total potential $\sum_i \Phi^{(i)}_{\text{eff}}$ exceeds its value at nearby Nash equilibria. Classical gradient descent with initial condition in the basin of attraction of $\mathbf{z}^*_A$ converges to $\mathbf{z}^*_A$ and cannot reach $\mathbf{z}^*_B$.

({prf:ref}`def-utility-gauge-freedom`) *definition* — **Utility Gauge Freedom**

Let the Belief Wave-Function $\psi(z)$ be defined as in Definition {prf:ref}`def-belief-wave-function`:

$$
\psi(z) = \sqrt{\rho(z)} \exp\left(\frac{i V(z)}{\sigma}\right),

$$

where:
- $\rho(z)$ is the belief density (Definition {prf:ref}`def-belief-density`)
- $V(z)$ is the Value function (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`)
- $\sigma = T_c \cdot \tau_{\text{update}}$ is the Cognitive Action Scale (Definition {prf:ref}`def-cognitive-action-scale`)

The system's observables are:
1. **Probability density:** $\rho = |\psi|^2$
2. **Probability current:** $J^\mu = \text{Im}(\psi^* \partial^\mu \psi) = \frac{\rho}{\sigma} \partial^\mu V$

Both are invariant under the global phase transformation:

$$
\psi(z) \to e^{i\theta} \psi(z), \quad \theta \in \mathbb{R}.

$$

This corresponds to the global gauge invariance of the Value function: $V(z) \to V(z) + \sigma\theta$. The addition of a constant baseline does not alter the policy gradient $\nabla V$.

({prf:ref}`def-cognitive-isospin-multiplet`) *definition* — **The Cognitive Isospin Multiplet (Doublet for r=2)**

We define the **Left-Handed Field** $\Psi_L$ as an isospin $r$-plet (doublet for $r=2$) residing in the fundamental
representation of $SU(r)$:

$$
\Psi_L(x) = \begin{pmatrix} \psi_1(x) \\ \vdots \\ \psi_r(x) \end{pmatrix}

$$

In the minimal $r=2$ case:
- $\psi_1 \equiv \psi_{\text{obs}}$ is the **Observation** channel (the incoming sensory update)
- $\psi_2 \equiv \psi_{\text{act}}^{\text{pre}}$ is the **Pre-commitment Action** channel (the outgoing motor intent)

We define the **Right-Handed Field** $\Psi_R$ as an isospin singlet (invariant under $SU(r)$):

$$
\Psi_R(x) = \psi_{\text{act}}^{\text{commit}}(x)

$$

representing the **Committed Action** plan after mixing and projection.

*Cross-reference:* This decomposition mirrors {ref}`Section 12 <sec-belief-dynamics-prediction-update-projection>`'s Belief Dynamics (Prediction-Update-Projection) and the Kalman filtering structure.

({prf:ref}`def-feature-dimension-parameter`) *definition* — **Feature Dimension Parameter**

The **Feature Dimension** $N_f \in \mathbb{Z}_{>0}$ is the intrinsic dimensionality of the feature representation at each layer of the hierarchical encoder. This parameter is determined by:

1. **Environment Structure:** The minimal basis required to represent distinguishable features in the agent's sensory domain
2. **Computational Constraints:** The capacity allocated to the binding mechanism

**Special Cases:**
- Physics (Standard Model): $N_f = 3$ (spatial dimensions, RGB channels)
- Vision-only agents: $N_f \in \{3, 4\}$ (RGB or RGBA)
- Abstract reasoning agents: $N_f$ determined by the embedding dimension of the domain

*Remark:* The gauge structure $SU(N_f)_C$ emerges for any $N_f \geq 2$.

({prf:ref}`def-feature-color-space`) *definition* — **The Feature Color Space**

Let the nuisance vector $z_n$ at layer $\ell$ of the TopoEncoder be an element of a vector bundle with fiber $\mathbb{C}^{N_f}$, where $N_f$ is the Feature Dimension (Definition {prf:ref}`def-feature-dimension-parameter`). We transform the basis:

$$
\psi_{\text{feature}}(x) \to U(x) \psi_{\text{feature}}(x), \quad U(x) \in SU(N_f)

$$

This symmetry represents the **Internal Basis Invariance** of a concept: an object's identity $K$ is invariant under the mixing of its constituent feature definitions, provided the geometric relationship between them is preserved.

*Justification:* The dimension $N_f$ is determined by the agent's environment and architecture. For physical systems with 3D spatial structure, $N_f = 3$ (e.g., RGB channels, XYZ coordinates). For other agents, $N_f$ may differ based on the intrinsic dimensionality of the sensory domain.

({prf:ref}`def-cognitive-spinor`) *definition* — **The Cognitive Spinor**

The belief state is a spinor field $\Psi(x)$ belonging to the **Inference Hilbert Space** (Definition {prf:ref}`def-inference-hilbert-space`):

$$
\Psi(x) = \begin{pmatrix} \Psi_L(x) \\ \Psi_R(x) \end{pmatrix} \in L^2(\mathcal{M}, \mathbb{C}^4 \otimes \mathbb{C}^{r} \otimes \mathbb{C}^{N_f})

$$

where $\mathbb{C}^4$ is the Dirac spinor space, $\mathbb{C}^r$ is the $SU(r)_L$ mode space (minimal case $r=2$), and
$\mathbb{C}^{N_f}$ is the $SU(N_f)_C$ color space. The components are:
1. **$\Psi_L$ (The Active Multiplet, doublet for $r=2$):** The Left-handed component, transforming as an $r$-plet under
   $SU(r)_L$. It contains the **Prediction** and **Observation** amplitudes in the minimal case (Definition
   {prf:ref}`def-cognitive-isospin-multiplet`).

2. **$\Psi_R$ (The Passive Singlet):** The Right-handed component, invariant under $SU(r)_L$. It contains the **Action**
   intention.

**Probabilistic Interpretation:** The physical probability density (belief mass) is the vector current:

$$
J^\mu = \bar{\Psi} \gamma^\mu \Psi

$$

where $J^0 = \Psi^\dagger \Psi = \rho$ is the probability density (WFR mass from Definition {prf:ref}`def-the-wfr-action`), and $\vec{J}$ is the probability flux. Conservation $\partial_\mu J^\mu = 0$ corresponds to unitarity.

({prf:ref}`def-universal-covariant-derivative`) *definition* — **The Universal Covariant Derivative**

The operator moving the belief spinor through the latent manifold is:

$$
D_\mu = \underbrace{\partial_\mu}_{\text{Change}} - \underbrace{ig_1 \frac{Y}{2} B_\mu}_{U(1)_Y \text{ (Value)}} - \underbrace{ig_2 \frac{\tau^a}{2} W^a_\mu}_{SU(2)_L \text{ (Error)}} - \underbrace{ig_s \frac{\lambda^a}{2} G^a_\mu}_{SU(N_f)_C \text{ (Binding)}}

$$

where $\lambda^a$ ($a = 1, \ldots, N_f^2 - 1$) are the generators of $SU(N_f)$, and:
- **$B_\mu$ (Opportunity Field):** Adjusts the belief for local shifts in the value baseline and path-dependent opportunity
- **$W_\mu$ (Error Field):** Adjusts the belief for the rotation between Prior and Posterior
- **$G_\mu$ (Binding Field):** Adjusts the belief for the permutation of sub-symbolic features

**Operational Interpretation:** The quantity $D_\mu \Psi$ measures the deviation from parallel transport. When $D_\mu \Psi = 0$, the belief state is covariantly constant along the direction $\mu$---all changes are accounted for by the gauge connection. When $D_\mu \Psi \neq 0$, there is a residual force acting on the belief.

({prf:ref}`def-ontological-order-parameter`) *definition* — **The Ontological Order Parameter**

Let the local chart structure at spacetime point $x$ be described by a complex scalar field $\phi(x) \in \mathbb{C}$:

$$
\phi(x) = r(x) e^{i\theta(x)}

$$

where:
1. **Modulus $r(x) \ge 0$:** Represents the **Metric Separation** between daughter queries $\{q_+, q_-\}$ in the Attentive Atlas (Definition {prf:ref}`def-query-fission`).
   - $r=0$: Coalescence (Single Chart / Vacuum)
   - $r>0$: Fission (Distinct Concepts)

2. **Phase $\theta(x)$:** Represents the **Orientation** of the split in the latent fiber (the specific feature axis along which differentiation occurs).

The field $\phi$ transforms as a doublet under the gauge group $SU(2)_L$, coupling it to the inference spinor.

({prf:ref}`def-decision-coupling`) *definition* — **The Decision Coupling**

Let $\Psi_L = (\psi_{\text{pred}}, \psi_{\text{obs}})^T$ be the belief doublet and $\Psi_R = \psi_{\text{act}}$ be the action singlet. The transfer of information from Belief to Action is mediated by the **Ontological Order Parameter** $\phi$.

The simplest $G_{\text{Fragile}}$-invariant coupling is:

$$
\mathcal{L}_{\text{Yukawa}} = -Y_{ij} \left( \bar{\Psi}_{L,i} \cdot \phi \cdot \Psi_{R,j} + \bar{\Psi}_{R,j} \cdot \phi^\dagger \cdot \Psi_{L,i} \right)

$$

where $Y_{ij}$ is the **Affordance Matrix** (a learned weight matrix determining which concepts trigger which actions).

*Cross-reference:* This implements the TopologicalDecoder ({ref}`Section 7.10 <sec-decoder-architecture-overview-topological-decoder>`) which maps belief geometry to motor output.

({prf:ref}`def-value-4-potential`) *definition* — **The Value 4-Potential**

We lift the effective potential $\Phi_{\text{eff}}(z)$ (Definition {prf:ref}`def-effective-potential`) to an external 4-potential:

$$
A^{\text{ext}}_\mu(z) = (-\Phi_{\text{eff}}(z), \vec{0})

$$

This is an **external background field**, distinct from the internal gauge field $B_\mu$.

({prf:ref}`def-cognitive-lagrangian`) *definition* — **The Standard Model of Cognition**

$$
\boxed{
\begin{aligned}
\mathcal{L}_{\text{SM}} = \quad & \underbrace{-\frac{1}{4} B_{\mu\nu}B^{\mu\nu} -\frac{1}{4} W^a_{\mu\nu}W^{a\mu\nu} -\frac{1}{4} G^a_{\mu\nu}G^{a\mu\nu}}_{\text{I. Gauge Sector: Strategic Curvature}} \\
& + \underbrace{\bar{\Psi}_L i \gamma^\mu D_\mu \Psi_L + \bar{\Psi}_R i \gamma^\mu D_\mu \Psi_R}_{\text{II. Inference Sector: Belief Dynamics}} \\
& + \underbrace{|D_\mu \phi|^2 - \left(-\mu^2 |\phi|^2 + \lambda |\phi|^4\right)}_{\text{III. Scalar Sector: Ontological Stability}} \\
& - \underbrace{Y_{ij} (\bar{\Psi}_L \phi \Psi_R + \text{h.c.})}_{\text{IV. Yukawa Sector: Decision Weight}} \\
& - \underbrace{\bar{\Psi} \gamma^\mu A^{\text{ext}}_\mu \Psi}_{\text{V. External Sector: Value Drive}}
\end{aligned}
}

$$

({prf:ref}`def-agent-parameter-vector`) *definition* — **The Agent Parameter Vector**

Let the **Agent Parameter Vector** $\Lambda$ be the tuple of fundamental operational constants:

$$
\Lambda = (c_{\text{info}}, \sigma, \ell_L, T_c, g_s, \gamma)

$$

where:
1. **$c_{\text{info}}$:** Information propagation speed (Axiom {prf:ref}`ax-information-speed-limit`)
2. **$\sigma$:** Cognitive Action Scale (Definition {prf:ref}`def-cognitive-action-scale`)
3. **$\ell_L$:** Levin Length, the minimal distinguishable scale (Definition {prf:ref}`def-levin-length`)
4. **$T_c$:** Cognitive Temperature. The critical value is $T_c^* = \mu^2/4$ where $\mu = 1/2 + u_\pi^r$ is the bifurcation parameter (Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`). For small policy control ($u_\pi^r \ll 1$), $T_c^* \approx 1/16$.
5. **$g_s$:** Binding coupling strength (Theorem {prf:ref}`thm-emergence-binding-field`)
6. **$\gamma$:** Temporal discount factor, $\gamma \in (0,1)$

**Dimensional Analysis:**

| Parameter | Symbol | Dimension | SI Units |
|:----------|:-------|:----------|:---------|
| Information speed | $c_{\text{info}}$ | $[L \, T^{-1}]$ | m/s |
| Cognitive action scale | $\sigma$ | $[E \, T]$ | J·s |
| Levin length | $\ell_L$ | $[L]$ | m |
| Cognitive temperature | $T_c$ | $[E]$ | J (with $k_B = 1$) |
| Binding coupling | $g_s$ | $[1]$ | dimensionless |
| Discount factor | $\gamma$ | $[1]$ | dimensionless |

**Derived Quantities:**

Define the **Causal Horizon Length** $\ell_0 = c_{\text{info}} \cdot \tau_{\text{proc}}$ with dimension $[L]$. Let the temporal discount rate be $\lambda := -\ln\gamma / \Delta t$ and identify the processing interval $\Delta t := \tau_{\text{proc}}$. The **Spatial Screening Mass** is then:

$$
\kappa = \frac{\lambda}{c_{\text{info}}} = \frac{-\ln\gamma}{\ell_0}

$$

with dimension $[L^{-1}]$ (Corollary {prf:ref}`cor-discount-as-screening-length`).

These correspond to the physics constants $\{c, \hbar, \ell_P, k_B T, \alpha_s, \gamma_{\text{cosmo}}\}$ under the isomorphism of {ref}`Section 34.6 <sec-isomorphism-dictionary>`.

({prf:ref}`def-sieve-constraint-system`) *definition* — **The Sieve Constraint System**

Let $\mathcal{S}(\Lambda)$ denote the vector of constraint functions. The agent is **viable** if and only if:

$$
\mathcal{S}(\Lambda) \le \mathbf{0}

$$

where the inequality holds component-wise. Each component corresponds to a Sieve node that enforces a specific consistency condition. A constraint violation ($\mathcal{S}_i > 0$) triggers a diagnostic halt at the corresponding node.

({prf:ref}`def-planck-levin-correspondence`) *definition* — **The Planck-Levin Correspondence**

Under the physics isomorphism ({ref}`Section 34.6 <sec-isomorphism-dictionary>`), the Levin Length $\ell_L$ corresponds to the Planck Length $\ell_P$:

$$
\ell_L \leftrightarrow \ell_P = \sqrt{\frac{\hbar G}{c^3}}

$$

The holographic bound becomes the Bekenstein-Hawking entropy bound:

$$
S_{\text{BH}} = \frac{A}{4\ell_P^2}

$$

*Remark:* The coefficient $\nu_2 = 1/4$ is derived in Theorem {prf:ref}`thm-a-complete-derivation-area-law` from first principles, recovering the Bekenstein-Hawking result without invoking black hole physics.

({prf:ref}`def-metabolic-parameters`) *definition* — **Metabolic Parameters**

The agent possesses:
1. **$\dot{E}_{\text{met}}$:** Metabolic power budget (energy flux available for computation)
2. **$\dot{I}_{\text{erase}}$:** Information erasure rate (bits forgotten per unit time)
3. **$T_c$:** Cognitive Temperature (entropy-exploration tradeoff)

({prf:ref}`def-coupling-function`) *definition* — **The Coupling Function**

Let the binding coupling $g_s(\mu)$ (dimensionless) be a function of the **resolution scale** $\mu$, which has dimension $[L^{-1}]$ (inverse length). Equivalently, $\mu$ can be expressed as an energy scale via $\mu \sim E/(\sigma)$ where $\sigma$ is the Cognitive Action Scale.

The limits are:
- $\mu \to 0$: Macro-scale (coarse representation, low in TopoEncoder hierarchy)
- $\mu \to \infty$: Micro-scale (texture level, high in TopoEncoder hierarchy)

The coupling evolves according to the **Beta Function**:

$$
\mu \frac{dg_s}{d\mu} = \beta(g_s)

$$

where both sides are dimensionless (since $g_s$ is dimensionless and $\mu \, dg_s/d\mu$ has $[\mu] \cdot [\mu^{-1}] = [1]$).

For $SU(N_f)$ gauge theories, $\beta(g_s) < 0$ for $N_f \ge 2$ (asymptotic freedom).

({prf:ref}`def-stiffness-parameter`) *definition* — **The Stiffness Parameter**

Let $\Delta E$ denote the characteristic energy gap between metastable states in the agent's latent manifold. Define the **Stiffness Ratio**:

$$
\chi = \frac{\Delta E}{T_c}

$$

This ratio determines the tradeoff between memory persistence and adaptability.

({prf:ref}`def-constraint-matrix`) *definition* — **The Constraint Matrix**

Let $\Lambda = (c_{\text{info}}, \sigma, \ell_L, T_c, g_s, \gamma)$ be the parameter vector. The Sieve constraints form the system:

$$
\mathbf{A} \cdot \Lambda \le \mathbf{b}

$$

where:

| Constraint | Inequality | Node |
|:-----------|:-----------|:-----|
| Causal Lower | $d_{\text{sync}}/\tau_{\text{proc}} \le c_{\text{info}}$ | 2 |
| Causal Upper | $c_{\text{info}} \le L_{\text{buf}}/\tau_{\text{proc}}$ | 62 |
| Holographic | $\ell_L^{D-1} \le \nu_D \text{Area}_\partial / I_{\text{req}}$ | 56 |
| Landauer | $T_c \le \dot{E}_{\text{met}} / (\dot{I}_{\text{erase}} \ln 2)$ | 52 |
| IR Binding | $g_s(\mu_{\text{IR}}) \ge g_s^{\text{crit}}$ | 40 |
| UV Decoupling | $g_s(\mu_{\text{UV}}) \le \epsilon$ (for $\epsilon \to 0$) | 29 |
| Stiffness Lower | $\Delta E > T_c$ | 7 |
| Stiffness Upper | $\Delta E < \chi_{\text{max}} T_c$ | 7 |
| Discount Lower | $\gamma > \gamma_{\text{min}}$ | --- |
| Discount Upper | $\gamma < 1$ | --- |

({prf:ref}`def-dual-objective`) *definition* — **The Dual Objective**

The agent's objective trades representational power against computational cost:

$$
\mathcal{J}(\Lambda) = \underbrace{I_{\text{bulk}}(\Lambda)}_{\text{World Model Capacity}} - \beta \cdot \underbrace{\mathcal{V}_{\text{metabolic}}(\Lambda)}_{\text{Thermodynamic Cost}}

$$

where:
- $I_{\text{bulk}}$: Bulk information capacity (increases with resolution)
- $\mathcal{V}_{\text{metabolic}}$: Metabolic cost of computation
- $\beta > 0$: Cost sensitivity parameter

({prf:ref}`def-waste-quotient`) *definition* — **The Waste Quotient**

For a consensus protocol $\mathcal{P}$, the **Waste Quotient** is:

$$
W_\mathcal{P} := 1 - \frac{\Delta I_{\text{world}}}{\int \dot{\mathcal{M}}(t) \, dt}

$$

where:
- $\Delta I_{\text{world}}$ is the mutual information gained about the world through the computation
- $\dot{\mathcal{M}}(t)$ is the metabolic flux (Definition {prf:ref}`def-metabolic-flux`)

*Units:* $[W_\mathcal{P}] = \text{dimensionless}$.

*Examples:*
- **Bitcoin:** $W_{\text{BTC}} \approx 1$. SHA-256 hashes produce zero structural information about the world: $I(X_{\text{world}}; \text{Hash}) = 0$.
- **Target:** $W_{\text{PoUW}} \to 0$. Energy dissipation equals the reduction in model uncertainty.

({prf:ref}`def-model-state`) *definition* — **The Global Model State**

The **Global Model State** at block height $h$ is a parameter vector:

$$
\Theta_h \in T_{\bar{z}} \mathcal{Z} \cong \mathbb{R}^D

$$

where:
- $D$ is the model dimension
- $T_{\bar{z}} \mathcal{Z}$ is the tangent space at the current mean belief $\bar{z}$
- The metric on parameter space inherits from the Capacity-Constrained Metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`)

*Units:* $[\Theta] = [z]$ (latent coordinates).

*Interpretation:* $\Theta_h$ represents the collective belief state of the network—the shared world model encoded in the blockchain.

({prf:ref}`def-curriculum-block`) *definition* — **The Curriculum Block**

A **Curriculum Block** $B_h$ at height $h$ is a tuple:

$$
B_h := (\mathcal{H}_{\text{prev}}, \mathcal{H}_D, g_h, \pi_{\text{stake}}, \zeta_h)

$$

where:
- $\mathcal{H}_{\text{prev}} \in \{0,1\}^{256}$ is the hash of the previous block
- $\mathcal{H}_D \in \{0,1\}^{256}$ is the content identifier of training data $D_h$ (e.g., IPFS CID)
- $g_h \in \mathbb{R}^D$ is the **gradient update** computed on $D_h$
- $\pi_{\text{stake}} \in \{0,1\}^{512}$ is the staking proof (signature over stake tokens)
- $\zeta_h \in \mathbb{R}^{d_\zeta}$ is the **Sieve certificate** (validation metadata)

*Units:* $[g_h] = \text{nat}/[z]$ (gradient in latent coordinates).

({prf:ref}`def-chain-evolution`) *definition* — **The Chain Evolution Rule**

The global model evolves by **Stochastic Gradient Descent**:

$$
\Theta_{h+1} = \Theta_h - \eta_h \cdot g_h

$$

where $\eta_h > 0$ is the learning rate at height $h$, determined by the difficulty adjustment algorithm (Definition {prf:ref}`def-difficulty-adjustment`).

*Interpretation:* Each block advances the collective belief toward lower loss on the public curriculum. The blockchain is a **thermodynamic record** of this learning process.

({prf:ref}`def-gradient-mining-puzzle`) *definition* — **The Gradient Mining Puzzle**

A miner solving block $h$ must:

1. **Fetch Data:** Retrieve training batch $D_h$ from the curriculum queue
2. **Compute Gradient:** Calculate $g = \nabla_\Theta \mathcal{L}(\Theta_{h-1}, D_h)$
3. **Satisfy Sieve Constraints:**
   - **CostBoundCheck (Node 1):** $\|g\|_G \leq E_{\max}$ (bounded energy)
   - **TextureFirewallCheck (Node 29):** $\|\partial_{z_{\text{tex}}} g\| < \epsilon_{\text{tex}}$ (no texture leakage)
   - **CausalEnclosureCheck (Node 53):** $\Delta_{\text{causal}}(g) < \delta_{\text{causal}}$ (causal consistency)
4. **Submit Block:** Broadcast $(B_h, \Theta_h)$ to the network

*Difficulty Adjustment:* See Definition {prf:ref}`def-difficulty-adjustment`.

({prf:ref}`def-difficulty-adjustment`) *definition* — **The Difficulty Adjustment Algorithm**

The network **Difficulty** $\mathcal{D}_h$ at height $h$ controls the minimum batch size $|D_h|$ required for valid blocks:

$$
\mathcal{D}_{h+1} = \mathcal{D}_h \cdot \exp\left( \alpha_{\text{diff}} \left( \frac{t_h - t_{\text{target}}}{t_{\text{target}}} \right) \right)

$$

where:
- $t_h$ is the actual time to mine block $h$
- $t_{\text{target}}$ is the target block time (e.g., 10 minutes)
- $\alpha_{\text{diff}} > 0$ is the adjustment rate

*Units:* $[\mathcal{D}] = \text{samples}$.

*Constraint:* A valid block must satisfy $|D_h| \geq \mathcal{D}_h$.

({prf:ref}`def-boundary-flux-certificate`) *definition* — **The Boundary Flux Certificate**

The **Boundary Flux Certificate** $\zeta_h$ included in block $B_h$ contains:

$$
\zeta_h := \left( \|g_h\|_G, \, \nabla_{\partial} g_h, \, \text{Tr}(H_h), \, \sigma_{\text{sample}} \right)

$$

where:
- $\|g_h\|_G$ is the gradient norm in the capacity-constrained metric
- $\nabla_{\partial} g_h$ is the boundary gradient (projection onto interface coordinates)
- $\text{Tr}(H_h)$ is the trace of the Hessian (curvature summary)
- $\sigma_{\text{sample}}$ is a random seed for spot-check sampling

*Units:* $[\zeta] = \text{mixed}$ (norm: $\text{nat}/[z]$; trace: $\text{nat}/[z]^2$).

({prf:ref}`def-optimistic-verification`) *definition* — **The Optimistic Verification Protocol**

The network verifies blocks using **Optimistic Acceptance with Challenge Period**:

1. **Submission:** Miner submits block $B_h$ with stake $S_h$
2. **Optimistic Acceptance:** Block is provisionally accepted
3. **Challenge Window:** For duration $T_{\text{challenge}}$, any node may challenge
4. **Challenge:** Challenger computes gradient on random subset $d \subset D_h$ with $|d| = \lceil 0.01 |D_h| \rceil$
5. **Adjudication:** If $\cos(g_h, g_{\text{challenger}}) < \theta_{\text{min}}$, miner is **slashed** (stake burned)
6. **Finalization:** After $T_{\text{challenge}}$ with no successful challenge, block is finalized

({prf:ref}`def-mining-game`) *definition* — **The Mining Game**

The **Mining Game** $\Gamma$ is defined by:

- **Players:** $N$ miners indexed by $i \in \{1, \ldots, N\}$
- **Strategy Space:** Each miner chooses $\sigma_i \in \{\text{Honest}, \text{Cheat}\}$
  - **Honest:** Compute true gradient $g_i = \nabla_\Theta \mathcal{L}(\Theta, D)$
  - **Cheat:** Submit fake gradient $g_i' \neq g_{\text{true}}$
- **Payoffs:**
  - Block reward: $R > 0$ (received if block accepted)
  - Stake: $S > 0$ (lost if successfully challenged)
  - Computation cost: $C_{\text{honest}} > C_{\text{cheat}}$

**Key Assumptions:**
1. The detection probability $p_{\text{detect}}$ is exogenous (determined by the spot-check protocol, independent of other miners' strategies)
2. Block rewards are per-miner (not split among winners)
3. Miners play pure strategies (mixed strategies analyzed in Corollary {prf:ref}`cor-stake-reward-ratio`)

({prf:ref}`def-network-metric-tensor`) *definition* — **The Network Metric Tensor**

Each validator $i$ maintains a local metric tensor $G^{(i)}$ on the shared latent manifold. The **Network Metric Friction** between chains $\mathcal{C}_A$ and $\mathcal{C}_B$ is:

$$
\mathcal{F}(\mathcal{C}_A, \mathcal{C}_B) := \sum_{i,j} \mathcal{F}_{ij}(\Theta_{\text{head}}^A, \Theta_{\text{head}}^B)

$$

where $\mathcal{F}_{ij}$ is the pairwise metric friction (Definition {prf:ref}`def-metric-friction`).

({prf:ref}`def-metric-friction-consensus`) *definition* — **Metric Friction Consensus**

The **Canonical Chain** is selected by minimizing global metric friction:

$$
\mathcal{C}^* = \arg\min_{\mathcal{C}} \sum_{i < j} \mathcal{F}_{ij}(\Theta_{\text{head}}^\mathcal{C})

$$

*Mechanism:*
1. Miners propose competing updates $\{g_A, g_B, \ldots\}$
2. Validators compute local metric tensors $G^{(i)}(\Theta + g_k)$ for each candidate
3. The update minimizing pairwise friction is accepted
4. Ties broken by timestamp (first-seen)

({prf:ref}`def-token-standard`) *definition* — **The Token Standard**

The $\text{COG}$ token has three fundamental operations:

1. **Minting (Supply).** Tokens are minted when **Ontological Stress** $\Xi$ is reduced:

$$
\Delta \text{Supply} = \kappa_{\text{mint}} \cdot \max(0, -\Delta \Xi_{\text{global}})

$$

where $\kappa_{\text{mint}}$ is the minting coefficient (tokens per nat of stress reduction).

*Interpretation:* Value is created only when the network learns something new.

2. **Burning (Demand).** Tokens are burned to request **Inference**:

$$
\text{Cost}(Q) = \mathfrak{T}_{\text{harvest}}^{-1}(\dot{\mathcal{M}}_Q)

$$

where $\dot{\mathcal{M}}_Q$ is the metabolic cost of answering query $Q$.

3. **Transfer.** Standard ERC-20-like transfers between accounts.

*Units:* $[\text{COG}] = \text{Joules}$ (energy equivalent).

*Value Anchor:* $1 \, \text{COG} \approx 1 \, \text{Joule}$ of useful gradient computation at reference temperature $T_c$.

({prf:ref}`def-chain-renormalization`) *definition* — **Chain Renormalization (Pruning)**

Old blocks are **coarse-grained** into **Epoch Blocks** via the Projection Operator:

$$
B_{\text{epoch}} = \Pi\left( \sum_{h \in \text{epoch}} B_h \right)

$$

where $\Pi$ projects onto the low-frequency components of the gradient history.

*Mechanism:*
1. Every $N_{\text{epoch}}$ blocks, compress the epoch into a summary
2. Discard individual block data (retain Merkle proofs)
3. The agent remembers the "gist" but forgets the "noise"

*Thermodynamics:* This is **information erasure** (Landauer cost). It releases storage but maintains the essential learning trajectory.

({prf:ref}`def-a-boundary-capacity-form`) *definition* — **A.1.1 (Boundary capacity form)**

Define the boundary capacity $(n\!-\!1)$-form

$$
\omega_{\partial} := \frac{1}{\eta_\ell}\, dA_G,

$$
so that $C_{\partial}(\partial\mathcal{Z})=\oint_{\partial\mathcal{Z}}\omega_{\partial}$ (Definition 17.1.3).

({prf:ref}`def-a-boundary-capacity-constraint-functional`) *definition* — **A.1.2 (Boundary-capacity constraint functional)**

Define the saturation functional

$$
\mathcal{C}[G,V]
:=
\underbrace{\int_{\mathcal{Z}} \rho_I(G,V)\, d\mu_G}_{I_{\text{bulk}}}
\;-\;
\underbrace{\oint_{\partial\mathcal{Z}}\omega_{\partial}}_{C_{\partial}},

$$
where $\rho_I(G,V)$ is an *information density* (nats per unit $d\mu_G$) compatible with the agent's representation scheme (Definition 17.1.2). This $\rho_I$ is distinct from the belief density $p$ used in {ref}`Section 2.11 <sec-variance-value-duality-and-information-conservation>`. When $\rho_I$ is instantiated via the split shutter, the most conservative computable proxy is a global one, $I_{\text{bulk}}\approx \mathbb{E}[I(X;K)]$ (Node 13), and the window theorem (Theorem {prf:ref}`thm-information-stability-window-operational`) supplies the admissible operating range.

({prf:ref}`def-a-risk-lagrangian-density`) *definition* — **A.1.3 (Risk Lagrangian density)**

Fix a smooth potential $V\in C^\infty(\mathcal{Z})$. A canonical risk Lagrangian density is the scalar-field functional

$$
\mathcal{L}_{\text{risk}}(V;G) := \frac{1}{2}\,G^{ab}\nabla_a V\,\nabla_b V + U(V),

$$
where $U:\mathbb{R}\to\mathbb{R}$ is a (possibly learned) on-site potential capturing non-gradient costs. (The sign convention is chosen for a Riemannian metric; see e.g. Lee, *Riemannian Manifolds*, 2018, for the variational identities used below.)

({prf:ref}`def-a-capacity-constrained-curvature-functional`) *definition* — **A.1.4 (Capacity-constrained curvature functional)**

Let $R(G)$ be the scalar curvature of $G$ and let $\Lambda\in\mathbb{R}$ be a constant. Define the constrained functional

$$
\mathcal{S}[G,V]
:=
\int_{\mathcal{Z}}\left(R(G)-2\Lambda + 2\kappa\,\mathcal{L}_{\text{risk}}(V;G)\right)d\mu_G
\;-\;
2\kappa\oint_{\partial\mathcal{Z}}\omega_{\partial},

$$
with coupling $\kappa\in\mathbb{R}$. The last term is the explicit boundary capacity penalty, and $\Lambda$ is a bulk capacity offset that remains once the boundary is clamped at finite resolution.

*Remark (why $\Lambda$ is allowed).* A constant term in the integrand is the simplest coordinate-invariant scalar density and produces a $\Lambda G_{ij}$ term in the metric Euler–Lagrange equation. Here $\Lambda$ plays the role of a baseline curvature / capacity offset.

({prf:ref}`def-a-computational-microstate`) *definition* — **A.6.0c (Computational Microstate)**

A **computational microstate** at resolution $\ell$ is a complete specification of the agent's internal configuration $\mu = (\rho, K, \theta)$ where:
- $\rho \in \mathcal{P}(\mathcal{Z})$ is the belief distribution over the latent manifold
- $K \in \{1, \ldots, |\mathcal{K}|\}$ is the active chart assignment
- $\theta$ are the model parameters

discretized at the Levin Length scale: positions resolved to precision $\ell_L$, probabilities resolved to precision $e^{-1}$ in KL divergence.

Two microstates $\mu_1, \mu_2$ are **boundary-distinguishable** if an external observer, receiving only boundary observations $\partial\mathcal{Z}$, can distinguish them with probability $> 1 - e^{-1}$.

*Remark (Analogy to Physics).* In black hole thermodynamics, a microstate is a specific quantum configuration of the horizon degrees of freedom. Here, a microstate is a specific configuration of the agent's belief state. The boundary plays the role of the horizon: internal distinctions not visible at the boundary do not count toward the entropy.

({prf:ref}`def-a-information-horizon`) *definition* — **A.6.3 (Information Horizon)**

The **information horizon** $r_h$ is the smallest positive root of:

$$
1 - \frac{2\mu(r_h)}{(n-2)r_h^{n-2}} - \frac{\Lambda_{\text{eff}} r_h^2}{n(n-1)} = 0.

$$
At this radius, $A(r_h) \to \infty$ and $G^{rr}(r_h) \to 0$.

({prf:ref}`def-e7-strategic-metric`) *definition* — **E.7.1 (The Strategic Metric)**

Let $G^{(i)}$ be the capacity-constrained metric on $\mathcal{Z}^{(i)}$ (Theorem {prf:ref}`thm-capacity-constrained-metric-law`). The **Strategic Metric** $\mathbf{g}$ on $\mathcal{M}$ is the block-diagonal sum perturbed by the Game Tensor $\mathcal{G}$ (Definition {prf:ref}`def-the-game-tensor`):

$$
\mathbf{g}(\mathbf{z}) := \bigoplus_{i=1}^N G^{(i)}(z^{(i)}) + \alpha \sum_{i \neq j} \mathcal{G}_{ij}(\mathbf{z}),

$$
where the pullback of the cross-Hessian interaction acts on tangent vectors in the obvious way.

*Assumption 1 (Ellipticity):* We assume $\alpha > 0$ is sufficiently small such that $\mathbf{g}$ remains positive-definite and defines a valid Riemannian structure on $\mathcal{M}$. This is guaranteed when $\|\alpha \mathcal{G}\|_{\text{op}} < \lambda_{\min}(\bigoplus G^{(i)})$.

({prf:ref}`def-e7-strategic-hamiltonian`) *definition* — **E.7.2 (The Strategic Hamiltonian)**

The self-adjoint **Strategic Hamiltonian** operator $\hat{H}_\sigma: H^2(\mathcal{M}) \to L^2(\mathcal{M}, d\mu_{\mathbf{g}})$ acts on the joint wave-function $\Psi$:

$$
\hat{H}_\sigma := -\frac{\sigma^2}{2} \Delta_{\mathbf{g}} + \mathcal{V}(\mathbf{z}),

$$
where:
- $\Delta_{\mathbf{g}}$ is the Laplace-Beltrami operator associated with the strategic metric $\mathbf{g}$
- $\mathcal{V}(\mathbf{z}) := \sum_{i=1}^N \Phi^{(i)}_{\text{eff}}(z^{(i)}) + \sum_{i < j} \Phi_{ij}(z^{(i)}, z^{(j)})$ is the joint potential
- $\sigma > 0$ is the cognitive action scale (Definition {prf:ref}`def-cognitive-action-scale`)

*Assumption 2 (Regularity):* $\mathcal{V} \in C^2(\mathcal{M})$ and is bounded below.

({prf:ref}`def-e7-forbidden-region`) *definition* — **E.7.3 (The Forbidden Region and Nash Basins)**

Let $E_0 := \inf \text{spec}(\hat{H}_\sigma)$ be the ground state energy. The **Classically Forbidden Region** (Barrier) is:

$$
\mathcal{K} := \{ \mathbf{z} \in \mathcal{M} : \mathcal{V}(\mathbf{z}) > E_0 \}.

$$
Let $\Omega_A, \Omega_B \subset \mathcal{M} \setminus \mathcal{K}$ be disjoint open sets (Nash basins) where $\mathcal{V}(\mathbf{z}) \leq E_0$.

*Geometric interpretation:* $\Omega_A$ and $\Omega_B$ are "potential wells" corresponding to distinct Nash equilibria (Theorem {prf:ref}`thm-nash-ground-state`). The barrier $\mathcal{K}$ separates these wells.

({prf:ref}`def-e7-agmon-metric`) *definition* — **E.7.4 (The Agmon Metric)**

Inside the barrier $\mathcal{K}$, we define the **Agmon Metric** $\rho_E$, a degenerate conformal rescaling of $\mathbf{g}$:

$$
(\rho_E)_{ij}(\mathbf{z}) := \max\left(0, \mathcal{V}(\mathbf{z}) - E_0\right) \cdot \mathbf{g}_{ij}(\mathbf{z}).

$$
The **Agmon distance** between points $\mathbf{x}, \mathbf{y} \in \mathcal{M}$ is:

$$
d_{\text{Ag}}(\mathbf{x}, \mathbf{y}) := \inf_{\gamma: \mathbf{x} \to \mathbf{y}} \int_0^1 \sqrt{\max(0, \mathcal{V}(\gamma(t)) - E_0)} \cdot \|\dot{\gamma}(t)\|_{\mathbf{g}} \, dt,

$$
where the infimum is over all piecewise smooth paths $\gamma$ from $\mathbf{x}$ to $\mathbf{y}$.

*Properties:*
1. $d_{\text{Ag}}(\mathbf{x}, \mathbf{y}) = 0$ if there exists a path entirely within $\mathcal{M} \setminus \mathcal{K}$ (the "classical" region)
2. $d_{\text{Ag}}(\mathbf{x}, \mathbf{y}) > 0$ if all paths must traverse $\mathcal{K}$ (tunneling required)
3. The Agmon distance is a pseudo-metric (satisfies triangle inequality)

## Axioms

({prf:ref}`ax-bulk-boundary-decoupling`) *axiom* — **Bulk-Boundary Decoupling**

The state decomposition $Z = (K, z_n, z_{\text{tex}})$ satisfies a **partition condition**:

1. **Interior (Planning Domain):** The trajectory $z(\tau)$ evolves strictly on the manifold $\mathcal{Z} = \mathcal{K} \times \mathcal{Z}_n$. It contains no texture component. Planning depends only on geometry and topology:

$$
\dot{z} = f(z, u_\pi) \quad (\text{No } z_{\text{tex}} \text{ dependence})

$$
2. **Boundary Interface:** Texture $z_{\text{tex}}$ is a stochastic component that exists **only** at the interface where the internal state meets the external observation:

$$
z_{\text{tex}} \sim \mathcal{N}(0, \Sigma(z_{\text{final}}))

$$
Formally, the partition condition is:

$$
\frac{\partial}{\partial z_{\text{tex}}} \left[ \dot{z}^k, \lambda_{\text{jump}}, u_\pi \right] = 0 \quad \forall \tau \in [0, \tau_{\text{stop}})

$$

({prf:ref}`ax-motor-texture-firewall`) *axiom* — **Motor Texture Firewall**

Motor texture is decoupled from the Bulk dynamics:

$$
\partial_{z_{\text{tex,motor}}} \dot{z} = 0, \qquad \partial_{z_{\text{tex,motor}}} u_\pi = 0.

$$
The policy $\pi_\theta$ operates on $(K, z_n, A, z_{n,\text{motor}})$ but **never** on $(z_{\text{tex}}, z_{\text{tex,motor}})$.

*Remark (Sim-to-Real Gap).* The **motor texture variance** $\sigma_{\text{motor}}^2$ is the mathematical definition of the "Sim-to-Real gap":
- **Simulation:** $\sigma_{\text{motor}} \approx 0$ (deterministic, no tremor)
- **Reality:** $\sigma_{\text{motor}} > 0$ (friction, sensor noise, motor tremor)
- **Robustness:** The Bulk policy $u_\pi$ is invariant; only the Action Decoder learns to manage domain-specific noise.

**Cross-references:** {ref}`Section 21.3 <sec-the-retrieval-texture-firewall>` (Texture Firewall), Axiom {prf:ref}`ax-bulk-boundary-decoupling`.

({prf:ref}`ax-the-boltzmann-value-law`) *axiom* — **The Generalized Boltzmann-Value Law**

The scalar potential $\Phi(z)$ from the Hodge decomposition (Theorem {prf:ref}`thm-hodge-decomposition`) represents the **Gibbs Free Energy** of the state $z$:

$$
\Phi(z) = E(z) - T_c S(z),

$$
where:
- $E(z)$ is the **task risk/cost** at state $z$
- $S(z)$ is the **exploration entropy** (measure of uncertainty/optionality)
- $T_c$ is the **cognitive temperature** ({prf:ref}`def-cognitive-temperature`, {ref}`Section 21.1 <sec-hyperbolic-volume-and-entropic-drift>`)

*Units:* $[\Phi] = [E] = [T_c S] = \mathrm{nat}$.

**Conservative Case ($\mathcal{F} = 0$):** When the Value Curl vanishes, $\mathcal{R} = d\Phi$ and the scalar potential $\Phi$ is the complete value function $V(z)$.

**Non-Conservative Case ($\mathcal{F} \neq 0$):** The scalar potential $\Phi$ captures only the optimizable component of the reward field. The solenoidal component $\delta\Psi$ creates additional cyclic dynamics.

({prf:ref}`ax-metric-isometry`) *axiom* — **Metric Isometry**

There exists a canonical isometry $\Phi: \mathcal{Z}_{\text{int}} \to \mathcal{Z}_{\text{ext}}$ such that for all $z, z' \in \mathcal{Z}_{\text{int}}$:

$$
d_{G_{\text{int}}}(z, z') = d_{G_{\text{ext}}}(\Phi(z), \Phi(z')),

$$
where both manifolds carry the Poincare metric (Definition {prf:ref}`def-hyperbolic-volume-growth`):

$$
G_{ij}(z) = \frac{4\delta_{ij}}{(1 - \|z\|^2)^2}.

$$
*Interpretation:* The isometry axiom asserts that embedding models trained on shared semantic corpora induce compatible distance structures. This is the mathematical foundation for cross-modal retrieval.

({prf:ref}`ax-ontological-expansion-principle`) *axiom* — **Ontological Expansion Principle**

The agent should expand its chart structure (increase $N_c$) if and only if the expected value improvement exceeds the complexity cost:

$$
\mathbb{E}\left[\Delta V \mid \text{fission}\right] > \mathcal{C}_{\text{complexity}}(N_c \to N_c + 1),

$$
where $\Delta V$ is the value gain from finer discrimination and $\mathcal{C}_{\text{complexity}}$ is measured in nats (to match units with value).

*Remark.* This is the MDL/rate-distortion principle ({ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>`) applied to ontology: expand only if the distortion reduction exceeds the rate increase.

({prf:ref}`ax-ontological-simplification`) *axiom* — **Ontological Simplification Principle**

The agent shall reduce ontological complexity when the expected value of maintaining a distinction is negative:

$$
\mathcal{C}_{\text{saved}}(N_c \to N_c - 1) > G_\Delta(i, j) + \mathbb{E}[\Delta V \mid \text{no fusion}]

$$
where $\mathcal{C}_{\text{saved}}$ is the metabolic savings from eliminating a chart.

*Remark.* This is the dual of {prf:ref}`ax-ontological-expansion-principle` (Ontological Expansion Principle). Both derive from the same MDL objective: minimize description length plus expected regret.

({prf:ref}`ax-dual-horizon-action`) *axiom* — **Dual-Horizon Action**

For any interaction step $t$, the agent selects a total computation budget $S \in [0, S_{\max}]$ that minimizes the **Deliberation Action** $\mathcal{S}_{\text{delib}}$:

$$
\mathcal{S}_{\text{delib}}[S] = -\underbrace{\mathbb{E}_{z \sim \rho_S} [V(z)]}_{\text{Expected Terminal Value}} + \underbrace{\Psi_{\text{met}}(S)}_{\text{Computational Cost}},

$$
where $V(z)$ is the task potential ({ref}`Section 24.2 <sec-hodge-decomposition-of-value>`). Units: $[\mathcal{S}_{\text{delib}}] = \text{nat}$.

*Physical interpretation:* The agent faces a trade-off: longer deliberation ($S$ large) improves the expected value $\langle V \rangle_{\rho_S}$ by refining the belief toward high-value regions, but incurs greater metabolic cost $\Psi_{\text{met}}(S)$. The optimal $S^*$ balances these competing pressures.

*Remark (Sign convention).* We write $-\langle V \rangle$ because the agent seeks to **maximize** value. The Deliberation Action $\mathcal{S}_{\text{delib}}$ is minimized when value is maximized and cost is minimized.

({prf:ref}`ax-szilard-correspondence`) *axiom* — **The Szilard Correspondence (Information-Work Duality)**

Information about low-entropy configurations can be converted to extractable work. Specifically, if an agent possesses $I$ nats of mutual information with a thermal reservoir at temperature $T_{\text{env}}$, it can extract at most:

$$
W_{\max} = k_B T_{\text{env}} \cdot I

$$

joules of work, where $k_B$ is Boltzmann's constant.

*Physical basis:* This is the inverse of Landauer's principle. Landauer states that erasing 1 bit costs $k_B T \ln 2$ joules. Szilard's engine demonstrates that acquiring 1 bit about a system enables extracting $k_B T \ln 2$ joules. The two are thermodynamically dual.

*Cognitive interpretation:* A reward signal $r_t > 0$ encodes mutual information between the agent's state and resource availability. This information, when acted upon, enables work extraction from the environment.

({prf:ref}`ax-energy-conservation-battery`) *axiom* — **Energy Conservation (First Law)**

The battery evolves according to the First Law of Thermodynamics:

$$
\frac{dB}{dt} = \underbrace{\mathfrak{T}_{\text{harvest}}(r_t)}_{\text{Income}} - \underbrace{\dot{\mathcal{M}}(t)}_{\text{Metabolic Cost}} - \underbrace{\gamma_{\text{leak}} B(t)}_{\text{Passive Dissipation}}

$$

where:
- $\mathfrak{T}_{\text{harvest}}(r_t)$ is the transduced energy from rewards (Definition {prf:ref}`def-metabolic-transducer`)
- $\dot{\mathcal{M}}(t)$ is the metabolic cost from Theorem {prf:ref}`thm-generalized-landauer-bound`
- $\gamma_{\text{leak}} \geq 0$ is the passive self-discharge rate (basal metabolic rate)

*Terminal Condition:* If $B(t) \leq 0$, the agent undergoes **Thermodynamic Death**. The metric collapses (Theorem {prf:ref}`thm-fading-metric-law`), inference halts, and the agent can no longer perform coherent computation.

({prf:ref}`ax-finite-communication-bandwidth`) *axiom* — **Finite Communication Bandwidth**

The communication channel $\mathcal{L}$ between agents has finite Shannon capacity $C_{\mathcal{L}}$. By the Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`):

$$
C_{\mathcal{L}} \leq \nu_D \cdot \frac{\text{Area}(\partial\mathcal{L})}{\ell_L^{D-1}}

$$

*Justification:* Communication occurs through the agent's boundary interface. The Area Law limits the information rate of any boundary channel.

({prf:ref}`ax-information-speed-limit`) *axiom* — **Information Speed Limit**

There exists a maximum speed $c_{\text{info}} > 0$ at which information propagates through the environment $\mathcal{E}$. The **Causal Delay** between agents $i$ and $j$ is:

$$
\tau_{ij} := \frac{d_{\mathcal{E}}^{ij}}{c_{\text{info}}},

$$
where $d_{\mathcal{E}}^{ij}$ is the environment distance (Definition {prf:ref}`def-environment-distance`).

*Units:* $[c_{\text{info}}] = [\text{length}]/[\text{time}]$, $[\tau_{ij}] = [\text{time}]$.

*Examples:*
- **Physical systems:** $c_{\text{info}} = c \approx 3 \times 10^8$ m/s (speed of light)
- **Acoustic systems:** $c_{\text{info}} \approx 343$ m/s (speed of sound)
- **Networked systems:** $c_{\text{info}} \approx d/\text{latency}$ (effective propagation speed)
- **Co-located agents:** $c_{\text{info}} \to \infty$ effective limit when $d_{\mathcal{E}}^{ij} = 0$

({prf:ref}`ax-local-gauge-invariance`) *axiom* — **Local Gauge Invariance (Nuisance Invariance)**

The physical dynamics of the multi-agent system are invariant under position-dependent rotations of the internal nuisance coordinates. Formally, let $G$ be a compact Lie group with Lie algebra $\mathfrak{g}$. For any smooth map $U: \mathcal{Z} \to G$, the transformation

$$
\psi'(z, t) = U(z)\psi(z, t)

$$

leaves observable quantities (reward, policy output, Nash conditions) unchanged.

*Units:* $[U] = \text{dimensionless}$ (group element).

*Interpretation:* Agent $i$ at location $z$ is free to rotate its internal representation (the "basis" in which it encodes nuisance). This is not a symmetry to be broken but a **redundancy** in the description that must be properly handled via gauge theory.

({prf:ref}`ax-local-utility-invariance`) *axiom* — **Local Utility Invariance**

In a distributed agent with finite information speed $c_{\text{info}}$ (Axiom {prf:ref}`ax-information-speed-limit`), there is no global clock to synchronize the Value baseline across the manifold simultaneously. The agent must possess **Local Gauge Invariance**:

$$
\psi(x) \to e^{i\theta(x)} \psi(x),

$$

where $x$ denotes the spacetime coordinate on the agent's computational manifold. The choice of "zero utility" can vary locally across different charts without affecting the physical transfer of control authority.

*Justification:* This follows from the Causal Interval (Definition {prf:ref}`def-causal-interval`): spacelike-separated modules cannot instantaneously agree on a common baseline.

({prf:ref}`ax-cybernetic-parity-violation`) *axiom* — **Cybernetic Parity Violation**

The agent's interaction with the environment is **Chiral**, as established by the boundary condition asymmetry in {ref}`Section 23 <sec-the-boundary-interface-symplectic-structure>`:

1. **Sensors (Dirichlet Boundary, Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`):** The internal state $\psi$ is *updated* by boundary data. The boundary clamps the field value: $\phi|_{\partial\mathcal{Z}} = \phi_D$.

2. **Motors (Neumann Boundary, Definition {prf:ref}`def-neumann-boundary-condition-motors`):** The internal state *drives* the boundary flux. The boundary clamps the normal derivative: $\nabla_n \phi|_{\partial\mathcal{Z}} = j_N$.

The belief dynamics are not invariant under the exchange of Input and Output. The agent processes information (Left-Handed) differently than it emits control (Right-Handed).

({prf:ref}`ax-feature-confinement`) *axiom* — **Feature Confinement**

The agent observes and manipulates **Concepts** (Macro-symbols $K$), not raw **Features** (Nuisance coordinates $z_n$). From Definition {prf:ref}`def-bounded-rationality-controller`:

1. **Composite Structure:** A Concept $K$ is a bound state of sub-symbolic features processed through the Stacked TopoEncoder (Definition {prf:ref}`def-the-peeling-step`).

2. **Observability Constraint:** Free features are never observed in isolation at the boundary $\partial\mathcal{Z}$ (Definition {prf:ref}`def-boundary-markov-blanket`). Only "color-neutral" (bound) states can propagate to the macro-register.

*Cross-reference:* This is the representational analog of quark confinement in QCD.

({prf:ref}`ax-cognitive-dirac-equation`) *axiom* — **The Cognitive Dirac Equation**

The dynamics of the belief state follow the Dirac equation on the curved latent manifold:

$$
(i \gamma^\mu D_\mu - m) \Psi = 0

$$

*Justification:* The WFR equation ({ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`) is a second-order diffusion (Fokker-Planck). In the relativistic limit with finite information speed $c_{\text{info}}$ (Axiom {prf:ref}`ax-information-speed-limit`), this factorizes into two first-order wave equations coupled by mass. The Dirac equation is the unique first-order differential equation invariant under Lorentz transformations (causal structure) and the internal gauge group $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$.

- $\gamma^\mu$: The **Cognitive Gamma Matrices**, satisfying $\{\gamma^\mu, \gamma^\nu\} = 2g^{\mu\nu}$. They encode the local causal structure of the latent space.
- $m$: The **Inference Mass** (inverse correlation length).

({prf:ref}`ax-minimal-value-coupling`) *axiom* — **Minimal Value Coupling**

The belief current $J^\mu = \bar{\Psi} \gamma^\mu \Psi$ couples to the Value potential via minimal coupling:

$$
\mathcal{L}_{\text{Drive}} = J^\mu A^{\text{ext}}_\mu = -\rho(z) \Phi_{\text{eff}}(z)

$$

where $\rho = \Psi^\dagger \Psi = J^0$.

({prf:ref}`ax-causal-buffer-architecture`) *axiom* — **Causal Buffer Architecture**

Let the agent possess:
1. **$L_{\text{buf}}$:** Maximum buffer depth (spatial extent of causal memory)
2. **$\tau_{\text{proc}}$:** Minimum processing interval (temporal resolution)
3. **$d_{\text{sync}}$:** Minimum synchronization distance (coherence length)

These define the operational envelope within which the agent maintains consistent state updates.

({prf:ref}`ax-a-operational-distinguishability`) *axiom* — **A.6.0a (Operational Distinguishability)**

Two probability distributions $p, q \in \mathcal{P}(\mathcal{Z})$ are **operationally distinguishable** if and only if:

$$
D_{\text{KL}}(p \| q) \geq 1 \text{ nat}.

$$
*Justification.* This is an **operational definition**, not a derived fact. The choice of 1 nat as the threshold is grounded in:

1. **Asymptotic error exponent.** For $n$ i.i.d. samples, the optimal Type II error probability at fixed Type I error decays as $\exp(-n \cdot D_{\text{KL}})$ (Stein's lemma). Thus $D_{\text{KL}} = 1$ nat corresponds to error decay rate $e^{-n}$.

2. **Information-theoretic meaning.** 1 nat = log(e) ≈ 1.44 bits represents a "natural unit" of information, where the likelihood ratio $p(x)/q(x)$ has expected log-value 1 under $p$.

3. **Dimensional analysis.** The nat is the natural unit when using natural logarithms; choosing 1 nat as the threshold makes the subsequent formulas dimensionally consistent.

*Remark.* Alternative thresholds (e.g., 1 bit = ln 2 nats) would change the numerical coefficient in the Area Law but not its structure.

({prf:ref}`ax-the-bridge-principle`) *axiom* — **The Bridge Principle**

An agent commits to a **response function** $\sigma: \mathcal{O} \to \mathcal{A}$ mapping observations to actions, not a fixed policy $\pi: \mathcal{S} \to \mathcal{A}$ over states. The response function:

1. Is computable given bounded observations
2. Does not require access to opponent internal states or policies
3. Defines the agent's strategic interface at the boundary $\partial\mathcal{X}$

*Consequence:* Strategic interactions reduce to boundary conditions on the response function, eliminating the need for opponent omniscience.

## Assumptions

({prf:ref}`asm-regularity-conditions`) *assumption* — **Regularity Conditions for the Fragile Agent**

1. **Smoothness:** $V \in C^2(\mathcal{Z})$ --- the Hessian exists and is continuous
2. **Positive Definiteness:** $G(z) \succ 0$ for all $z \in \mathcal{Z}$ --- the metric is non-degenerate
3. **Lipschitz Dynamics:** $\|f(z_1, a) - f(z_2, a)\| \leq L\|z_1 - z_2\|$ --- no discontinuities
4. **Bounded State Space:** $\mathcal{Z}$ is compact, or $V$ has appropriate growth at infinity

## Lemmas

({prf:ref}`lem-variance-curvature-correspondence`) *lemma* — **Variance-Curvature Correspondence**

The covariance of the policy $\pi(a|z)$ is coupled to the curvature/sensitivity encoded by $G$. In entropy-regularized control, a natural scaling is:

$$
\Sigma_\pi(z) \propto \beta(z)^{-1} \cdot G^{-1}(z)

$$

({prf:ref}`lem-continuity-equation-for-transport`) *lemma* — **Continuity Equation for Transport**

If the belief density evolves only by deterministic transport under $v$ (no internal sources/sinks), then it satisfies the continuity equation

$$
\frac{\partial p}{\partial s} + \nabla_i \left( p v^i \right) = 0

$$
where $\nabla_i$ denotes the Levi-Civita covariant derivative associated with $G$.

({prf:ref}`lem-gromov-hyperbolicity`) *lemma* — **Gromov Hyperbolicity**

The tree metric space $(\mathcal{T}, d_{\mathcal{T}})$ is $0$-hyperbolic in the sense of Gromov. That is, for any geodesic triangle, each side is contained in the $0$-neighborhood of the union of the other two sides.

({prf:ref}`lem-metric-divergence-at-saturation`) *lemma* — **Metric Divergence at Saturation**

Consider an isotropic latent space of dimension $n \ge 3$ with polar coordinates $(r, \Omega)$. At saturation with uniform stress $T_{ij} = \sigma_{\max} G_{ij}$, the radial metric component $G_{rr} = A(r)$ satisfies the Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) and takes the form:

$$
A(r) = \left( 1 - \frac{2\mu(r)}{(n-2)r^{n-2}} - \frac{\Lambda_{\text{eff}} r^2}{n(n-1)} \right)^{-1},

$$
where $\mu(r) := \frac{\kappa}{n-2} \int_0^r \sigma_{\max} r'^{n-1} dr'$ is the integrated **information mass** (with $\kappa$ the coupling constant from the Metric Law) and $\Lambda_{\text{eff}} = \Lambda + \kappa\sigma_{\max}$.

*Remark ($n=2$ case).* For $n=2$ (the Poincare disk), the $(n-2)$ factor vanishes and the solution requires separate treatment. The Poincare metric $G_{ij} = 4\delta_{ij}/(1-|z|^2)^2$ is the correctly regularized saturation geometry, with the horizon at $|z|=1$.


*Critical observation.* The metric component $A(r)$ diverges at the horizon radius $r_h$ satisfying:

$$
1 - \frac{2\mu(r_h)}{(n-2)r_h^{n-2}} - \frac{\Lambda_{\text{eff}} r_h^2}{n(n-1)} = 0.

$$
At this radius, $G_{rr} \to \infty$ and consequently $G^{rr} \to 0$.

({prf:ref}`lem-virtual-work-of-recall`) *lemma* — **Virtual Work of Recall**

The infinitesimal work performed by the memory force during displacement $dz$ is:

$$
dW_{\text{mem}} := \langle -\nabla_G \Psi_{\text{mem}}, dz \rangle_G = -G_{kj}\,G^{k\ell}\partial_\ell \Psi_{\text{mem}}\, dz^j = -\partial_j \Psi_{\text{mem}}\, dz^j.

$$
*Units:* $[dW_{\text{mem}}] = \text{nat}$.

*Interpretation:* When the agent moves toward regions of low $\Psi_{\text{mem}}$ (attractive memory, i.e., $d\Psi_{\text{mem}} < 0$), positive work $dW_{\text{mem}} > 0$ is extracted from the memory field. This corresponds to "reward from recall"---revisiting previously successful states.

({prf:ref}`lem-default-mapping-to-vacuum`) *lemma* — **Default Mapping to Vacuum**

Let $\{q_i\}_{i=1}^{N_c}$ be the chart query bank (Definition {prf:ref}`def-attentive-routing-law`) and assume the queries are **centered**: $\sum_{i=1}^{N_c} q_i = 0$. Then for any key $k(x)$ such that all inner products are equal---$\langle q_i, k(x) \rangle = c$ for all $i$---the router weights are uniform:

$$
w_i(x) = \frac{1}{N_c} \quad \forall i \in \{1, \ldots, N_c\}.

$$
The resulting soft codebook embedding is the **barycenter**:

$$
z_q(x) = \sum_{i=1}^{N_c} w_i(x) e_{i, K_{\text{code},i}(x)} = \frac{1}{N_c} \sum_{i=1}^{N_c} e_{i,*},

$$
which equals $0$ if the per-chart codebooks are also centered ($\sum_c e_{i,c} = 0$ for each chart $i$).


*Interpretation.* When the observation $x$ is equally compatible with all charts (or incompatible with all), the router outputs uniform weights. Under centering, this maps to the vacuum---the maximum-entropy state in latent space.

**Architectural Requirement 30.1.3 (Codebook Centering).** To ensure the vacuum is reachable, initialize and regularize codebooks to satisfy $\sum_i q_i = 0$ and $\sum_c e_{i,c} = 0$. This can be enforced via:

$$
\mathcal{L}_{\text{center}} := \left\|\sum_{i=1}^{N_c} q_i\right\|^2 + \sum_{i=1}^{N_c} \left\|\sum_{c=1}^{N_v} e_{i,c}\right\|^2.

$$

({prf:ref}`lem-redundancy-gain`) *lemma* — **Redundancy-Gain Relationship**

Under the assumption that charts partition the observation space and the encoder is deterministic given observation $x$:

$$
G_\Delta(i, j) \leq H(K_i, K_j) - H(K_{i \cup j}) = \log 2 - H(K_i | K_j) \cdot \mathbb{I}[\Upsilon_{ij} < 1]

$$
When $\Upsilon_{ij} \to 1$, the bound tightens: $G_\Delta \to 0$.

({prf:ref}`lem-the-interventional-singularity`) *lemma* — **The Interventional Singularity**

An intervention at state $z$ is a point-source singularity in the field theory. It imposes a non-natural boundary condition that forces the system to explore the off-equilibrium response of the environment law $P_\partial$ ({ref}`Section 1.1.1 <sec-the-environment-is-an-input-output-law>`).


*Remark (Surgery vs. Conditioning).* The key distinction from Bayesian conditioning is that $P(z' | do(a)) \neq P(z' | a)$ in general. Conditioning updates beliefs given evidence; intervention changes the generating mechanism. The former is reversible; the latter is a topological surgery.

({prf:ref}`lem-friction-bounds-utility`) *lemma* — **Metric Friction Bounds Cooperative Utility**

Let $V_{\text{coop}}$ denote the cooperative value achievable by agents $A$ and $B$. The friction bound is:

$$
V_{\text{coop}} \leq V_{\text{max}} \cdot \exp\left(-\frac{\mathcal{F}_{AB}}{\mathcal{F}_0}\right)

$$

where $V_{\text{max}}$ is the optimal cooperative value under perfect alignment and $\mathcal{F}_0$ is a characteristic friction scale.

({prf:ref}`lem-gradient-observability`) *lemma* — **Gradient Observability**

The gradient $g$ uniquely determines the local metric tensor $G(\Theta + \epsilon g)$ to first order:

$$
G_{ij}(\Theta + \epsilon g) = G_{ij}(\Theta) + \epsilon \, \partial_k G_{ij} \cdot g^k + O(\epsilon^2)

$$


*Consequence:* Validators can infer each other's metrics from observed gradients without direct communication.

({prf:ref}`lem-a-divergence-to-boundary-conversion`) *lemma* — **A.3.1 (Divergence-to-boundary conversion)**

For any sufficiently regular information flux field $\mathbf{j}$ on $\mathcal{Z}$,

$$
\int_{\mathcal{Z}} \operatorname{div}_G(\mathbf{j})\, d\mu_G = \oint_{\partial \mathcal{Z}} \langle \mathbf{j}, \mathbf{n}\rangle\, dA_G,

$$
which is the Riemannian divergence theorem underlying the global balance equation in Theorem {prf:ref}`thm-generalized-conservation-of-belief`.

({prf:ref}`lem-a-geodesic-distance-probability-simplex`) *lemma* — **A.6.0d (Geodesic Distance on the Probability Simplex)**

On the 1-simplex $\Delta^1 = \{(p, 1-p) : p \in [0,1]\}$ with Fisher Information Metric, the geodesic distance from the uniform distribution $(1/2, 1/2)$ to a vertex $(1, 0)$ is:

$$
d_{\text{Fisher}}\left(\tfrac{1}{2}, 1\right) = \frac{\pi}{2}.

$$

$$
ds^2 = \frac{dp^2}{p(1-p)}.

$$
Introduce the angular parameterization $p = \cos^2(\theta/2)$, so that $1-p = \sin^2(\theta/2)$ and:

$$
dp = -\cos(\theta/2)\sin(\theta/2)d\theta = -\frac{1}{2}\sin\theta \, d\theta.

$$
Then:

$$
ds^2 = \frac{\frac{1}{4}\sin^2\theta \, d\theta^2}{\cos^2(\theta/2)\sin^2(\theta/2)} = \frac{\frac{1}{4}\sin^2\theta \, d\theta^2}{\frac{1}{4}\sin^2\theta} = d\theta^2.

$$
The uniform distribution $(1/2, 1/2)$ corresponds to $\theta = \pi/2$. The vertex $(1, 0)$ corresponds to $\theta = 0$. The geodesic distance is:

$$
d = \int_0^{\pi/2} d\theta = \frac{\pi}{2}. \quad \square

$$
*Interpretation.* One bit of information (distinguishing "heads" from "tails") corresponds to geodesic distance $\pi/2$ in Fisher geometry. This is a derived quantity, not an assumption.

({prf:ref}`lem-a-curvature-normalization-factor-4`) *lemma* — **A.6.0e (Curvature Normalization and the Factor of 4)**

The Poincare disk model with constant sectional curvature $K = -1$ has metric:

$$
ds^2 = \frac{4(dx^2 + dy^2)}{(1-|z|^2)^2}.

$$
The factor of 4 is uniquely determined by the curvature normalization.

$$
K = -\frac{1}{2\lambda}\Delta(\log \lambda),

$$
where $\Delta = \partial_x^2 + \partial_y^2$ is the flat Laplacian.

For $\lambda = c/(1-r^2)^2$ where $r^2 = x^2 + y^2$ and $c > 0$:

**Step 1:** Compute $\log \lambda = \log c - 2\log(1-r^2)$.

**Step 2:** Compute the Laplacian. Let $f = \log(1-r^2)$. Then:

$$
\partial_x f = \frac{-2x}{1-r^2}.

$$
Applying the quotient rule to $\partial_x f = -2x \cdot (1-r^2)^{-1}$:

$$
\partial_x^2 f = \frac{-2(1-r^2) - (-2x)(-2x)}{(1-r^2)^2} = \frac{-2 + 2r^2 - 4x^2}{(1-r^2)^2}.

$$
Similarly for $y$. Adding:

$$
\Delta f = \frac{(-2 + 2r^2 - 4x^2) + (-2 + 2r^2 - 4y^2)}{(1-r^2)^2} = \frac{-4 + 4r^2 - 4r^2}{(1-r^2)^2} = \frac{-4}{(1-r^2)^2}.

$$
**Step 3:** Therefore $\Delta(\log \lambda) = -2\Delta f = \frac{8}{(1-r^2)^2}$.

**Step 4:** The curvature is:

$$
K = -\frac{1}{2\lambda} \cdot \frac{8}{(1-r^2)^2} = -\frac{(1-r^2)^2}{2c} \cdot \frac{8}{(1-r^2)^2} = -\frac{4}{c}.

$$
**Step 5:** For $K = -1$, we require $c = 4$. $\square$

*Significance.* The choice $K = -1$ is canonical: it sets the "radius of curvature" to unity, making the hyperbolic distance formula $d(0,z) = 2\text{arctanh}|z|$ dimensionless. The factor of 4 in the metric is a *derived consequence* of the curvature normalization, not an assumption.

({prf:ref}`lem-a-bulk-to-boundary-conversion`) *lemma* — **A.6.1 (Bulk-to-Boundary Conversion)**

For a stationary information distribution satisfying the Metric Law, the bulk information integral can be expressed as a boundary integral:

$$
I_{\text{bulk}} = \int_{\mathcal{Z}} \rho_I \, d\mu_G = \frac{1}{\kappa} \oint_{\partial\mathcal{Z}} \text{Tr}(K) \, dA_G,

$$
where $K_{ij}$ is the extrinsic curvature (second fundamental form) of the boundary and $\kappa$ is the coupling constant from the Metric Law.

$$
R - 2\Lambda = \kappa \, T,

$$
where $T = G^{ij}T_{ij}$ is the trace of the stress tensor. For uniform saturation, $T = n \cdot \sigma_{\max}$.

Integrating the Einstein tensor identity over $\mathcal{Z}$ and applying Lemma {prf:ref}`lem-a-divergence-to-boundary-conversion`:

$$
\int_{\mathcal{Z}} R \, d\mu_G = 2 \oint_{\partial\mathcal{Z}} \text{Tr}(K) \, dA_G.

$$
Combining with $R = \kappa T + 2\Lambda$ and noting that the $\Lambda$ term contributes a volume integral that cancels under the capacity constraint, we obtain the stated identity. $\square$

({prf:ref}`lem-a-geodesic-distance-simplex`) *lemma* — **A.6.4 (Geodesic Distance on the Probability Simplex)**

On the 1-simplex $\Delta^1 = \{(p, 1-p) : p \in [0,1]\}$ with the Fisher Information Metric, the geodesic distance between the uniform distribution $(1/2, 1/2)$ and a vertex $(1, 0)$ is:

$$
d_{\text{Fisher}}\left(\frac{1}{2}, 1\right) = \frac{\pi}{2}.

$$

## Propositions

({prf:ref}`prop-mass-conservation-in-a-closed-enclosure`) *proposition* — **Mass Conservation in a Closed Enclosure**

If $\sigma\equiv 0$ and the boundary flux vanishes (e.g. $\langle p v,n\rangle=0$ on $\partial\mathcal{Z}$), then the total belief mass

$$
\mathcal{V}(s):=\int_{\mathcal{Z}}p(z,s)\,d\mu_G

$$
is constant in time.

$$
\frac{d\mathcal{V}}{ds} = \int_{\mathcal{Z}} \frac{\partial p}{\partial s} d\mu_G = -\int_{\mathcal{Z}} \operatorname{div}_G(p v) d\mu_G = -\int_{\partial \mathcal{Z}} \langle p v, n \rangle dA = 0

$$
assuming there is no net boundary contribution and no internal source term. In applications we do not estimate $\sigma$ pointwise; instead we monitor surrogate checks (e.g. BoundaryCheck and coupling-window metrics) that are sensitive to persistent boundary decoupling (Sections 3 and 15).

({prf:ref}`prop-texture-as-the-ideal-boundary`) *proposition* — **Texture as the Ideal Boundary**

Let $\mathcal{M}$ be the Riemannian manifold constructed above. The **texture residual** $z_{\mathrm{tex}}$ corresponds to the behavior of the state at the **conformal boundary at infinity**, $\partial_\infty \mathbb{H}^n$.

1. Consider a sequence of refining codes $(K_{\text{chart}}^{(n)}, K_{\text{code}}^{(n)})$ representing a path $\gamma$ in the tree $\mathcal{T}$ extending to infinite depth.
2. As the depth $n \to \infty$, the volume of the region covered by code $K^{(n)}$ in the observation space $\mathcal{X}$ shrinks to zero (assuming a non-degenerate shutter).
3. In the hyperbolic metric of the latent space, the distance from the basepoint $d(o, \gamma(n)) \to \infty$.
4. The residual $z_{\mathrm{tex}}$ is defined as the information remaining after finite truncation at level $n$. Specifically, $z_{\mathrm{tex}} = \Delta_{\text{total}} - z_n$.
5. If we interpret the encoding process as a flow toward the boundary of $\mathbb{H}^n$, then $z_{\mathrm{tex}}$ represents the **transverse coordinates** at the cutoff surface $\Sigma_\epsilon$.
6. Taking the limit $\epsilon \to 0$, $z_{\mathrm{tex}}$ maps to the **limit set** $\Lambda \subset \partial_\infty \mathbb{H}^n$. The mathematical structure parallels the AdS/CFT bulk-boundary correspondence: the fields $(K, z_n)$ reconstruct $(x)$ up to a cutoff; $z_{\mathrm{tex}}$ is the UV (high-frequency) data living strictly at the conformal boundary. $\square$

**Operational Implication:**
This formalizes why $z_{\mathrm{tex}}$ must be excluded from dynamics ($S_t$) and control ($\pi_\theta$). The dynamics $S_t$ operate on the **bulk** (finite-energy excitations inside the hyperbolic volume). The texture $z_{\mathrm{tex}}$ lives at the **boundary at infinity** (infinite energy / zero scale). Coupling the bulk dynamics to the boundary fluctuations violates the separation of scales and leads to the Labyrinthine failure mode (Mode T.C).

({prf:ref}`prop-gradient-preservation-via-orthogonality`) *proposition* — **Gradient Preservation via Orthogonality**

Let $W$ be a weight matrix satisfying $W^T W = I$ (semi-orthogonality). Then:
1. All singular values of $W$ equal 1.
2. The backward gradient $\nabla_x \mathcal{L} = W^T \nabla_y \mathcal{L}$ satisfies $\|\nabla_x \mathcal{L}\| = \|\nabla_y \mathcal{L}\|$.
3. Neither explosion nor vanishing occurs across the layer.


This is why the gradient flow table ({ref}`Section 7.7.2 <sec-orthonormal-constraints-for-atlas-charts>`) shows Preserved for orthogonal $W$ versus Explodes or vanishes for arbitrary $W$.

({prf:ref}`prop-forward-activation-stability`) *proposition* — **Forward Activation Stability**

With variance rescaling:
1. $\mathrm{Var}(x^{(\ell)}) = 1$ for all $\ell$ (by construction).
2. Non-linearities (GELU) operate in their active region, avoiding saturation.
3. The backward gradient is scaled by $1/\sigma^{(\ell)}$, amplifying gradients for fine-scale layers.

**Gradient Amplification Analysis:** Let the loss $\mathcal{L}$ depend on the output of block $\ell$. The gradient flowing back to block $\ell-1$ includes the factor:

$$
\frac{\partial x^{(\ell)}}{\partial z_{\mathrm{tex}}^{(\ell-1)}} = \frac{1}{\sigma^{(\ell-1)}}

$$
Since each block successfully explains part of the signal, the residual standard deviation $\sigma^{(\ell)} < 1$ (the texture has less variance than the unit-normalized input). This implies:
- **Without rescaling:** inputs to deeper layers decay exponentially ($\|x^{(\ell)}\| \to 0$), killing activations.
- **With rescaling:** inputs $x^{(\ell)}$ remain $O(1)$ (unit variance), keeping non-linearities in their active region.
- **Gradient amplification:** the backward gradient includes the factor $1/\sigma^{(\ell-1)} > 1$, counteracting the natural decay of fine-scale influence on the global loss.

This prevents the **Spectral Bias** where neural networks preferentially learn low frequencies and ignore high-frequency structure.

({prf:ref}`prop-parameter-efficiency`) *proposition* — **Parameter Efficiency**

The factorized parameterization requires $O(K \cdot r \cdot d_n)$ parameters instead of $O(K^2 \cdot d_n^2)$.


For typical values ($K = 64$, $d_n = 16$, $r = 8$), this yields $64 \times (2 \times 8 \times 16 + 8 + 16) = 17,920$ parameters—approximately a $58\times$ reduction compared to the naive $\sim 10^6$.

({prf:ref}`prop-soft-bellman-form-discrete-actions`) *proposition* — **Soft Bellman form, discrete actions**

Assume finite $\mathcal{A}$. Define the soft state value

$$
V^*(k) := \max_{\pi} \ \mathbb{E}\Big[\sum_{t\ge 0}\gamma^t(\mathcal{R}+T_c\mathcal{H})\ \Big|\ K_0=k\Big].

$$
Then $V^*$ satisfies the entropic Bellman fixed point

$$
V^*(k)
=
T_c \log \sum_{a\in\mathcal{A}}
\exp\!\left(\frac{1}{T_c}\left(\mathcal{R}(k,a)+\gamma\,\mathbb{E}_{k'\sim\bar{P}(\cdot\mid k,a)}[V^*(k')]\right)\right),

$$
and the corresponding optimal policy is the softmax policy

$$
\pi^*(a\mid k)\propto
\exp\!\left(\frac{1}{T_c}\left(\mathcal{R}(k,a)+\gamma\,\mathbb{E}[V^*(k')]\right)\right).

$$
**Consequence.** The same mathematics can be read as:
1) maximize reward while retaining policy entropy (MaxEnt RL), or
2) maximize reachability/diversity of future macro trajectories (intrinsic motivation).

({prf:ref}`prop-limiting-regimes`) *proposition* — **Limiting Regimes**

The WFR metric seamlessly unifies discrete and continuous dynamics:

1. **Continuous Movement (Flow):** When moving within a chart, $r \approx 0$. The dynamics are dominated by $\nabla \cdot (\rho v)$, and the metric reduces to $W_2$ (Wasserstein-2). This recovers the Riemannian manifold structure of the nuisance fibres.

2. **Discrete Movement (Jump):** When the flow reaches a topological obstruction (chart boundary without overlap), transport becomes infinitely expensive. It becomes cheaper to use the source term $r$:
   - $r < 0$ on the old chart (mass destruction)
   - $r > 0$ on the new chart (mass creation)
   This recovers the **Fisher-Rao metric** on the discrete simplex $\Delta^{|\mathcal{K}|}$.

3. **Mixed Regime (Overlap):** In chart overlaps, both $v$ and $r$ are active. The optimal path smoothly interpolates between transport and reaction.

({prf:ref}`prop-gksl-embedding`) *proposition* — **GKSL Embedding**

The GKSL generator from Definition {prf:ref}`def-gksl-generator` embeds into the WFR framework as follows:
- The Hamiltonian $H$ generates the velocity field via $v \propto G^{-1}\nabla_z \langle H \rangle_\varrho$ (gradient of expected energy)
- Each Lindblad operator $L_j$ contributes to the reaction rate via $r \propto \sum_j \gamma_j(\mathrm{Tr}(L_j^\dagger L_j \varrho) - 1)$

This provides a **geometric foundation** for the otherwise algebraic GKSL construction. The correspondence is heuristic; see Carlen & Maas (2014) {cite}`carlen2014wasserstein` for rigorous connections between quantum Markov semigroups and gradient flows on Wasserstein space.

({prf:ref}`prop-isotropic-radial-expansion`) *proposition* — **Isotropic Radial Expansion**

If acting alone (no policy steering), the entropic drift produces the isotropic expansion:

$$
r(\tau) = \tanh(\tau/2)

$$
This represents isotropic diffusion---expanding uniformly in all directions.

({prf:ref}`prop-riemannian-gradient-of`) *proposition* — **Riemannian Gradient of $U$**

The gradient in the Poincare metric is:

$$
\nabla_G U(z) = G^{-1} \nabla U = -\frac{(1-|z|^2)}{2} z.

$$
The **entropic drift** (negative gradient) pushes radially outward:

$$
-\nabla_G U(z) = \frac{(1-|z|^2)}{2} z.

$$
*Remark (Connection to {ref}`Section 7.11 <sec-the-geometry-of-the-latent-space-a-hyperbolic-hierarchy>`).* The Poincare coordinate $z$ relates to depth via $\rho = d_{\mathbb{D}}(0, z) = 2\operatorname{artanh}(|z|)$. Chart transitions are handled by the WFR jump process ({ref}`Section 22.2 <sec-the-coupled-jump-diffusion-sde>`), governed by the {prf:ref}`def-the-wfr-action`.

**Cross-references:** Definition {prf:ref}`def-information-density-and-bulk-information-volume`, Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

({prf:ref}`prop-so-d-symmetry-at-origin`) *proposition* — **SO(D) Symmetry at Origin**

At $z = 0$:
1. The metric is isotropic: $G(0) = 4I$
2. The entropic force vanishes: $F_{\text{entropy}}(0) = 0$
3. The system has full rotational symmetry $SO(D)$

*Cross-reference (Gauge Breaking):* This $SO(D)$ symmetry is the special case where the stabilizer subgroup $H_0 = \{e\}$ is trivial. In multi-agent settings, this symmetry is spontaneously broken via the Higgs mechanism (Theorem {prf:ref}`thm-higgs-mechanism`), yielding massive gauge bosons and effective agent masses.

({prf:ref}`prop-conformal-texture-scaling`) *proposition* — **Conformal Texture Scaling**

The texture variance scales with the inverse metric:

| **Region** | **$\lvert z\rvert$** | **$\Sigma(z)$**                            | **Interpretation**        |
|------------|----------------------|--------------------------------------------|---------------------------|
| Origin     | $\approx 0$          | $\sigma_{\text{tex}}^2/4 \cdot I$          | Moderate texture (coarse) |
| Mid-disk   | $\approx 0.5$        | $\sigma_{\text{tex}}^2 \cdot 9/64 \cdot I$ | Reduced texture           |
| Boundary   | $\to 1$              | $\to 0$                                    | Deterministic texture     |

*Remark (Conformal suppression).* Near the boundary (high resolution/specificity), the metric $G$ diverges, so $G^{-1} \to 0$ and texture fluctuations are suppressed.

({prf:ref}`prop-epistemic-barrier`) *proposition* — **Epistemic Barrier**

The partition condition enforces **BarrierEpi** (Epistemic Limit): The agent does not waste capacity predicting the noise---it only predicts the *statistics* of the noise ($\Sigma$).

({prf:ref}`prop-mass-scaling-near-boundary`) *proposition* — **Mass Scaling Near Boundary**

For the Poincare disk, the mass tensor scales as:

$$
\mathbf{M}(z) = \frac{4}{(1-|z|^2)^2} I_d \quad \xrightarrow{|z| \to 1} \quad +\infty.

$$
The metric diverges as $|z| \to 1$, which bounds all finite-action trajectories to the interior of the disk.

({prf:ref}`prop-most-probable-path`) *proposition* — **Most Probable Path**

For the controlled diffusion

$$
dz^k = b^k(z)\,ds + \sqrt{2T_c}\,\sigma^{kj}(z)\,dW^j_s,

$$
where $\sigma \sigma^T = G^{-1}$, the most probable path connecting $z(0) = z_0$ and $z(T) = z_1$ minimizes the Onsager-Machlup action $S_{\mathrm{OM}}[z]$ subject to the boundary conditions.

({prf:ref}`prop-a-explicit-christoffel-symbols-for-poincar-disk`) *proposition* — **a (Explicit Christoffel Symbols for Poincare Disk)**

For the Poincare disk model with metric $G_{ij} = \frac{4\delta_{ij}}{(1-|z|^2)^2}$, the Christoffel symbols in Cartesian coordinates are:

$$
\Gamma^k_{ij}(z) = \frac{2}{1-|z|^2}\left(\delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k\right).

$$
The geodesic correction term $\Gamma^k_{ij}\dot{z}^i\dot{z}^j$ contracts to:

$$
\Gamma^k_{ij}\dot{z}^i\dot{z}^j = \frac{4(z \cdot \dot{z})}{1-|z|^2}\dot{z}^k - \frac{2|\dot{z}|^2}{1-|z|^2}z^k.

$$

*Geometric interpretation:* The first term $(z \cdot \dot{z})\dot{z}$ accelerates motion radially when moving outward; the second term $|\dot{z}|^2 z$ provides centripetal correction. Together they ensure geodesics are circular arcs perpendicular to the boundary.

({prf:ref}`prop-jump-intensity-from-value-discontinuity`) *proposition* — **Jump Intensity from Value Discontinuity**

The jump intensity $\lambda_{\text{jump}}(z)$ is determined by the value difference across chart boundaries:

$$
\lambda_{\text{jump}}(z) = \lambda_0 \cdot \exp\left(\beta \cdot \left( V_{\text{target}}(L(z)) - V_{\text{source}}(z) - c_{\text{transport}} \right) \right),

$$
where:
- $\lambda_0 > 0$ is a base jump rate
- $\beta > 0$ is the inverse temperature (sharpness)
- $V_{\text{target}}$ and $V_{\text{source}}$ are the value functions on the target and source charts
- $L: \mathcal{Z}_{\text{source}} \to \mathcal{Z}_{\text{target}}$ is the chart transition operator
- $c_{\text{transport}} \ge 0$ is the transport cost (WFR term)

*Remark (SMC Interpretation).* The mass $m(s)$ is precisely the **importance weight** in Sequential Monte Carlo (SMC) / particle filtering. The agent is a single-particle realization of the WFR flow from {ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`. Multiple particles can be used for ensemble-based generation.

**Cross-references:** {ref}`Section 20.2 <sec-the-wfr-metric>` ({prf:ref}`def-the-wfr-action`), {ref}`Section 20.6 <sec-the-unified-world-model>` (WFR world model), {ref}`Section 11 <sec-intrinsic-motivation-maximum-entropy-exploration>` (Filtering and projection).

({prf:ref}`prop-mode-interpretation`) *proposition* — **Mode Interpretation**

The parameter $\alpha$ interpolates between pure generation and pure control:

| Regime              | $\alpha$ Value      | Behavior                                                       |
|---------------------|---------------------|----------------------------------------------------------------|
| **Pure Generation** | $\alpha = 1$        | Flow follows $-\nabla_G U$ (holographic expansion, {ref}`Section 21 <sec-radial-generation-entropic-drift-and-policy-control>`) |
| **Pure Control**    | $\alpha = 0$        | Flow follows $-\nabla_G V_{\text{critic}}$ (policy gradient)   |
| **Hybrid**          | $\alpha \in (0, 1)$ | Balanced generation and control                                |

*Remark (Risk Modulation).* The $\gamma_{risk}$ term provides an additional penalty in high-stress regions (large $T_{ij}$), which further discourages risky trajectories beyond the geometric slowdown from Mass=Metric.

({prf:ref}`prop-baoab-preserves-boltzmann`) *proposition* — **BAOAB Preserves Boltzmann**

The BAOAB integrator preserves the Boltzmann distribution $\rho(z, p) \propto \exp(-\Phi_{\text{eff}}(z)/T_c - \|p\|_G^2 / (2T_c))$ to second order in $h$.


*Remark (Comparison to Euler-Maruyama).* Euler-Maruyama has $O(h)$ bias in the stationary distribution, whereas BAOAB achieves $O(h^2)$. For long trajectories, this difference is critical.

({prf:ref}`prop-phase-transition-interpretation`) *proposition* — **Phase Transition Interpretation**

The agent lifecycle corresponds to a thermodynamic phase transition:

| Phase | Thermodynamic Analogy | Order Parameter |
|-------|----------------------|-----------------|
| Init (gas) | High entropy, symmetric | $\lVert z\rVert = 0$ |
| Kick (nucleation) | Symmetry breaking | $u_\pi \neq 0$ |
| Bulk (liquid) | Directed flow | $0 < \lVert z\rVert < R_{cutoff}$ |
| Boundary (solid) | Crystallization | $\lVert z\rVert \geq R_{cutoff}$ |

({prf:ref}`prop-automatic-phase-transitions`) *proposition* — **Automatic Phase Transitions**

With adaptive temperature $T_c(z)$ satisfying the Einstein relation:

| Regime                      | Metric $G(z)$ | Effective Noise | Phase Behavior                |
|-----------------------------|---------------|-----------------|-------------------------------|
| **Uncertain** (near origin) | Small         | Large           | Gas phase (exploration)       |
| **Certain** (near boundary) | Large         | Small           | Solid phase (crystallization) |

*Remark.* This automatic phase transition emerges from the geometry alone---no explicit temperature schedule is needed.

({prf:ref}`prop-symplectic-duality-principle`) *proposition* — **Symplectic Duality Principle**

Under the canonical transformation $(q, p) \mapsto (p, -q)$:
- Dirichlet conditions become Neumann conditions
- Sensors become motors
- Perception becomes action

This duality is the mathematical foundation for the symmetric treatment of sensing and actuation.


**Cross-references:** {ref}`Section 2.11.4 <sec-the-interface-and-observation-inflow>` (Observation inflow), Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`.

({prf:ref}`prop-carnot-efficiency-bound`) *proposition* — **Carnot Efficiency Bound**

The agent's efficiency in converting sensory information to control information is bounded:

$$
\eta = \frac{I(A_t; K_t)}{I(X_t; K_t)} \leq 1 - \frac{T_{\text{motor}}}{T_{\text{sensor}}},

$$
where $T_{\text{sensor}}$ and $T_{\text{motor}}$ are the effective temperatures at the sensory and motor boundaries.

*Interpretation:* Perfect efficiency ($\eta = 1$) requires $T_{\text{motor}} = 0$ (deterministic motors) or $T_{\text{sensor}} \to \infty$ (infinite sensory entropy). Real systems operate at $\eta < 1$.

**Cross-references:** {ref}`Section 22.7 <sec-adaptive-thermodynamics>` (Adaptive Thermodynamics), {ref}`Section 14.2 <sec-the-equivalence-theorem>` (MaxEnt Control).

*Forward reference (Reward as Heat).* {ref}`Section 24.3 <sec-the-bulk-potential-screened-poisson-equation>` establishes that Reward is the thermodynamic **heat input** that drives the cycle: the Boltzmann-Value Law (Axiom {prf:ref}`ax-the-boltzmann-value-law`) identifies $V(z) = E(z) - T_c S(z)$ as Gibbs Free Energy, and Theorem {prf:ref}`thm-wfr-consistency-value-creates-mass` proves that WFR dynamics materialize the agent in high-value regions ("Value Creates Mass").

({prf:ref}`prop-grounding-rate-via-boundary-flux`) *proposition* — **Grounding Rate via Boundary Flux**

The grounding rate (cf. Definition 16.1.1) is:

$$
G_t = \oint_{\partial\mathcal{Z}_{\text{sense}}} j_{\text{obs}} \cdot dA - \oint_{\partial\mathcal{Z}_{\text{motor}}} j_{\text{motor}} \cdot dA,

$$
which is:
- **Positive** during waking (net information inflow from sensors)
- **Zero** during dreaming (closed system)
- **Negative** during pure actuation (net information outflow to motors)

**Cross-references:** {ref}`Section 20.2 <sec-the-wfr-metric>` (WFR Action), {ref}`Section 20.6 <sec-the-unified-world-model>` (WFR World Model), Section 2.11.4 (Observation Inflow).

({prf:ref}`prop-value-cycle-detection`) *proposition* — **Value Cycle Detection**

The Value Curl $\mathcal{F}$ can be estimated from trajectory data. For a closed loop $\gamma$ in latent space:

$$
\oint_\gamma \mathcal{R} = \int_\Sigma \mathcal{F} \, d\Sigma \neq 0 \implies \text{Non-conservative rewards.}

$$
**Diagnostic:** If the TD-error accumulated around closed loops in latent space has non-zero mean, the value field is non-conservative.

({prf:ref}`prop-green-s-function-interpretation`) *proposition* — **Green's Function Interpretation**

The Critic computes the **Green's function** of the screened Laplacian on the latent geometry:

$$
V(z) = \int_{\partial\Omega} G_\kappa(z, z') \sigma_r(z') \, d\Sigma(z'),

$$
where $G_\kappa(z, z')$ is the Green's function satisfying $(-\Delta_G + \kappa^2) G_\kappa(z, \cdot) = \delta_z$.

*Remark.* The value at $z$ is a weighted integral of boundary rewards, with weights given by the Green's function. This is a superposition principle: the Helmholtz equation is linear.

({prf:ref}`prop-green-s-function-decay`) *proposition* — **Green's Function Decay**

On a manifold with bounded curvature, the Green's function decays exponentially:

$$
G_\kappa(z, z') \sim \frac{1}{d_G(z, z')^{(d-2)/2}} \exp\left(-\kappa \cdot d_G(z, z')\right),

$$
where $d_G$ is the geodesic distance and $d$ is the dimension.

({prf:ref}`prop-ness-decomposition`) *proposition* — **NESS Decomposition**

The probability current in a NESS decomposes into:

$$
J = J_{\text{gradient}} + J_{\text{cyclic}}

$$
where:
- $J_{\text{gradient}} = -D\rho\nabla\ln\rho + \rho\nabla\Phi$ derives from the scalar potential
- $J_{\text{cyclic}} = \rho \cdot v_{\text{curl}}$ derives from the solenoidal component

At stationarity, $\nabla \cdot J = 0$, but only $J_{\text{gradient}} = 0$ at true equilibrium. NESS has $J_{\text{cyclic}} \neq 0$.

({prf:ref}`prop-risk-curvature-mechanism`) *proposition* — **Risk-Curvature Mechanism**

The conformal factor encodes the local "importance" of the value landscape:

| Value Landscape           | $\lVert\nabla^2 V\rVert$ | $\Omega$    | Effect                           |
|---------------------------|--------------------------|-------------|----------------------------------|
| **Flat** (low importance) | $\approx 0$              | $\approx 1$ | Default hyperbolic bulk geometry |
| **Curved** (ridge/valley) | $\gg 0$                  | $\gg 1$     | Distances expand, mass increases |
| **Saddle** (transition)   | moderate                 | $> 1$       | Intermediate slowdown            |

({prf:ref}`prop-conformal-laplacian-transformation`) *proposition* — **Conformal Laplacian Transformation**

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

*Interpretation:* In high-curvature regions ($\Omega$ large), the effective screening mass decreases, making the field more "massless" and allowing longer-range correlations. This is the **self-focusing** effect: important regions become more interconnected.

**Cross-references:** Theorem {prf:ref}`thm-capacity-constrained-metric-law`, {ref}`Section 22.1 <sec-the-stochastic-action-principle>` (Mass=Metric), Proposition {prf:ref}`prop-mass-scaling-near-boundary`.

({prf:ref}`prop-soft-injectivity`) *proposition* — **Soft Injectivity**

The sub-atlases need not be disjoint. Charts in $\mathcal{A}_i \cap \mathcal{A}_j$ for $i \neq j$ are **transition regions** characterized by:

1. **Low purity:** $\max_y P(Y=y \mid K=k) < 1 - \epsilon_{\text{purity}}$ for all $y$
2. **High entropy:** $H(Y \mid K=k) > H_{\text{transition}}$ (conditional entropy; see {cite}`cover1991elements`)
3. **Low information content:** These charts carry less semantic information per the information bottleneck principle {cite}`tishby2015ib`

*Remark (Geometric Interpretation).* Transition charts correspond to saddle regions of the semantic potential landscape---unstable fixed points between class regions of attraction.

**Cross-references:** {ref}`Section 23.6 <sec-relationship-to-the-context-conditioned-framework>` (Context-Conditioned Policies), Definition 2.2.1 (Macro-State Register), {ref}`Section 7.8 <sec-tier-the-attentive-atlas>` (Router Weights).

({prf:ref}`prop-effective-disconnection`) *proposition* — **Effective Disconnection**

As $\gamma_{\text{sep}} \to \infty$, the effective WFR distance between charts of different classes diverges:

$$
d_{\text{WFR}}(\mathcal{A}_{y_1}, \mathcal{A}_{y_2}) \to \infty \quad \text{for } y_1 \neq y_2.

$$
1. **Transport-only paths:** If $\mathcal{A}_{y_1}$ and $\mathcal{A}_{y_2}$ are not geometrically adjacent (no shared chart boundary), pure transport paths have infinite cost.

2. **Jump paths:** Any path using cross-class jumps incurs reaction cost. In the GKSL interpretation ({ref}`Section 20.5 <sec-connection-to-gksl-master-equation>`), the suppressed jump rate $\lambda^{\text{sup}} = \lambda^{(0)} e^{-\gamma_{\text{sep}}}$ means mass transfer between unlike-class charts requires longer dwell times, increasing the action.

3. **Divergence:** As $\gamma_{\text{sep}} \to \infty$, cross-class jumps become arbitrarily rare. The optimal path cost diverges because: (a) pure transport is blocked by chart boundaries, and (b) the reaction term penalizes staying in transition states waiting for rare jumps.

The precise scaling (exponential, polynomial, etc.) depends on the manifold geometry, but divergence is guaranteed. $\square$

({prf:ref}`prop-purity-information-duality`) *proposition* — **Purity-Information Duality**

Minimizing $\mathcal{L}_{\text{purity}}$ is equivalent to maximizing the mutual information $I(K; Y)$:

$$
\mathcal{L}_{\text{purity}} = H(Y) - I(K; Y).

$$
Since $H(Y)$ is fixed by the data, $\min \mathcal{L}_{\text{purity}} \Leftrightarrow \max I(K; Y)$.

({prf:ref}`prop-class-conditioned-langevin`) *proposition* — **Class-Conditioned Langevin**

The generative Langevin equation {cite}`welling2011sgld,song2019ncsn` (Definition {prf:ref}`prop-so-d-symmetry-at-origin`) with class conditioning becomes:

$$
dz = -\nabla_G V_y(z, K)\,d\tau + \sqrt{2T_c}\,G^{-1/2}(z)\,dW_\tau,

$$
where $V_y$ is the class-conditioned potential (Definition {prf:ref}`def-class-conditioned-potential`).

*Interpretation:* To generate a sample of class $y$, we run Langevin dynamics with the $V_y$ potential. The semantic term $-\beta_{\text{class}} \log P(Y=y \mid K)$ biases the flow toward class-$y$ charts.

({prf:ref}`prop-scale-label-alignment`) *proposition* — **Scale-Label Alignment**

In the stacked TopoEncoder ({ref}`Section 7.12 <sec-stacked-topoencoders-deep-renormalization-group-flow>`), enforce purity at each scale:

- **Layer 0 (Bulk/Slow):** Charts at level 0 correspond to coarse classes. Enforce:

  $$
  \mathcal{L}_{\text{purity}}^{(0)} = H(\mathcal{Y}_0 \mid K^{(0)})

  $$
- **Layer $\ell$ (Intermediate):** Charts at level $\ell$ correspond to level-$\ell$ classes. Enforce:

  $$
  \mathcal{L}_{\text{purity}}^{(\ell)} = H(\mathcal{Y}_\ell \mid K^{(\ell)})

  $$
- **Layer $L$ (Boundary/Fast):** Charts at level $L$ correspond to fine classes. Enforce:

  $$
  \mathcal{L}_{\text{purity}}^{(L)} = H(\mathcal{Y}_L \mid K^{(L)})

  $$

({prf:ref}`prop-subsumption-of-section`) *proposition* — **Subsumption of {ref}`Section 3.5 <sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration>`**

The methods of {ref}`Section 3.5 <sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration>` are recovered as special cases of $\pi_{\mathfrak{G}}$:

| Method                     | Governor Instantiation                                                       |
|----------------------------|------------------------------------------------------------------------------|
| Primal-Dual (3.5.A)        | $\pi_{\mathfrak{G}}(s_t) = \lambda_{t-1} + \eta_\lambda s_t$ (affine, $H=1$) |
| PID (3.5.B)                | Linear filter with fixed $(K_p, K_i, K_d)$, $H \geq 2$                       |
| Learned Precisions (3.5.C) | Diagonal, no temporal dependence, $H=0$                                      |

({prf:ref}`prop-structure-of-diagnostic-inputs`) *proposition* — **Structure of Diagnostic Inputs**

The input to the Governor, $s_t = \Psi(\theta_t)$, consists of quantities that depend only on the learned representations, not on the raw data $\mathcal{D}$:
- Entropies: $H(K)$, $H(Y|K)$, $I(K;X)$
- Spectral norms: $\|\nabla V\|$, $\lambda_{\max}(G)$
- Curvatures: $\|\nabla^2 V\|$, $R_{\text{Ric}}$

These are computed from the model's internal state $\theta_t$ and its outputs on training batches.

*Example:* Codebook collapse is diagnosed by $H(K) \to 0$. The correction (increase VQ commitment loss $\beta$) depends only on the diagnostic value, not on whether the data is images, audio, or tabular.

({prf:ref}`prop-transfer-via-meta-generalization`) *proposition* — **Transfer via Meta-Generalization**

Under the conditions of the Meta-Generalization Metatheorem (**MT: Meta-Generalization** in `metalearning.md`), the Governor $\pi_{\mathfrak{G}}$ trained on a distribution of optimization landscapes $\mathcal{S}$ generalizes to new systems drawn from $\mathcal{S}$.

Specifically, if:
1. **Compact structural manifold:** The optimal diagnostic-to-correction mappings $\{\phi^*(S) : S \in \text{supp}(\mathcal{S})\}$ lie on a compact $C^1$ submanifold of the policy space
2. **Uniform local strong convexity:** The training regret $J(\phi)$ satisfies $c\,\text{dist}(\phi, \mathcal{M})^2 \leq J(\phi) \leq C\,\text{dist}(\phi, \mathcal{M})^2$ near the optimal manifold
3. **Lipschitz continuity:** The regret is Lipschitz in both the policy parameters and the training landscape

Then, with probability at least $1 - \delta$, a Governor trained on $N$ sampled landscapes satisfies:

$$
\mathbb{E}_{S \sim \mathcal{S}}[J_S(\hat{\phi}_N)] \leq C_1\left(\varepsilon_N + \sqrt{\frac{\log(1/\delta)}{N}}\right)

$$
where $\varepsilon_N$ is the optimization accuracy.

In plain terms: if different training landscapes require similar corrections for similar diagnostic signatures, and the training distribution is diverse enough, the learned mapping transfers to new landscapes in the same structural class.

::::{warning} Caveat

The Meta-Generalization Metatheorem is proven in the unpublished document `metalearning.md`. While the proof follows standard statistical learning arguments (uniform convergence, Rademacher complexity bounds), the document has not undergone peer review. The assumptions (compactness, Lipschitz, strong convexity) must be verified for specific applications.
::::

({prf:ref}`prop-dimensional-analysis`) *proposition* — **Dimensional Analysis**

All inputs to $\pi_{\mathfrak{G}}$ are either:
1. **Dimensionless ratios:** $\nu_{\text{cap}} = I_{\text{bulk}}/C_\partial$
2. **Entropies:** measured in nats
3. **Normalized defects:** $(C_k - \epsilon_k)/\epsilon_k$

All outputs are either dimensionless (multipliers $\lambda_k$) or have standard units ($\eta$ in step$^{-1}$, $T_c$ in nat). This ensures the Governor's function approximator operates in a well-conditioned, scale-invariant regime.

({prf:ref}`prop-kernel-alternatives`) *proposition* — **Kernel Alternatives {cite}`rasmussen2006gp`**

Alternative kernels may be used depending on application requirements:

1. **Gaussian (RBF) Kernel:**

   $$
   K_{\text{Gauss}}(z, z') := \exp\left(-\frac{d_G(z, z')^2}{2\ell^2}\right),

   $$
   where $d_G$ is the geodesic distance and $\ell > 0$ is the length scale. This provides fast (exponential) decay, suitable for short-range memory effects.

2. **Matérn Kernel:**

   $$
   K_{\nu}(z, z') \propto (-\Delta_G + \kappa^2)^{-\nu}\delta(z - z'),

   $$
   where $\nu > 0$ is the smoothness parameter and $\kappa > 0$ is the inverse correlation length. For $\nu = 1$, this recovers the Green's function $G_\kappa$ from {ref}`Section 24.2 <sec-the-bulk-potential-screened-poisson-equation>`. The Matérn kernel has polynomial (rather than exponential) tails, providing longer-range correlations. See {cite}`rasmussen2006gp` Chapter 4 for the Euclidean case.

*Cross-reference:* The Matern kernel with $\nu = 1$ coincides with the screened Poisson Green's function (Definition {prf:ref}`prop-green-s-function-decay`), establishing a direct connection between memory effects and value propagation.

({prf:ref}`prop-mass-creation-from-experience`) *proposition* — **Mass Creation from Experience**

The memory contribution to the reaction term is:

$$
r_{\text{mem}}(z) := \frac{\rho(z)(\Psi_{\text{mem}}(z) - \bar{\Psi}_{\text{mem}})}{T_c},

$$
where $\bar{\Psi}_{\text{mem}} = \int_{\mathcal{Z}} \Psi_{\text{mem}} \rho \, d\mu_G$.

*Interpretation:* Belief mass is created where $\Psi_{\text{mem}} < \bar{\Psi}_{\text{mem}}$ (attractive memory) and destroyed where $\Psi_{\text{mem}} > \bar{\Psi}_{\text{mem}}$ (repulsive memory). This acts as a *virtual source* that redistributes probability toward remembered high-reward regions, even when local dynamics (via $\Phi_{\text{eff}}$) do not support such transitions.

({prf:ref}`prop-exponential-complexity-of-specificity`) *proposition* — **Exponential Complexity of Specificity**

The volume of a geodesic ball in the Poincare disk grows exponentially with radius:

$$
\text{Vol}(B_r(z)) \sim \sinh^{d-1}(r) \sim \frac{1}{2^{d-1}} e^{(d-1)r} \quad \text{as } r \to \infty.

$$

*Interpretation:* As the agent descends toward the boundary (increasing semantic specificity), the number of accessible knowledge atoms grows exponentially. This captures the combinatorial explosion of specific facts relative to abstract concepts---compare TopoEncoder hierarchy ({ref}`Section 25 <sec-supervised-topology-semantic-potentials-and-metric-segmentation>`).

({prf:ref}`prop-superposition-of-non-local-forces`) *proposition* — **Superposition of Non-Local Forces**

The total non-local force is:

$$
\mathbf{f}_{\text{non-local}} = -G^{-1}\nabla_G(\Psi_{\text{mem}} + \Psi_{\text{ret}}),

$$
where:
- Memory force $\mathbf{f}_{\text{mem}}$ integrates over the agent's past trajectory
- Retrieval force $\mathbf{f}_{\text{ret}}$ integrates over the external archive

*Interpretation:* The agent simultaneously experiences attraction to its own memory ({ref}`Section 27 <sec-section-non-local-memory-as-self-interaction-functional>`) and to relevant external knowledge (this section).

({prf:ref}`prop-non-causal-transition-via-retrieval`) *proposition* — **Non-Causal Transition via Retrieval**

Mass injection at retrieved locations enables transitions without continuous geodesic paths:

$$
\rho(z', s + \Delta s) > 0 \quad \text{even if} \quad d_G(z, z') > \sup_{0 \leq \tau \leq \Delta s} \|\mathbf{v}(z, s+\tau)\| \cdot \Delta s.

$$
*Interpretation:* Retrieval teleports probability mass to semantically relevant regions, bypassing the diffusion constraint. This is the WFR-level description of "jumping to a retrieved fact."

({prf:ref}`prop-optimal-nonlocal-coupling`) *proposition* — **Optimal Non-Local Coupling**

Let the control vector be $\Lambda = (\Lambda_{\text{mem}}, \Lambda_{\text{ret}})$. The optimal coupling is the fixed point of the Governor's policy $\pi_{\mathfrak{G}}$ ({prf:ref}`def-the-universal-governor`) given the diagnostic state $s_t = (\Delta_{\text{causal}}, \Omega_{\text{mem}})$.

**Control Law Derivation:**

1. **Surprise Signal:** Let $\Delta_{\text{causal}} = D_{\text{KL}}(P_{\text{int}} \| P_{\text{obs}})$ be the Interventional Gap (Node 53).

2. **Overfitting Signal:** Let $\Omega_{\text{mem}}$ be the Non-Locality Ratio ({prf:ref}`def-non-locality-ratio`, Node 43).

3. **Governor Update:** The Lyapunov descent condition $\Delta V_{\mathfrak{L}} < 0$ ({prf:ref}`def-training-lyapunov-function`) implies the following qualitative update dynamics:

$$
\begin{aligned}
\dot{\Lambda}_{\text{ret}} &\propto \alpha_1 \cdot \Delta_{\text{causal}} \\
\dot{\Lambda}_{\text{mem}} &\propto \alpha_2 \cdot (\Delta_{\text{causal}}^{\text{target}} - \Delta_{\text{causal}}) - \alpha_3 \cdot \operatorname{ReLU}(\Omega_{\text{mem}} - \Omega_{\max})
\end{aligned}

$$
where $\alpha_1, \alpha_2, \alpha_3 > 0$ are learning rates and $\Omega_{\max}$ is the maximum tolerable non-locality ratio.

({prf:ref}`prop-fixed-points-of-ontological-ricci-flow`) *proposition* — **Fixed Points of Ontological Ricci Flow**

The flow has fixed points when:
1. The capacity-constrained metric law is satisfied: $R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa T_{ij}$
2. The Ontological Stress has vanishing Hessian: $\nabla_i \nabla_j \Xi = 0$

Condition (2) is satisfied when either $\Xi$ is constant (uniform stress) or $\Xi = 0$ (no stress).

*Computational Proxy.* In practice, we do not solve the Ricci flow PDE. The squared residual of the fixed-point condition can be used as a regularization loss:

$$
\mathcal{L}_{\text{Ricci}} := \left\|R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} - \kappa T_{ij}\right\|_F^2 + \nu^2 \|\nabla_i \nabla_j \Xi\|_F^2,

$$
encouraging the learned metric to satisfy the capacity constraint while penalizing stress gradients.

({prf:ref}`prop-equipartition`) *proposition* — **Equipartition of Meaning**

At metabolic equilibrium, the marginal utility per bit is uniform across the ontological hierarchy:

$$
\frac{\partial U}{\partial H(K_{\text{chart}})} \approx \frac{\partial U}{\partial H(K_{\text{code}})} \approx \text{const.}

$$
where $U$ is the total utility functional (value minus complexity cost).

*Interpretation:* The agent allocates representational capacity such that one additional bit of chart-level information provides the same marginal value as one additional bit of symbol-level information. This is the information-theoretic analogue of thermodynamic equipartition.

({prf:ref}`prop-interaction-kernel`) *proposition* — **Interaction Kernel**

The **pairwise interaction potential** $\Phi_{\text{int}}: \mathcal{Z} \times \mathcal{Z} \to \mathbb{R}$ between agents at positions $z, \zeta$ is the screened Green's function weighted by influence:

$$
\Phi_{\text{int}}(z, \zeta) := \alpha \cdot \mathcal{G}_{\kappa}(z, \zeta)

$$
where $\mathcal{G}_{\kappa}$ is the screened Green's function (Proposition {prf:ref}`prop-green-s-function-interpretation`) and $\alpha$ encodes the strategic relationship.

*Properties:*
- $\Phi_{\text{int}}(z, \zeta) = \Phi_{\text{int}}(\zeta, z)$ (symmetric in cooperative settings)
- $\Phi_{\text{int}} \to 0$ as $d_G(z, \zeta) \to \infty$ (locality via screening)
- $\nabla^2_z \Phi_{\text{int}}$ defines the Game Tensor contribution (Definition {prf:ref}`def-the-game-tensor`)

({prf:ref}`prop-retarded-greens-function`) *proposition* — **Retarded Green's Function**

The solution to the inhomogeneous Klein-Gordon equation is given by convolution with the **Retarded Green's Function**:

$$
V^{(i)}(z, t) = \int_{-\infty}^{t} \int_{\mathcal{Z}^{(i)}} G_{\text{ret}}(z, t; \zeta, \tau) \left[ \rho^{(i)}_r(\zeta, \tau) + \sum_{j \neq i} \rho^{\text{ret}}_{ij}(\zeta, \tau) \right] d\mu_G(\zeta) \, d\tau,

$$
where $G_{\text{ret}}$ satisfies:

$$
\left( \frac{1}{c_{\text{info}}^2} \frac{\partial^2}{\partial t^2} - \Delta_G + \kappa^2 \right) G_{\text{ret}}(z, t; \zeta, \tau) = \delta(z - \zeta)\delta(t - \tau),

$$
with the **causal boundary condition** $G_{\text{ret}} = 0$ for $t < \tau$.

*Form in flat space:* For $\mathcal{Z} = \mathbb{R}^D$ with Euclidean metric:

$$
G_{\text{ret}}(z, t; \zeta, \tau) = \frac{\Theta(t - \tau)}{4\pi |z - \zeta|} \delta\left(t - \tau - \frac{|z-\zeta|}{c_{\text{info}}}\right) \cdot e^{-\kappa|z-\zeta|}.

$$

({prf:ref}`prop-retarded-metric-propagation`) *proposition* — **Retarded Metric Propagation**

The effective metric $\tilde{G}^{(i)}(z, t)$ satisfies a wave-like propagation equation:

$$
\frac{\partial \tilde{G}^{(i)}_{kl}}{\partial t} = \sum_{j \neq i} \beta_{ij} \frac{\partial \mathcal{G}_{ij,kl}^{\text{ret}}}{\partial t} = \sum_{j \neq i} \beta_{ij} \frac{d\mathcal{G}_{ij,kl}}{dt}\bigg|_{t-\tau_{ij}}.

$$

The metric perturbation at time $t$ depends on the opponent's dynamics at time $t - \tau_{ij}$. Information about strategic coupling propagates at speed $c_{\text{info}}$.

({prf:ref}`prop-gauge-transformation-connection`) *proposition* — **Gauge Transformation of the Connection**

Under a local gauge transformation $U(z) \in G$, the connection transforms as:

$$
A'_\mu = U A_\mu U^{-1} - \frac{i}{g}(\partial_\mu U)U^{-1}

$$

where $g > 0$ is the **coupling constant** (strategic coupling strength).

$$
\begin{aligned}
D'_\mu\psi' &= (\partial_\mu - igA'_\mu)(U\psi) \\
&= (\partial_\mu U)\psi + U(\partial_\mu\psi) - igA'_\mu U\psi
\end{aligned}

$$

For this to equal $U(\partial_\mu - igA_\mu)\psi = U(\partial_\mu\psi) - igUA_\mu\psi$, we require:

$$
(\partial_\mu U)\psi - igA'_\mu U\psi = -igUA_\mu\psi

$$

Solving for $A'_\mu$ yields the stated transformation law. $\square$

*Interpretation:* The inhomogeneous term $-\frac{i}{g}(\partial_\mu U)U^{-1}$ compensates for the "frame twist" introduced by position-dependent gauge transformations. The connection must counter-twist to maintain covariance.

({prf:ref}`prop-minimal-coupling`) *proposition* — **Minimal Coupling Principle**

To maintain gauge invariance, all derivatives in the dynamics must be replaced by covariant derivatives:

$$
\partial_\mu \longrightarrow D_\mu = \partial_\mu - igA_\mu

$$

This **Minimal Coupling Principle** ensures that:
1. The WFR continuity equation becomes gauge-covariant
2. The HJB equation becomes gauge-covariant
3. Learning gradients transform properly under internal rotations

*Consequence for implementation:* Any gradient-based update rule $\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$ must use the covariant gradient $D_\theta \mathcal{L}$ to maintain frame-independence.

({prf:ref}`prop-game-tensor-gauge-transformation`) *proposition* — **Game Tensor Gauge Transformation**

Under a local gauge transformation $U(z)$, the Game Tensor transforms as:

$$
\mathcal{G}'_{ij}(z) = U(z) \mathcal{G}_{ij}(z) U(z)^\dagger + \mathcal{C}_{ij}[A, U]

$$

where $\mathcal{C}_{ij}[A, U]$ is a **connection correction** involving commutators $[A_\mu, \mathcal{G}_{ij}]$.

For **Abelian** gauge groups ($[T_a, T_b] = 0$), the correction vanishes:

$$
\mathcal{G}'_{ij} = \mathcal{G}_{ij} \quad \text{(Abelian)}

$$

For **non-Abelian** groups, the Game Tensor is not gauge-invariant but transforms covariantly.

*Interpretation:* In non-Abelian settings, strategic coupling itself depends on the choice of internal frame. The "strength" of conflict between agents cannot be measured without specifying a gauge.

({prf:ref}`prop-field-strength-transformation`) *proposition* — **Covariant Transformation of Field Strength**

Under gauge transformation $U(z)$, the field strength transforms **covariantly** (not invariantly):

$$
\mathcal{F}'_{\mu\nu} = U \mathcal{F}_{\mu\nu} U^{-1}

$$

$$
\begin{aligned}
\mathcal{F}'_{\mu\nu} &= \partial_\mu A'_\nu - \partial_\nu A'_\mu - ig[A'_\mu, A'_\nu] \\
&= U(\partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu])U^{-1} \\
&= U\mathcal{F}_{\mu\nu}U^{-1}
\end{aligned}

$$

The inhomogeneous terms from $A'_\mu$ cancel exactly. $\square$

*Consequence:* While $\mathcal{F}_{\mu\nu}$ is not gauge-invariant, the trace $\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})$ **is** gauge-invariant and can appear in the action.

({prf:ref}`prop-gauge-energy-momentum`) *proposition* — **Gauge Field Energy-Momentum Tensor**

The energy-momentum tensor of the gauge field is:

$$
T^{\text{gauge}}_{\mu\nu} = -\frac{1}{g^2}\text{Tr}\left(\mathcal{F}_{\mu\rho}\mathcal{F}_\nu^{\ \rho} - \frac{1}{4}\tilde{G}_{\mu\nu}\mathcal{F}_{\rho\sigma}\mathcal{F}^{\rho\sigma}\right)

$$

*Properties:*
1. **Symmetric:** $T^{\text{gauge}}_{\mu\nu} = T^{\text{gauge}}_{\nu\mu}$
2. **Traceless** (for $D = 4$): $T^{\text{gauge}\mu}_{\ \ \ \ \mu} = 0$
3. **Conserved:** $\nabla_\mu T^{\text{gauge}\mu\nu} = 0$ (on-shell)

*Interpretation:* The gauge field carries energy and momentum. Regions of high strategic curvature $\|\mathcal{F}\|$ have high energy density—strategic conflict is energetically costly.

({prf:ref}`prop-laplace-beltrami-self-adjointness`) *proposition* — **Self-Adjointness of the Laplace-Beltrami Operator**

The Laplace-Beltrami operator

$$
\Delta_G := \frac{1}{\sqrt{|G|}} \partial_i \left( \sqrt{|G|} G^{ij} \partial_j \right)

$$
is essentially self-adjoint on $\mathcal{H} = L^2(\mathcal{Z}, d\mu_G)$ with domain $C_c^\infty(\mathcal{Z})$ (smooth functions with compact support), provided either:
1. $(\mathcal{Z}, G)$ is **geodesically complete**, or
2. $\mathcal{Z}$ has a boundary $\partial \mathcal{Z}$ with **Dirichlet conditions** $\psi|_{\partial \mathcal{Z}} = 0$ (sensors, Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`) or **Neumann conditions** $\nabla_n \psi|_{\partial \mathcal{Z}} = 0$ (motors, Definition {prf:ref}`def-neumann-boundary-condition-motors`).


*Consequence:* Self-adjointness guarantees that $-\Delta_G$ has a real spectrum bounded below, enabling spectral decomposition and ground state analysis.

({prf:ref}`prop-operator-ordering-invariance`) *proposition* — **Operator Ordering and Coordinate Invariance**

The kinetic term $-\frac{\sigma^2}{2}\Delta_G$ in the Inference Hamiltonian uses the unique **coordinate-invariant** ordering:

$$
-\frac{\sigma^2}{2}\Delta_G \psi = -\frac{\sigma^2}{2} \cdot \frac{1}{\sqrt{|G|}} \partial_i \left( \sqrt{|G|} G^{ij} \partial_j \psi \right).

$$
This is equivalent to:

$$
-\frac{\sigma^2}{2}\Delta_G = -\frac{\sigma^2}{2} \left( G^{ij} \partial_i \partial_j + \Gamma^k \partial_k \right),

$$
where $\Gamma^k := G^{ij}\Gamma^k_{ij}$ is the trace of Christoffel symbols.

**Alternative orderings** (Weyl, symmetric, etc.) would introduce frame-dependent terms that break the geometric interpretation.

*Cross-reference:* This matches the Laplace-Beltrami operator used in the Helmholtz equation (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`), ensuring consistency between the PDE and wave-function formulations.

({prf:ref}`prop-partial-trace-reduced-dynamics`) *proposition* — **Partial Trace and Reduced Dynamics**

For a pure joint state $|\Psi\rangle \in \mathcal{H}^{(N)}$, the **reduced density operator** for agent $i$ is:

$$
\hat{\rho}^{(i)} := \text{Tr}_{j \neq i}\left[ |\Psi\rangle\langle\Psi| \right] = \int_{\prod_{j \neq i} \mathcal{Z}^{(j)}} |\Psi|^2 \prod_{j \neq i} d\mu_{G^{(j)}}.

$$
The diagonal elements give the **marginal belief density**:

$$
\rho^{(i)}(z^{(i)}) = \langle z^{(i)} | \hat{\rho}^{(i)} | z^{(i)} \rangle = \int |\Psi(z^{(i)}, z^{(-i)})|^2 d\mu_{G^{(-i)}},

$$
which is exactly the marginalization from the joint WFR density.

**Mixed state evolution:** Even if $\Psi$ evolves unitarily, the reduced state $\hat{\rho}^{(i)}$ generally evolves **non-unitarily** (with decoherence) due to entanglement with other agents.

({prf:ref}`prop-imaginary-time-nash-finding`) *proposition* — **Imaginary Time Evolution for Nash Finding**

The substitution $s \to -i\tau$ (**Wick rotation**) transforms the Schrödinger equation into a diffusion equation:

$$
-\sigma \frac{\partial \Psi}{\partial \tau} = \hat{H}_{\text{strat}} \Psi.

$$
Under this **imaginary time evolution**, any initial state $\Psi_0$ converges to the ground state:

$$
\Psi(\tau) = e^{-\hat{H}_{\text{strat}} \tau / \sigma} \Psi_0 \xrightarrow{\tau \to \infty} c \cdot \Psi_{\text{Nash}},

$$
where $c$ is a normalization constant.

**Computational interpretation:**
1. Imaginary time evolution is equivalent to **Value Iteration** in dynamic programming
2. The propagator $e^{-\hat{H}\tau/\sigma}$ is the **Bellman backup operator** in infinite-horizon limit
3. Convergence rate is set by the **spectral gap** $E_1 - E_0$ (energy of first excited state minus ground state)

**Algorithm sketch (Quantum Value Iteration):**
```
Initialize Ψ randomly
For τ = 0 to T:
    Ψ ← exp(-H_strat Δτ / σ) Ψ    # Diffusion step
    Ψ ← Ψ / ||Ψ||                  # Renormalize
Return Ψ (approximates Nash ground state)
```

*Cross-reference:* This connects to the **imaginary-time path integral** formulation used in quantum Monte Carlo methods.

({prf:ref}`prop-wfr-reaction-tunneling`) *proposition* — **WFR Reaction as Tunneling Mechanism**

The WFR reaction term $r(z)$ (Definition {prf:ref}`def-the-wfr-action`) enables tunneling via **mass creation on the far side** of barriers:

1. Agent detects high-value region $\mathbf{z}^*_B$ beyond barrier $\mathcal{B}_P$
2. Reaction term $r(\mathbf{z}^*_B) > 0$ creates belief mass at $\mathbf{z}^*_B$
3. Reaction term $r(\mathbf{z}^*_A) < 0$ destroys mass at old position $\mathbf{z}^*_A$
4. Net effect: belief "teleports" without traversing $\mathcal{B}_P$

The rate of this process is controlled by the **teleportation length** $\lambda$ (Definition {prf:ref}`def-canonical-length-scale`):
- $\lambda \gg$ barrier width: tunneling is fast (reaction-dominated)
- $\lambda \ll$ barrier width: tunneling is slow (transport-dominated)

({prf:ref}`prop-a-area-minimal-distinguishable-cell`) *proposition* — **A.6.0f (Area of a Minimal Distinguishable Cell)**

On a 2-dimensional latent manifold with Fisher-compatible geometry (curvature $K = -1$), the Riemannian area of a cell containing exactly one nat of distinguishable information is:

$$
A_{\text{1 nat}} = 4\ell_L^2,

$$
where $\ell_L$ is the Levin Length.

**Step 1: Definition of $\ell_L$ (Implementation-Determined).** The Levin Length $\ell_L$ is the fundamental coordinate resolution of the computational manifold, determined by implementation constraints (discretization precision, floating-point resolution, etc.). This is analogous to how the Planck length $\ell_P = \sqrt{\hbar G/c^3}$ is determined by physical constants, not by the form of the area law.

**Step 2: Geodesic-to-Coordinate Relationship (From Fisher Metric).** On the Poincare disk with $K = -1$, the line element at the origin is:

$$
ds = 2 \, dx \quad \text{(from } ds^2 = 4(dx^2 + dy^2) \text{ at } z = 0\text{)}.

$$
A coordinate displacement $\ell_L$ corresponds to geodesic (Riemannian) distance $2\ell_L$.

**Step 3: Information-Geodesic Correspondence (From Chentsov).** By Theorem {prf:ref}`thm-a-chentsov-uniqueness`, the Fisher metric is the unique metric where KL divergence corresponds to squared geodesic distance (locally). Specifically, for nearby distributions $p$ and $q$:

$$
D_{\text{KL}}(p \| q) \approx \frac{1}{2} d_{\text{geo}}(p, q)^2.

$$
Thus, 1 nat of KL divergence corresponds to geodesic distance $\sqrt{2}$.

**Combining:** A coordinate cell of side $\ell_L$ has:
- Coordinate area: $\ell_L^2$
- Riemannian area: $\ell_L^2 \cdot \sqrt{\det G(0)} = \ell_L^2 \cdot 4 = 4\ell_L^2$
- Information capacity: proportional to Riemannian area $/$ (geodesic length per nat)$^2$

The factor of 4 emerges from the conformal factor $\sqrt{\det G(0)} = 4$, which was derived in Lemma {prf:ref}`lem-a-curvature-normalization-factor-4` from the curvature normalization $K = -1$, not from any assumption about information capacity. $\square$

*Remark (Non-Circularity).* In this derivation:
- $\ell_L$ is defined by implementation constraints (Step 1)
- The factor of 4 is derived from $K = -1$ (Lemma A.6.0e)
- The information-geometry correspondence is from Chentsov's theorem (Step 3)

No step assumes the form of the Area Law. Compare with Strominger-Vafa: they derive $S = A/(4\ell_P^2)$ by counting D-brane configurations, where $\ell_P$ is determined by string parameters and the 1/4 emerges from the counting.

({prf:ref}`prop-a-saturation-metric-solution`) *proposition* — **A.6.2 (Saturation Metric Solution)**

Under uniform saturation $T_{ij} = \sigma_{\max} G_{ij}$, the Metric Law reduces to:

$$
\frac{n-2}{r^2}\left(1 - \frac{1}{A(r)}\right) + \frac{n-2}{r} \cdot \frac{A'(r)}{A(r)^2} = \kappa \sigma_{\max} + \Lambda.

$$
The solution is:

$$
A(r) = \left( 1 - \frac{2\mu(r)}{(n-2)r^{n-2}} - \frac{\Lambda_{\text{eff}} r^2}{n(n-1)} \right)^{-1},

$$
where $\mu(r) = \frac{\kappa}{n-2} \int_0^r \sigma_{\max} r'^{n-1} dr'$ is the information mass function and $\Lambda_{\text{eff}} = \Lambda + \kappa \sigma_{\max}$.

1. Compute the Ricci tensor components for the ansatz
2. Substitute into the Metric Law
3. The radial component of the field equations gives a first-order ODE for $A(r)$
4. Integrate with boundary condition $A(0) = 1$ (regularity at origin)

The integration constant is determined by requiring $\lim_{r \to 0} A(r) = 1$. $\square$

({prf:ref}`prop-a-area-minimal-cell`) *proposition* — **A.6.5 (Area of a Minimal Information Cell)**

On a 2-dimensional Fisher manifold, the area of a cell corresponding to 1 nat of distinguishable information is:

$$
A_{\text{cell}} = 4 \ell_L^2.

$$

## Theorems

({prf:ref}`thm-generalized-conservation-of-belief`) *theorem* — **Generalized Conservation of Belief**

The evolution of the belief density $p$ satisfies the **Global Balance Equation**:

$$
\frac{d}{ds}\int_{\mathcal{Z}}p\,d\mu_G
=
-\oint_{\partial \mathcal{Z}} \langle p v,n\rangle\,dA_G
\;+\;
\int_{\mathcal{Z}} \sigma\,d\mu_G.

$$
where $n$ is the outward unit normal and $dA_G$ is the induced boundary area element. (Equivalently, if $\iota:\partial\mathcal{Z}\hookrightarrow \mathcal{Z}$ is the inclusion map, then the boundary flux is the pullback $\iota^*(p v\;\lrcorner\; d\mu_G)$.)

**The Architectural Sieve Condition (Node 13: BoundaryCheck).** The idealized "fully grounded" regime corresponds to $\sigma\approx 0$ in the interior: net changes in internal belief mass should be attributable to boundary influx and explicit projection events. Operationally we do not estimate $\sigma$ pointwise; instead Node 13 and the coupling-window diagnostics (Theorem {prf:ref}`thm-information-stability-window-operational`) enforce that the macro register remains coupled to boundary data (non-collapse of $I(X;K)$) and does not saturate ($H(K)$ stays below $\log|\mathcal{K}|$).

$$
\frac{d\mathcal{V}}{ds}
=
-\oint_{\partial \mathcal{Z}} \langle p v,n\rangle\,dA_G,

$$
in the case $\sigma\equiv 0$.

Here $\langle p v,n\rangle$ is the outward flux density across the boundary (negative values correspond to net inflow).

**Distinction: boundary-driven updates vs ungrounded updates**

1.  **Valid learning (boundary-driven):** The belief changes because there is non-negligible boundary flux, i.e. new observations justify updating the internal state.
2.  **Ungrounded update (internal source):** The belief changes despite negligible boundary flux, corresponding to $\sigma>0$ under the transport model. Operationally, this is a warning sign that internal rollouts are decoupled from the data stream and should be treated as unreliable for control until re-grounded.

({prf:ref}`thm-dynamical-isometry-without-skip-connections`) *theorem* — **Dynamical Isometry without Skip Connections**

A stacked TopoEncoder with:
1. OrthogonalLinear layers satisfying $\|W^T W - I\|_F < \epsilon_{\text{orth}}$,
2. Variance rescaling at each scale transition,
3. Spectral normalization with $\sigma_{\max}(W_\ell) \leq K$,

achieves approximate dynamical isometry: the singular values of the input-output Jacobian $J = \partial \hat{x} / \partial x$ satisfy $\sigma_i(J) \in [1/\kappa, \kappa]$ for a condition number $\kappa = O(K^L \cdot \prod_\ell (1 + \epsilon_{\text{orth}}))$.

({prf:ref}`thm-equivalence-of-entropy-regularized-control-forms-discrete-macro`) *theorem* — **Equivalence of Entropy-Regularized Control Forms; discrete macro**

Assume:
1. finite macro alphabet $\mathcal{K}$ and (for simplicity) finite action set $\mathcal{A}$,
2. an enclosure-consistent macro kernel $\bar{P}(k'\mid k,a)$,
3. bounded reward flux $\mathcal{R}(k,a)$.

Then the following are equivalent characterizations of the same optimal control law:

1. **MaxEnt control (utility + freedom):** $\pi^*$ maximizes $J_{T_c}(\pi)$ from Definition 10.2.1.
2. **Exponentially tilted trajectory measure (KL-regularization).** Fix a reference (prior) policy $\pi_0(a\mid k)$ with full support (uniform when $\mathcal{A}$ is finite). For the finite-horizon trajectory

   $$
   \omega := (A_t,\dots,A_{t+H-1},K_{t+1},\dots,K_{t+H}),

   $$
   the optimal controlled path law admits an exponential-family form relative to the reference measure induced by $\pi_0$ and $\bar{P}$:

   $$
   P^*(\omega\mid K_t=k)\ \propto\
   \Big[\prod_{h=0}^{H-1}\pi_0(A_{t+h}\mid K_{t+h})\,\bar{P}(K_{t+h+1}\mid K_{t+h},A_{t+h})\Big]\,
   \exp\!\left(\frac{1}{T_c}\sum_{h=0}^{H-1}\gamma^h\,\mathcal{R}(K_{t+h},A_{t+h})\right),

   $$
   where the normalizer is the (state-dependent) path-space normalizing constant.
3. **Soft Bellman optimality:** the optimal value function $V^*$ satisfies the soft Bellman recursion of Proposition 10.2.2, and $\pi^*$ is the corresponding softmax policy.

Moreover, the path-space log-normalizer is (up to scaling) the soft value. Gradients of the log-normalizer therefore induce a well-defined exploration direction in any differentiable macro coordinate system. The link between soft optimality and path entropy is cleanest when stated as a KL-regularized variational identity: if $P_0(\omega\mid k)$ denotes the reference trajectory measure induced by $\pi_0$ and $\bar{P}$, then

$$
\log Z(k)
=
\sup_{P(\cdot\mid k)}
\left\{
\frac{1}{T_c}\,\mathbb{E}_{P}\!\left[\sum_{h=0}^{H-1}\gamma^h\,\mathcal{R}\right]
-D_{\mathrm{KL}}(P(\cdot\mid k)\Vert P_0(\cdot\mid k))
\right\},

$$
and the optimizer is exactly the exponentially tilted law {math}`P^*`. In the special case where {math}`P_0` is uniform (or treated as constant), the KL term differs from Shannon path entropy by an additive constant, recovering the standard "maximize entropy subject to expected reward" view.

({prf:ref}`thm-information-stability-window-operational`) *theorem* — **Information-stability window; operational**

A necessary condition for stable, grounded macrostates is the existence of constants $0<\epsilon<\log|\mathcal{K}|$ such that, along typical trajectories,

$$
\epsilon \le I(X_t;K_t) \quad\text{and}\quad H(K_t)\le \log|\mathcal{K}|-\epsilon,

$$
and the net entropy balance satisfies

$$
\lambda_{\text{in}} \gtrsim \lambda_{\text{mix}}.

$$
Violations correspond to identifiable barrier modes:
- If $I(X;K)\approx 0$: under-coupling - ungrounded inference / decoupling (Mode D.C).
- If $H(K)\approx \log|\mathcal{K}|$: over-coupling or dispersion - symbol dispersion (BarrierScat).

*Remark.* This theorem is intentionally stated at the level of measurable information quantities (Gate Nodes) so it can be audited online; strengthening it to a sufficient condition requires specifying the macro kernel class and a contraction inequality (e.g. log-Sobolev / Doeblin-type conditions).

({prf:ref}`thm-capacity-constrained-metric-law`) *theorem* — **Capacity-constrained metric law**

Under the regularity and boundary-clamping hypotheses stated in {ref}`Appendix A <sec-appendix-a-full-derivations>`, and under the soundness condition that bulk structure is boundary-grounded (no internal source term $\sigma$ on $\operatorname{int}(\mathcal{Z})$; Definition {prf:ref}`def-source-residual`), stationarity of a capacity-constrained curvature functional implies

$$
R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij},

$$
where $\Lambda$ and $\kappa$ are constants and $T_{ij}$ is the **total Risk Tensor** induced by the reward field. *Units:* $\Lambda$ has the same units as curvature ($[R]\sim [z]^{-2}$), and $\kappa$ is chosen so that $\kappa\,T_{ij}$ matches those curvature units.

*Operational reading.* Curvature is the geometric mechanism that prevents the internal information volume (Definition 18.1.2a) from exceeding the boundary's information bandwidth (Definition {prf:ref}`def-a-bulk-information-volume`) while remaining grounded.

**Implementation hook.** The squared residual of this identity defines a capacity-consistency regularizer $\mathcal{L}_{\text{cap-metric}}$; see {ref}`Appendix B <sec-appendix-b-units-parameters-and-coefficients>` for the consolidated list of loss definitions and naming conventions.

({prf:ref}`thm-wfr-stress-energy-tensor-variational-form`) *theorem* — **WFR Stress-Energy Tensor; variational form**

Let the WFR action be

$$
\mathcal{S}_{\mathrm{WFR}}
=
\frac12\int_0^T\int_{\mathcal{Z}}
\rho\left(\|v\|_G^2+\lambda^2 r^2\right)\,d\mu_G\,ds,

$$
with continuity equation

$$
\partial_s\rho+\nabla\!\cdot(\rho v)=\rho r.

$$
Define

$$
T_{ij}:=
-\frac{2}{\sqrt{|G|}}\frac{\delta(\sqrt{|G|}\,\mathcal{L}_{\mathrm{WFR}})}{\delta G^{ij}}
\quad\text{(holding }\rho,v,r\text{ fixed).}

$$
Then

$$
T_{ij}=\rho\,v_i v_j + P\,G_{ij},
\qquad
P=\frac12\,\rho\left(\|v\|_G^2+\lambda^2 r^2\right),

$$
which is the perfect-fluid form with reaction contributing an additive pressure term
{math}`P_{\mathrm{react}}=\tfrac12\lambda^2\rho r^2`.

({prf:ref}`thm-unified-control-interpretation`) *theorem* — **Unified Control Interpretation**

The control field $u_\pi$ admits three equivalent interpretations:

| **Mode**                     | **Control Field $u_\pi$**                          | **Interpretation**                         |
|------------------------------|----------------------------------------------------|--------------------------------------------|
| **RL**                       | $u_\pi = G^{-1} \nabla_z V_{\text{critic}}$        | Points toward high-value regions           |
| **Conditioned Generation**   | $u_\pi = G^{-1} \cdot \text{embed}(\text{prompt})$ | Clamped to user's prompt embedding         |
| **Unconditional (Dreaming)** | $u_\pi = 0$                                        | Pure thermal fluctuation selects direction |

({prf:ref}`thm-pitchfork-bifurcation-structure`) *theorem* — **Pitchfork Bifurcation Structure {cite}`strogatz2015nonlinear`**

Near the origin, the combined dynamics exhibit a **supercritical pitchfork bifurcation**:

$$
\dot{r} = \mu r - r^3 + \sigma \xi

$$
where $r = |z|$, $\mu = 1$ (unstable fixed point), and $\sigma = \sqrt{2T_c}$ is the noise amplitude (see {prf:ref}`def-cognitive-temperature`).

**Phase Transition:**
- **Symmetric phase** ($T_c$ large): Random walk near origin, symmetry preserved
- **Broken phase** ($T_c$ small): Deterministic flow to boundary along selected direction

$$
dr = \left(\frac{1-r^2}{2} + u_\pi^r\right) d\tau + \sqrt{T_c(1-r^2)} dW_\tau

$$
where $u_\pi^r = u_\pi \cdot \hat{r}$ is the radial component of the control field. Taylor expanding near $r = 0$:

$$
dr \approx \left(\frac{1}{2} + u_\pi^r - \frac{r^2}{2}\right) d\tau + \sqrt{T_c}\, dW_\tau.

$$
For small control $u_\pi^r \ll 1$ and setting $\mu = 1/2 + u_\pi^r$, this matches the normal form $\dot{r} = \mu r - r^3/2 + \sigma\xi$.

**Critical temperature:** The effective potential $U_{\text{eff}}(r) = -\mu r^2/2 + r^4/8$ has minima at $r^* = \pm\sqrt{2\mu}$ for $\mu > 0$. The barrier height is $\Delta U = \mu^2/4$. Symmetry is preserved when thermal fluctuations overcome the barrier:

$$
T_c^* = \frac{\mu^2}{4} = \frac{1}{16}(1 + 2u_\pi^r)^2 \approx \frac{1}{16}.

$$
For $T_c > T_c^*$: symmetric phase; for $T_c < T_c^*$: broken phase with directional flow. $\square$

({prf:ref}`thm-overdamped-limit`) *theorem* — **Overdamped Limit**

Consider the second-order SDE from Definition {prf:ref}`def-bulk-drift-continuous-flow` with friction coefficient $\gamma$:

$$
m\,\ddot{z}^k + \gamma\,\dot{z}^k + G^{kj}\partial_j\Phi + \Gamma^k_{ij}\dot{z}^i\dot{z}^j = \sqrt{2T_c}\,\left(G^{-1/2}\right)^{kj}\,\xi^j,

$$
where $m$ is the "inertial mass" and $\xi$ is white noise. In the limit $\gamma \to \infty$ with $m$ fixed (or equivalently, $m \to 0$ with $\gamma$ fixed), the dynamics reduce to the first-order Langevin equation:

$$
dz^k = -G^{kj}(z)\,\partial_j\Phi_{\text{gen}}(z)\,ds + \sqrt{2T_c}\,\left(G^{-1/2}(z)\right)^{kj}\,dW^j_s.

$$

({prf:ref}`thm-atlas-duality-via-legendre-transform`) *theorem* — **Atlas Duality via Legendre Transform**

The Visual and Action Atlases are related by the Legendre transform $\mathcal{L}: T\mathcal{Q} \to T^*\mathcal{Q}$:

$$
\mathcal{A}_{\text{act}} = \mathcal{L}(\mathcal{A}_{\text{vis}}),

$$
where the chart transition functions satisfy:

$$
\psi_\beta \circ \mathcal{L} \circ \phi_\alpha^{-1} = \nabla_{\dot{q}} L(q, \dot{q})

$$
for Lagrangian $L(q, \dot{q}) = \frac{1}{2}\|\dot{q}\|_G^2 - V(q)$.

$$
\mathcal{L}: T\mathcal{Q} \to T^*\mathcal{Q}, \qquad (q, \dot{q}) \mapsto \left(q, \frac{\partial L}{\partial \dot{q}}\right).

$$
For $L = \frac{1}{2}\|\dot{q}\|_G^2 - V(q)$, this gives $p = G(q)\dot{q}$, which is invertible when $G > 0$.

**Step 2 (Symplectic preservation).** The Legendre transform is a diffeomorphism that pulls back the canonical symplectic form $\omega_{T^*\mathcal{Q}} = dp \wedge dq$ to the Poincare-Cartan form $\omega_{T\mathcal{Q}} = d\theta_L$ where $\theta_L = \frac{\partial L}{\partial \dot{q}^i}dq^i$. This ensures that Hamiltonian flow on $T^*\mathcal{Q}$ corresponds to Lagrangian flow on $T\mathcal{Q}$.

**Step 3 (Chart compatibility).** Let $(U_\alpha, \phi_\alpha)$ be a chart in $\mathcal{A}_{\text{vis}}$ with coordinates $(q^\alpha, \dot{q}^\alpha)$. Define the induced action chart $(V_\beta, \psi_\beta)$ by $V_\beta = \mathcal{L}(U_\alpha \times T_{U_\alpha}\mathcal{Q})$ with coordinates $(q^\alpha, p^\alpha)$. The transition function is:

$$
\psi_\beta \circ \mathcal{L} \circ \phi_\alpha^{-1}: (q^\alpha, \dot{q}^\alpha) \mapsto (q^\alpha, G_{\alpha\beta}(q)\dot{q}^\beta),

$$
which is smooth and invertible by positive-definiteness of $G$. $\square$

*Remark (Why Legendre?).* The Legendre transform is the unique smooth map relating configuration-velocity (perception) to configuration-momentum (action) that:
1. Preserves the symplectic structure (Proposition {prf:ref}`prop-symplectic-duality-principle`)
2. Interchanges Dirichlet and Neumann boundary conditions ({ref}`Section 23.1 <sec-the-symplectic-interface-position-momentum-duality>`)
3. Maps kinetic energy to Hamiltonian dynamics

*Cross-reference:* The metric $G$ appearing here is the capacity-constrained metric from Theorem {prf:ref}`thm-capacity-constrained-metric-law`, ensuring that the "mass" in the Legendre relation $p = G\dot{q}$ is the same "mass" that determines geodesic inertia (Definition {prf:ref}`def-mass-tensor`).

({prf:ref}`thm-perception-as-compression`) *theorem* — **Perception as Compression**

During perception, the agent compresses external entropy into internal free energy:

$$
W_{\text{compress}} = T_c \cdot I(X_t; K_t) \geq 0,

$$
where $T_c$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`) and $I(X_t; K_t)$ is the mutual information extracted from the observation $X_t$ into the macro-state $K_t$.

*Mechanism:* The Visual Encoder $E_\phi$ compresses high-entropy raw data $\phi_{\text{raw}}$ into a low-entropy macro-state $z$. The "heat" absorbed is the raw sensory stream.

*Information-theoretic interpretation:* Entropy decreases ($\Delta S < 0$). The Information Bottleneck cost bounds the compression.

({prf:ref}`thm-action-as-expansion`) *theorem* — **Action as Expansion**

During action, the agent expands internal free energy into external control:

$$
W_{\text{expand}} = T_c \cdot I(A_t; K_t) \geq 0,

$$
where $I(A_t; K_t)$ is the mutual information injected from the intention into the motor output.

*Mechanism:* The Action Decoder $D_A$ "expands" the low-entropy Intention $u_\pi$ into high-dimensional motor commands $a_{\text{raw}}$, injecting motor texture.

*Information-theoretic interpretation:* Entropy increases ($\Delta S > 0$). The agent injects stochastic texture into motor outputs.

({prf:ref}`thm-wfr-mode-switching`) *theorem* — **WFR Mode Switching**

The transition from waking to dreaming corresponds to a **boundary condition phase transition**:

| Mode         | Sensory BC                             | Motor BC             | Internal Flow | Information Balance       |
|--------------|----------------------------------------|----------------------|---------------|---------------------------|
| **Waking**   | Dirichlet ($\delta$-clamp)             | Neumann (flux-clamp) | Source-driven | $\oint j_{\text{in}} > 0$ |
| **Dreaming** | Reflective ($\nabla \rho \cdot n = 0$) | Reflective           | Recirculating | $\oint j = 0$             |

({prf:ref}`thm-universal-context-structure`) *theorem* — **Universal Context Structure**

All context instantiations share the same geometric structure:

1. **Embedding:** $c \mapsto e_c \in \mathbb{R}^{d_c}$ maps the context to a latent vector
2. **Symmetry-Breaking Kick:** $e_c$ determines the initial control field:

   $$
   u_\pi(0) = G^{-1}(0) \cdot e_c = \frac{1}{4} e_c

   $$
   (at the Poincare disk origin where $G(0) = 4I$)
3. **Motor Distribution:** The output distribution is:

   $$
   \pi(a | z, c) = \text{softmax}\left(-\frac{\Phi_{\text{eff}}(z, K, c)}{T_c}\right)

   $$

({prf:ref}`thm-hodge-decomposition`) *theorem* — **Hodge Decomposition of the Reward Field**

On the compact latent Riemannian manifold $(\mathcal{Z}, G)$, the Reward 1-form $\mathcal{R}$ uniquely decomposes into:

$$
\mathcal{R} = \underbrace{d\Phi}_{\text{Gradient}} + \underbrace{\delta \Psi}_{\text{Solenoidal}} + \underbrace{\eta}_{\text{Harmonic}}

$$
where:
1. **$\Phi \in \Omega^0(\mathcal{Z})$** (Scalar Potential): The conservative/optimizable component. $d\Phi$ is an exact form.
2. **$\Psi \in \Omega^2(\mathcal{Z})$** (Vector Potential): The rotational/cyclic component. $\delta\Psi$ is a coexact form (divergence-free).
3. **$\eta \in \mathcal{H}^1(\mathcal{Z})$** (Harmonic Flux): Topological cycles from manifold holes. Satisfies $d\eta = 0$ and $\delta\eta = 0$.

*Units:* $[\Phi] = \mathrm{nat}$, $[\Psi] = \mathrm{nat} \cdot [\text{length}]^2$, $[\eta] = \mathrm{nat}/[\text{length}]$.

({prf:ref}`thm-the-hjb-helmholtz-correspondence`) *theorem* — **The HJB-Helmholtz Correspondence {cite}`bellman1957dynamic,evans2010pde`**

Let the discount factor be $\gamma = e^{-\kappa \Delta t}$ where $\kappa > 0$ is the **screening mass**. The Bellman condition

$$
V(z) = \mathbb{E}[r + \gamma V(z')]

$$
approaches the following PDE in the limit $\Delta t \to 0$:

$$
\boxed{-\Delta_G V(z) + \kappa^2 V(z) = \rho_r(z)}

$$
where:
- $\Delta_G = \frac{1}{\sqrt{|G|}} \partial_i \left( \sqrt{|G|} G^{ij} \partial_j \right)$ is the **Laplace-Beltrami operator** on the manifold $(\mathcal{Z}, G)$
- $\kappa^2$ is the "mass" of the scalar field, causing the influence of distant rewards to decay exponentially
- $\rho_r(z)$ is the internal reward density plus propagated boundary conditions

$$
V(z) = r \Delta t + \gamma \mathbb{E}[V(z')] \approx r \Delta t + (1 - \kappa \Delta t)\left(V + \nabla V \cdot b \Delta t + T_c \Delta_G V \Delta t\right).

$$
Rearranging and dividing by $\Delta t$, then taking $\Delta t \to 0$:

$$
\kappa V = r + \nabla V \cdot b + T_c \Delta_G V.

$$
For the stationary case ($b = 0$) and absorbing the temperature into the source term, this yields the Helmholtz equation $-\Delta_G V + \kappa^2 V = \rho_r$. Details in {ref}`Appendix A.5 <sec-appendix-a-full-derivations>`. $\square$

Units: $[\kappa] = 1/\text{length}$, $[\Delta_G V] = \mathrm{nat}/\text{length}^2$, $[\rho_r] = \mathrm{nat}/\text{length}^2$.

*Cross-reference (Relativistic Extension):* This **elliptic** Helmholtz equation assumes instantaneous value propagation. When agents interact across spatial or computational separation with finite information speed $c_{\text{info}}$, the equation generalizes to the **hyperbolic Klein-Gordon equation**: $(\frac{1}{c^2}\partial_t^2 - \Delta_G + \kappa^2)V = \rho_r$. See Theorem {prf:ref}`thm-hjb-klein-gordon` in {ref}`Section 29.5 <sec-the-hyperbolic-value-equation>`.

*Cross-reference (Gauge-Covariant Generalization):* When the dynamics must be invariant under local nuisance transformations ({ref}`Section 29.13 <sec-local-gauge-symmetry-nuisance-bundle>`), all partial derivatives $\partial_\mu$ are promoted to covariant derivatives $D_\mu = \partial_\mu - igA_\mu$, where $A_\mu$ is the Strategic Connection (Definition {prf:ref}`def-strategic-connection`). The Helmholtz operator becomes $-D_\mu D^\mu + \kappa^2$.

({prf:ref}`thm-wfr-consistency-value-creates-mass`) *theorem* — **WFR Consistency: Value Creates Mass**

In the WFR dynamics ({prf:ref}`def-the-wfr-action`, {ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`), the reaction rate $r(z)$ in the unbalanced continuity equation is determined by the value function:

$$
r(z) = \frac{1}{s_r} \left( V(z) - \bar{V} \right),

$$
where $\bar{V} = \mathbb{E}_\rho[V]$ is the mean value and $s_r$ is the reaction time scale (computation time).

*Consequence:* The mass evolution satisfies:

$$
\dot{m}(s) = m(s) \cdot r(z(s)) \propto m(s) \cdot (V(z(s)) - \bar{V}).

$$

Probability density increases in regions where $V > \bar{V}$ and decreases where $V < \bar{V}$.

({prf:ref}`thm-ness-existence`) *theorem* — **Non-Equilibrium Steady State (NESS)**

**Non-Conservative Case ($\mathcal{F} \neq 0$):** If the Value Curl does not vanish, the stationary distribution $\rho_\infty$ is a **Non-Equilibrium Steady State** satisfying:

1. **Stationarity:** $\partial_s \rho_\infty = 0$
2. **Persistent Current:** The probability current $J = \rho v - D\nabla\rho$ is non-zero and divergence-free: $\nabla \cdot J = 0$ but $J \neq 0$
3. **Entropy Production:** The system continually produces entropy at rate:

$$
\dot{S}_i = \int_{\mathcal{Z}} \frac{\|J\|_G^2}{\rho D} \, d\mu_G > 0

$$

*Remark.* The probability density $\rho_\infty$ is time-independent, but individual trajectories circulate indefinitely. This distinguishes NESS from true equilibrium (where $J = 0$).

({prf:ref}`thm-rl-as-electrodynamics-on-a-curved-manifold`) *theorem* — **RL as Electrodynamics on a Curved Manifold**

The complete agent dynamics can be summarized as follows:

The agent is a **particle** with:
- **Position** $q \in \mathcal{Z}$ (from Perception / Dirichlet BC)
- **Momentum** $p \in T_q\mathcal{Z}$ (from Action / Neumann BC)
- **Mass** $G(q)$ (the Riemannian metric = information geometry)
- **Potential Energy** $V(q)$ (from Reward / Poisson source)
- **External Forces** $u_\pi(q)$ (from Policy / symmetry-breaking kick)

moving according to the **geodesic SDE** (Definition {prf:ref}`def-bulk-drift-continuous-flow`):

$$
dq^k = G^{kj}(q) p_j \, ds, \qquad
dp_k = -\frac{\partial V}{\partial q^k} ds - \frac{1}{2}\frac{\partial G^{ij}}{\partial q^k} p_i p_j \, ds + u_{\pi,k} \, ds + \sqrt{2T_c} \, (G^{1/2})_{kj} dW^j_s,

$$
on a **curved manifold** with metric $G$ satisfying the **capacity constraint** (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), in a **screened potential** $V$ satisfying the **Helmholtz equation** (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`). The term $\frac{1}{2}\frac{\partial G^{ij}}{\partial q^k} p_i p_j$ encodes the geodesic correction (equivalent to Christoffel symbols in the position formulation).

*Conclusion:* **Reinforcement Learning is Electrodynamics on a Curved Manifold.** The standard RL components (encoder, critic, policy) are revealed to be the components of a field theory:

| RL Component      | Field Theory Role                                         |
|-------------------|-----------------------------------------------------------|
| Encoder           | **Coordinate Chart** (embedding from boundary to bulk)    |
| Critic            | **Field Solver** (Green's function of screened Laplacian) |
| Policy            | **External Force** (symmetry-breaking current)            |
| Discount $\gamma$ | **Screening Mass** (controls correlation length)          |
| Temperature $T_c$ | **Thermal Bath** (fluctuation-dissipation source)         |

({prf:ref}`thm-causal-information-bound`) *theorem* — **The Causal Information Bound**

For a $D$-dimensional latent manifold $(\mathcal{Z}, G)$, the maximum information $I_{\max}$ that can be stably represented without the metric becoming singular is:

$$
\boxed{I_{\max} = \nu_D \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}}}

$$

where:
- $\text{Area}(\partial\mathcal{Z}) = \oint_{\partial\mathcal{Z}} dA_G$ is the $(D-1)$-dimensional boundary measure in the induced metric
- $\ell_L$ is the Levin Length (Definition {prf:ref}`def-levin-length`)
- $\nu_D$ is the Holographic Coefficient (Definition {prf:ref}`def-holographic-coefficient`)

*Corollary (Poincare disk, $D=2$).* For the 2-dimensional Poincare disk, the formula reduces to the Bekenstein-Hawking form:

$$
I_{\max} = \frac{\text{Area}(\partial\mathcal{Z})}{4\ell_L^2}

$$
where the $\ell_L^2$ (rather than $\ell_L^{D-1} = \ell_L$) arises from the Poincare disk metric normalization $G(0) = 4I$, which maps a coordinate cell of side $\ell_L$ to Riemannian area $4\ell_L^2$.

**Step 1 (Holographic Reduction).** The bulk-to-boundary conversion relies on the Einstein tensor divergence identity (valid in arbitrary dimension): integrating the scalar curvature over a compact manifold with boundary yields a boundary term involving the extrinsic curvature. Applying this to the Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) via Lemma {prf:ref}`lem-a-divergence-to-boundary-conversion`:

$$
I_{\text{bulk}} = \int_{\mathcal{Z}} \rho_I \, d\mu_G = \frac{1}{\kappa} \oint_{\partial\mathcal{Z}} \text{Tr}(K) \, dA_G,

$$
where $K$ is the extrinsic curvature of the boundary and $\kappa$ is the coupling constant from the Metric Law.

**Step 2 (Saturation Geometry).** At the saturation limit, the extrinsic curvature approaches $\text{Tr}(K) \to (D-1)/r_h$ where $r_h$ is the horizon radius from Lemma {prf:ref}`lem-metric-divergence-at-saturation`. The boundary area is $\text{Area}(\partial\mathcal{Z}) = \Omega_{D-1} r_h^{D-1}$ where $\Omega_{D-1}$ is the unit sphere surface area.

**Step 3 (Fisher Normalization).** The coupling constant $\kappa = 8\pi\ell_L^{D-1}$ is fixed by consistency with the Fisher Information Metric {cite}`amari2016information`. The dimension-dependent coefficient $\nu_D = (D-1)\Omega_{D-1}/(8\pi)$ emerges from the geometric factors.

Combining these steps yields the general bound. $\square$

*Operational interpretation.* The agent's "intelligence" (measured in grounded bits) is geometrically constrained by the size of its interface. To represent more information, you must either:
1. **Expand the boundary** (increase interface bandwidth), or
2. **Reduce the Levin Length** (improve resolution per unit area).

There is no third option. Adding internal parameters without expanding the interface yields diminishing returns as the agent approaches saturation.

*Remark (Dimensional efficiency).* High-dimensional latent spaces ($D \gg 1$) have $\nu_D \to 0$, meaning **less** information can be stored per unit boundary area. This provides a first-principles derivation of the "curse of dimensionality" and suggests that $D \approx 3$ is optimal for holographic efficiency.

({prf:ref}`thm-causal-stasis`) *theorem* — **Causal Stasis**

Let $v^k = dz^k/ds$ be the velocity of the agent's belief update in computation time $s$ (Definition {prf:ref}`def-bulk-drift-continuous-flow`). As $I_{\text{bulk}} \to I_{\max}$:

$$
\|v\|_G \to 0.

$$

$$
dz^k = \left( -G^{kj}\partial_j \Phi_{\text{eff}} + u_\pi^k - \Gamma^k_{ij}\dot{z}^i\dot{z}^j \right) ds + \sqrt{2T_c}(G^{-1/2})^{kj} dW^j_s.

$$
The drift velocity scales as:

$$
v^k \propto G^{kj} \partial_j \Phi_{\text{eff}}.

$$
As the information density approaches saturation, Lemma {prf:ref}`lem-metric-divergence-at-saturation` implies $G_{rr} \to \infty$, hence $G^{rr} \to 0$. The radial component of velocity:

$$
v^r = -G^{rr}\partial_r \Phi_{\text{eff}} \to 0. \quad \blacksquare

$$
*Operational interpretation.* The agent becomes **frozen in thought**. Its internal update rate slows as the "inertia" (mass = metric, per Definition {prf:ref}`def-mass-tensor`) becomes infinite. The agent can still receive observations (inflow), but it cannot process them into updated beliefs or emit actions (outflow). This is **Causal Stasis**: the agent is overwhelmed by its own representational complexity.

*Remark (Distinction from Deadlock).* Causal Stasis is not a software deadlock or resource exhaustion. It is a geometric phenomenon: the agent's belief manifold has curved so severely that motion becomes infinitely costly. The remedy is not debugging but **ontological surgery**—reducing $I_{\text{bulk}}$ via Fusion ({ref}`Section 30.8 <sec-ontological-fusion-concept-consolidation>`) or expanding the boundary capacity.

({prf:ref}`thm-classification-as-relaxation`) *theorem* — **Classification as Relaxation**

Under the overdamped dynamics ({ref}`Section 22.5 <sec-the-overdamped-limit>`) with potential $V_y$:

$$
dz = -G^{-1}(z) \nabla V_y(z, K)\, ds + \sqrt{2T_c}\, G^{-1/2}(z)\, dW_s, \quad T_c \text{ cognitive temperature } ({prf:ref}`def-cognitive-temperature`)

$$
the limiting chart assignment satisfies:

$$
\lim_{s \to \infty} K(z(s)) \in \mathcal{A}_y \quad \text{almost surely},

$$
provided:
1. $z(0) \in \mathcal{B}_y$ (initial condition in the basin)
2. $T_c$ is sufficiently small (low temperature limit)
3. The basins have positive measure and are separated by finite barriers

$$
\frac{dL}{ds} = \nabla V_y \cdot \dot{z} = -\|\nabla V_y\|_G^2 + \text{noise terms}.

$$
For small $T_c$, the deterministic term dominates, ensuring $L$ decreases until $z$ reaches a local minimum. The class-$y$ region is the global minimum of $V_y$ by construction. Full proof in {ref}`Appendix A.5 <sec-appendix-a-full-derivations>`. $\square$

({prf:ref}`thm-bilevel-structure`) *theorem* — **Bilevel Structure**

The training of the Universal Governor has bilevel structure:

$$
\min_\phi \; J(\phi) \quad \text{s.t.} \quad \theta_t = \theta_t(\Lambda_{0:t-1}), \quad \Lambda_t = \pi_{\mathfrak{G}}(s_{t:t-H}; \phi).

$$
The inner problem (agent learning) depends on the outer variables (Governor parameters) through the control sequence $\{\Lambda_t\}$.

*Remark (Gradient Computation).* Computing $\nabla_\phi J$ requires differentiating through the entire training trajectory. In practice, we use truncated backpropagation through time or evolutionary strategies.

**Cross-references:** {ref}`Section 3.4 <sec-joint-optimization>` (Joint Optimization).

({prf:ref}`thm-stable-training-trajectory`) *theorem* — **Stable Training Trajectory**

If the Governor $\pi_{\mathfrak{G}}$ selects $\Lambda_t$ such that:

$$
\Delta V_{\mathfrak{L}} := V_{\mathfrak{L}}(\theta_{t+1}) - V_{\mathfrak{L}}(\theta_t) < 0 \quad \forall t \text{ where } \theta_t \notin \Omega,

$$
then the training process converges to the largest invariant set $\Omega$ where $\Delta V_{\mathfrak{L}} = 0$. Under standard regularity (twice-differentiable $\mathcal{L}$, LICQ), $\Omega$ consists of KKT points.

({prf:ref}`thm-non-markovian-nature-of-memory`) *theorem* — **Non-Markovian Nature of Memory**

The force field $-\nabla_G \Psi_{\text{mem}}$ violates the Markov property.


*Remark (State Augmentation):* The non-Markovian character is essential for capturing genuine memory effects. The system state must be *augmented* to include $\Xi_T$ (or a sufficient statistic thereof) to recover a Markovian description in an extended state space.

*Remark (Computational Complexity):* Naively, evaluating $\Psi_{\text{mem}}(z)$ requires $O(T)$ kernel evaluations where $T$ is the trajectory length. For long histories, approximations are necessary: (i) truncate to recent history, (ii) subsample the trajectory, (iii) use inducing points {cite}`rasmussen2006gp`, or (iv) maintain a running kernel density estimate.

({prf:ref}`thm-memory-induced-barrier-crossing`) *theorem* — **Memory-Induced Barrier Crossing**

Let $z_t$ be the current position and suppose there exists a past time $t^* < t$ with $z^* := \gamma(t^*)$ such that:
1. $d_G(z_t, z^*) < \ell_{\text{mem}}$ for some memory influence radius $\ell_{\text{mem}}$,
2. $|\alpha(t^*)|$ is large (strong reward signal at time $t^*$).

Then the memory gradient $\|\nabla_G \Psi_{\text{mem}}\|_G$ can exceed the local barrier gradient $\|\nabla_G \Phi_{\text{eff}}\|_G$, enabling transitions that would be forbidden under purely local dynamics.

$$
\|\nabla_G \Psi_{\text{mem}}(z_t)\|_G \approx |\alpha(t^*)| \cdot \|\nabla_G H_\tau(z_t, z^*)\|_G.

$$
For $d_G(z_t, z^*) \sim O(\sqrt{\tau})$, the gradient $\|\nabla_G H_\tau\|_G \sim O(\tau^{-(d+1)/2})$ can be made arbitrarily large by choosing small $\tau$. If $|\alpha(t^*)|$ is sufficiently large, this dominates $\|\nabla_G \Phi_{\text{eff}}\|_G$. $\square$

*Cross-reference:* BarrierGap diagnostic ({ref}`Section 4 <sec-4-limits-barriers-the-limits-of-control>`).

*Interpretation:* Strong memories can "pull" the agent across local energy barriers, providing a mechanism for experience-guided exploration that transcends gradient-based planning.

({prf:ref}`thm-stability-of-retrieval-loop`) *theorem* — **Stability of Retrieval Loop**

Under the firewall constraint (Definition {prf:ref}`def-bulk-filtered-retrieval-potential`), the retrieval force field:

$$
\mathbf{f}_{\text{ret}} = -G^{-1}\nabla_G \Psi_{\text{ret}}

$$
is smooth (Lipschitz in $z$) and independent of external texture coordinates $z_{\text{tex,ext}}$.

*Consequence:* The control loop remains stable; external texture cannot inject high-frequency gradients that would trigger Mode T.C (Labyrinthine Overfitting).


*Cross-reference:* This theorem extends TextureFirewallCheck (Node 29) to external retrieval. See {ref}`Section 5 <sec-failure-modes>` for Mode T.C classification.

**Heuristic 28.3.4 (Side-Channel Texture Delivery).**
External texture $z_{\text{tex,ext}}$ is delivered to the decoder via a side channel:
1. At stopping radius $R_{\text{cutoff}}$ ({ref}`Section 21.3 <sec-the-retrieval-texture-firewall>`), retrieve the full atom $\xi = (K, z_n, z_{\text{tex}})$
2. Inject $z_{\text{tex}}$ directly to decoder attention, bypassing the EoM
3. The control loop only sees $(K, z_n)$

*Interpretation:* This is the retrieval analog of "reading a document without letting its style affect your reasoning."

({prf:ref}`thm-safe-retrieval-bandwidth`) *theorem* — **Safe Retrieval Bandwidth**

Let $\sigma_{\text{ret}}(z)$ be the retrieval source term in the WFR continuity equation ({prf:ref}`def-retrieval-source-term`). The latent geometry remains non-singular if and only if the total information flux satisfies:

$$
\int_{\mathcal{Z}} \left( \rho_I(z) + \sigma_{\text{ret}}(z) \right) \, d\mu_G \leq \kappa \, C_{\partial}

$$
where $C_{\partial} = \nu_D \cdot \text{Area}(\partial\mathcal{Z})/\ell_L^{D-1}$ is the boundary capacity (Definition {prf:ref}`def-holographic-coefficient`, {prf:ref}`def-levin-length`).

2. **Metric Response:** By the Capacity-Constrained Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), the radial metric component scales as $G_{rr} \propto (1 - \tilde{I}_{\text{bulk}}/C_{\partial})^{-1}$.

3. **Singularity:** If $\int \sigma_{\text{ret}} > C_{\partial} - I_{\text{bulk}}$, then $G_{rr} \to \infty$ at a radius $r < 1$ (the horizon moves inward).

4. **Dynamical Consequence:** The update velocity $\|v\|_G \to 0$ (Causal Stasis, {ref}`Section 33 <sec-causal-information-bound>`). The instability manifests as the freezing of the agent's inference dynamics due to saturation of the holographic bound. $\square$

({prf:ref}`thm-causal-isometry`) *theorem* — **Causal Isometry Theorem**

Let $\mathcal{M}_A$ and $\mathcal{M}_B$ be latent manifolds encoding modalities $A$ and $B$ of a common environment $\mathcal{E}$. Let $\Phi_{\text{causal}}$ be the Causal Information Potential ({ref}`Section 32 <sec-causal-discovery-interventional-geometry-and-the-singularity-of-action>`). If both representations are **Interventionally Closed** ({prf:ref}`thm-interventional-closure`), then the induced metrics $G_A$ and $G_B$ are isometric.

2. **Risk Invariance:** The risk Lagrangian $\mathcal{L}_{\text{risk}}(V) = \frac{1}{2}\|\nabla V\|^2 + U(V)$ depends only on the Value function $V$ and the Causal Potential $\Psi_{\text{causal}}$.

3. **Task Invariance:** The potentials $V$ and $\Psi_{\text{causal}}$ are functions of the *causal graph* of the environment $\mathcal{E}$, which is an invariant independent of the sensory modality (pixels vs. tokens).

4. **Uniqueness:** Assuming the solution to the metric field equation is unique (guaranteed for the Poincare disk ansatz in the saturation limit), the geometries $G_A$ and $G_B$ are identical up to a diffeomorphism determined by the encoder parameterization. $\square$

({prf:ref}`thm-vacuum-concentration-under-unknown-unknowns`) *theorem* — **Vacuum Concentration Under Unknown Unknowns**

Let $\mathcal{F}[p, \pi]$ be the entropy-regularized objective (Definition {prf:ref}`def-entropy-regularized-objective-functional`):

$$
\mathcal{F}[p, \pi] = \int_{\mathcal{Z}} p(z) \Big( V(z) - \tau H(\pi(\cdot|z)) \Big) d\mu_G.

$$
If the value function $V$ is **uninformative** in a region $\Omega \subset \mathcal{Z}$ -- i.e., $\nabla V|_\Omega \approx 0$ and $\nabla^2 V|_\Omega \approx 0$ -- then the entropy term dominates and the optimal belief concentrates toward maximum-entropy configurations:

$$
p^*(z) \propto \exp\left(-\frac{V(z)}{\tau}\right) \xrightarrow{\nabla V \to 0} \text{uniform on } \Omega.

$$
In the Poincare disk geometry, the maximum-entropy state is the vacuum $z = 0$.


*Interpretation.* When encountering observations outside the learned structure, the MaxEnt policy concentrates at the vacuum, correctly representing maximum uncertainty.

*Remark (Capacity Tension).* If belief mass accumulates at the vacuum such that bulk information $I_{\mathrm{bulk}}$ approaches the boundary capacity $C_\partial$ (the Capacity-Constrained Metric Law, Theorem {prf:ref}`thm-capacity-constrained-metric-law`), the current chart structure is insufficient. This tension -- high information density at a single point -- indicates fission is required to distribute the representational load.

({prf:ref}`thm-fission-criterion`) *theorem* — **Fission Criterion**

Let $\Xi$ be the Ontological Stress (Definition {prf:ref}`def-ontological-stress`) and let $\Xi_{\text{crit}} > 0$ be a threshold. Let $\Delta V_{\text{proj}}$ be the projected value improvement from splitting the highest-stress chart. The fission criterion is:

$$
\text{Fission} \iff \Xi > \Xi_{\text{crit}} \quad \text{AND} \quad \Delta V_{\text{proj}} > \mathcal{C}_{\text{complexity}}.

$$
*Units:* All quantities are in nats. The complexity cost $\mathcal{C}_{\text{complexity}}(N_c \to N_c + 1)$ includes the entropy increase $\log((N_c+1)/N_c)$ from the expanded codebook plus any regularization penalty on parameter count.

({prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`) *theorem* — **Supercritical Pitchfork Bifurcation for Charts**

The query fission dynamics exhibit the **supercritical pitchfork bifurcation** structure of Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`. Let $r := \|q_i^+ - q_i^-\|/2 = \epsilon$ be the half-separation of daughter queries. The radial evolution satisfies:

$$
\frac{dr}{ds} = (\Xi - \Xi_{\text{crit}}) r - \alpha r^3 + \sigma\xi,

$$
where:
- $\Xi - \Xi_{\text{crit}}$ plays the role of the bifurcation parameter $\mu$ in Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`
- $\alpha > 0$ is a stabilizing cubic coefficient (from competition for training data)
- $\sigma\xi$ is noise from stochastic gradient updates
- $s$ is the training step (flow time)

**Phase Transition:**
1. **Sub-critical ($\Xi < \Xi_{\text{crit}}$):** $r=0$ is the unique stable fixed point. The daughters collapse back to the parent ($r \to 0$).
2. **Super-critical ($\Xi > \Xi_{\text{crit}}$):** $r=0$ becomes unstable. The daughters separate toward a new equilibrium:

   $$
   r^* = \sqrt{\frac{\Xi - \Xi_{\text{crit}}}{\alpha}}.

   $$

$$
\Phi_{\text{fission}}(r) = -\frac{(\Xi - \Xi_{\text{crit}})}{2} r^2 + \frac{\alpha}{4} r^4,

$$
which has the standard pitchfork form. For $\Xi > \Xi_{\text{crit}}$, the origin has $\Phi_{\text{fission}}''(0) = -(\Xi - \Xi_{\text{crit}}) < 0$, becoming unstable. Stable minima appear at $r = \pm r^*$. The cubic term arises from router saturation: as daughters separate, they compete for data, and the loss landscape penalizes excessive separation. This matches the normal form of Theorem {prf:ref}`thm-pitchfork-bifurcation-structure` with $\mu = \Xi - \Xi_{\text{crit}}$. $\square$

*Critical Temperature Constraint.* From Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`, the critical temperature $T_c^* = 1/16$ implies that thermal fluctuations can restore symmetry (collapse daughters) if cognitive temperature ({prf:ref}`def-cognitive-temperature`) exceeds the barrier height. For stable fission, require:

$$
T_c < \frac{(\Xi - \Xi_{\text{crit}})^2}{4\alpha}.

$$

({prf:ref}`thm-fusion-criterion`) *theorem* — **Fusion Criterion**

Charts $i$ and $j$ shall be merged if and only if:

$$
G_\Delta(i, j) < \mathcal{C}_{\text{complexity}}(N_c) - \mathcal{C}_{\text{complexity}}(N_c - 1) + \epsilon_{\text{hysteresis}}

$$
where:
- $\mathcal{C}_{\text{complexity}}(N_c) = \log N_c + \lambda_{\text{param}} |\theta_{\text{chart}}|$ is the metabolic cost of maintaining $N_c$ charts ({ref}`Section 30.3 <sec-the-fission-criterion>`),
- $\epsilon_{\text{hysteresis}} > 0$ is a hysteresis constant preventing oscillatory fission-fusion ("ontological churn").

$$
\mathcal{C}_{\text{complexity}}(N_c) - \mathcal{C}_{\text{complexity}}(N_c - 1) = \log\frac{N_c}{N_c - 1} + \lambda_{\text{param}} |\theta_{\text{chart}}|

$$
The hysteresis term $\epsilon_{\text{hysteresis}}$ breaks the symmetry with Fission, ensuring that a chart is not immediately re-created after being destroyed. $\square$

*Remark (Units):* All terms are in nats. The criterion is dimensionally consistent.

({prf:ref}`thm-subcritical-pitchfork-fusion`) *theorem* — **Subcritical Pitchfork for Fusion**

Let $r(s) := \|q_i(s) - q_j(s)\|$ be the query separation at computation time $s$. During fusion, the dynamics become:

$$
\frac{dr}{ds} = -(\Upsilon_{ij} - \Upsilon_{\text{crit}}) r - \alpha r^3 + \sigma\xi(s)

$$
where:
- $\Upsilon_{\text{crit}} \in (0, 1)$ is the critical redundancy threshold,
- $\alpha > 0$ is the cubic stabilization coefficient,
- $\sigma\xi(s)$ is white noise with intensity $\sigma$.

When $\Upsilon_{ij} > \Upsilon_{\text{crit}}$:
1. The linear term is **negative** (attractive toward $r = 0$).
2. $r = 0$ becomes the **unique stable attractor**.
3. The queries "fall into each other" until they merge.

*Contrast with Fission ({prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`):*

| Property                | Fission (Supercritical)          | Fusion (Subcritical)                |
|:------------------------|:---------------------------------|:------------------------------------|
| Linear term sign        | $+\mu r$ (repulsive from origin) | $-\mu r$ (attractive to origin)     |
| Trigger                 | $\Xi > \Xi_{\text{crit}}$        | $\Upsilon > \Upsilon_{\text{crit}}$ |
| Stable fixed points     | $r^* = \pm\sqrt{\mu/\alpha}$     | $r^* = 0$                           |
| Physical interpretation | Charts repel and separate        | Charts attract and merge            |

({prf:ref}`thm-reallocation-gradient`) *theorem* — **Optimal Reallocation Gradient**

Let $k_{\text{dead}}$ satisfy $U_{k_{\text{dead}}} < \epsilon_U$ and let $k_{\text{stressed}}$ satisfy $\mathcal{D}_{k_{\text{stressed}}} = \max_k \mathcal{D}_k$. The expected reduction in global distortion per reallocated code is:

$$
\frac{\delta \mathcal{D}}{\delta N_{\text{codes}}} \approx \frac{\mathcal{D}_{k_{\text{stressed}}}}{H(K = k_{\text{stressed}})}

$$

({prf:ref}`thm-thermodynamic-hysteresis-bound`) *theorem* — **Thermodynamic Lower Bound on Hysteresis**

Let $\mathcal{C}$ be a cycle of ontological operations consisting of a fission event $N_c \to N_c + 1$ followed immediately by a fusion event $N_c + 1 \to N_c$. Let $T_c$ be the cognitive temperature and $\mathcal{W}_{\text{comp}}$ be the metabolic work of parameter instantiation. To satisfy the generalized Second Law of Thermodynamics for open cognitive systems (Theorem {prf:ref}`thm-generalized-landauer-bound`), the hysteresis threshold must satisfy:

$$
\epsilon_{\text{hysteresis}} \geq \frac{1}{\beta_{\text{eff}}} \left( \Delta H_{\text{Shannon}} + \frac{1}{T_c}\mathcal{W}_{\text{comp}} \right)

$$
where $\beta_{\text{eff}} = 1/T_c$ is the inverse cognitive temperature and $\Delta H_{\text{Shannon}}$ is the entropy reduction associated with the discarded distinction.

1. **Fission Cost:** The creation of a new chart requires initializing a set of parameters $\theta_{\text{new}}$. By Landauer's Principle ({ref}`Landauer's Principle <pi-landauer-principle>`), the erasure of the previous random state of these memory units to a low-entropy initialization requires work $\mathcal{W}_{\text{init}} \geq k T_c \ln 2 \cdot |\theta_{\text{new}}|$.

2. **Fusion Cost:** The merger of two charts implies the erasure of the mutual information $I(X; \{K_i, K_j\}) - I(X; K_{i \cup j})$, defined as the Discrimination Gain $G_\Delta$ ({prf:ref}`def-discrimination-gain`). This is an irreversible logical operation, dissipating heat $Q_{\text{fus}} \geq T_c G_\Delta$.

3. **Cycle Condition:** For the cycle $\mathcal{C}$ to be non-spontaneous (preventing chattering), the total free energy change must be positive. The Governor imposes a metabolic efficiency constraint $\eta_{\text{ROI}} > \eta_{\min}$ ({ref}`Section 26 <sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller>`).

4. **Derivation:** The utility gain of the cycle is zero (the topology is unchanged). The cost is $\mathcal{W}_{\text{init}} + Q_{\text{fus}}$. For the cycle to be rejected by the Fusion Criterion ({prf:ref}`thm-fusion-criterion`), the hysteresis term must exceed the minimum metabolic dissipation of the cycle:

$$
\epsilon_{\text{hysteresis}} \geq \inf_{\mathcal{C}} \oint \dot{\mathcal{M}}(s) ds

$$
Substituting the Landauer bound yields the stated inequality. $\square$

({prf:ref}`thm-frechet-fusion-uniqueness`) *theorem* — **Existence and Uniqueness of Fusion Center**

Since the Poincare disk $(\mathbb{D}, G)$ is a complete, simply connected Riemannian manifold with non-positive sectional curvature ($K=-1$), it is a Hadamard space (global CAT(0) space). The squared distance function $d^2_{\mathbb{D}}(\cdot, y)$ is strictly convex. Therefore, the functional $F(q) = \sum \bar{w}_i d^2_{\mathbb{D}}(q, q_i)$ admits a unique global minimizer.

({prf:ref}`thm-fission-inhibition`) *theorem* — **Fission Inhibition Corollary**

Let $\mathcal{E}^{(\ell)}$ be the encoder at scale $\ell$. A Topological Fission event at layer $\ell$ (increasing chart count $N_c^{(\ell)} \to N_c^{(\ell)}+1$) strictly reduces the probability of fission at layer $\ell+1$.

2. **Approximation Theory:** Fission adds a centroid to the Voronoi partition at layer $\ell$. By standard quantization theory (Zador's theorem), increasing codebook size strictly reduces the mean squared quantization error (distortion), provided the data is not uniform.

3. **Variance Reduction:** The reconstruction error $\|z_{\text{tex}}^{(\ell)}\|^2$ decreases, implying the scale factor $\sigma^{(\ell)}$ decreases.

4. **Stress Damping:** Ontological Stress at layer $\ell+1$ is upper-bounded by the mutual information of its input. Since the input variance is reduced (relative to the pre-fission state), the extractable structure $I(x^{(\ell+1)}_t; x^{(\ell+1)}_{t+1})$ decreases.

5. **Conclusion:** Macro-scale adaptation absorbs structural variance, starving the micro-scale of the stress required to trigger bifurcation. $\square$

({prf:ref}`thm-generalized-landauer-bound`) *theorem* — **Generalized Landauer Bound**

The metabolic flux $\dot{\mathcal{M}}$ provides a physical lower bound on the rate of entropy reduction within the agent. Specifically:

$$
\dot{\mathcal{M}}(s) \ge T_c \left| \frac{d}{ds} H(\rho_s) \right|,

$$
where $H(\rho_s) = -\int_{\mathcal{Z}} \rho \ln \rho \, d\mu_G$ is the Shannon entropy and $T_c$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`, {ref}`Section 22.4 <sec-the-geodesic-baoab-integrator>`).

$$
\frac{d}{ds} H(\rho_s) = -\int_{\mathcal{Z}} (1 + \ln \rho) \partial_s \rho \, d\mu_G.

$$
Substituting the WFR continuity equation and integrating by parts (assuming vanishing flux at $\partial\mathcal{Z}$):

$$
\frac{d}{ds} H = \int_{\mathcal{Z}} \rho \langle \nabla \ln \rho, v \rangle_G \, d\mu_G - \int_{\mathcal{Z}} r \ln \rho \cdot \rho \, d\mu_G.

$$
By the Cauchy-Schwarz inequality on the tangent bundle $(T\mathcal{Z}, G)$:

$$
\left| \int_{\mathcal{Z}} \rho \langle \nabla \ln \rho, v \rangle_G \, d\mu_G \right| \le \left( \int_{\mathcal{Z}} \rho \|\nabla \ln \rho\|_G^2 \, d\mu_G \right)^{1/2} \left( \int_{\mathcal{Z}} \rho \|v\|_G^2 \, d\mu_G \right)^{1/2}.

$$
The first factor is the **Fisher Information** $\mathcal{I}(\rho) = \int \rho \|\nabla \ln \rho\|_G^2 \, d\mu_G$ {cite}`amari2016information`. Under the optimal transport scaling $v = -T_c \nabla \ln \rho$ (gradient flow of the free energy), we recover the de Bruijn identity {cite}`stam1959some` and the bound follows. The reaction term satisfies an analogous inequality via the $L^2(\rho)$ norm. See {ref}`Appendix E.3 <sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws>` for the full proof. $\square$

*Remark (Landauer's Principle).* The classical Landauer bound states that erasing one bit of information requires dissipating at least $k_B T \ln 2$ joules of heat. Theorem {prf:ref}`thm-generalized-landauer-bound` is the information-geometric generalization: reducing belief entropy by $\Delta H$ nats requires dissipating at least $T_c \cdot |\Delta H|$ nats of metabolic energy.

({prf:ref}`thm-deliberation-optimality-condition`) *theorem* — **Deliberation Optimality Condition**

Let $\rho_s$ evolve as a gradient flow of $V$ under WFR dynamics. The optimal computation budget $S^*$ satisfies:

$$
\left. \frac{d}{ds} \langle V \rangle_{\rho_s} \right|_{s=S^*} = \dot{\mathcal{M}}(S^*),

$$
provided such an $S^*$ exists in $(0, S_{\max})$.

$$
\frac{d}{dS} \mathcal{S}_{\text{delib}} = -\frac{d}{dS} \langle V \rangle_{\rho_S} + \dot{\mathcal{M}}(S).

$$
The first term is the **Value-Improvement Rate**:

$$
\frac{d}{dS} \langle V \rangle_{\rho_S} = \int_{\mathcal{Z}} V(z) \partial_s \rho(S, z) \, d\mu_G.

$$
Applying the WFR continuity equation $\partial_s \rho = \rho r - \nabla \cdot (\rho v)$:

$$
\frac{d}{dS} \langle V \rangle_{\rho_S} = \int_{\mathcal{Z}} V \cdot \rho r \, d\mu_G + \int_{\mathcal{Z}} V (-\nabla \cdot (\rho v)) \, d\mu_G.

$$
Integrating the divergence term by parts (assuming vanishing flux at $\partial\mathcal{Z}$):

$$
\int_{\mathcal{Z}} V (-\nabla \cdot (\rho v)) \, d\mu_G = \int_{\mathcal{Z}} \rho \langle \nabla V, v \rangle_G \, d\mu_G.

$$
For gradient flow dynamics, $v = -G^{-1} \nabla V$ (up to temperature scaling), so $\langle \nabla V, v \rangle_G = -\|\nabla V\|_G^2 \le 0$. Thus:

$$
\frac{d}{dS} \langle V \rangle_{\rho_S} = \int_{\mathcal{Z}} \rho \left( V r - \|\nabla V\|_G^2 \right) d\mu_G.

$$
The stationarity condition $\frac{d}{dS} \mathcal{S}_{\text{delib}} = 0$ yields the optimality condition. See {ref}`Appendix E.4 <sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws>` for the full proof using the WFR adjoint operator. $\square$

*Physical interpretation:* The optimal stopping time $S^*$ is reached when the marginal gain in expected value (the "return on thinking") exactly equals the marginal metabolic cost (the "price of thinking"). At $S^*$, the agent has extracted all cost-effective information from deliberation.

({prf:ref}`thm-fast-slow-phase-transition`) *theorem* — **Fast/Slow Phase Transition**

Let $\Gamma(s) := \left| \frac{d}{ds} \langle V \rangle_{\rho_s} \right|$ be the **Value-Improvement Rate**. There exists a critical threshold such that:

1. **Reflexive Regime (Fast):** If $\Gamma(0) < \dot{\mathcal{M}}(0)$, then $S^* = 0$. The agent executes an immediate action based on the prior $\rho_0$.

2. **Deliberative Regime (Slow):** If $\Gamma(0) > \dot{\mathcal{M}}(0)$, then $S^* > 0$. The agent enters a planning state, terminating only when the marginal gain in Value equals the marginal metabolic cost.

$$
\left. \frac{d}{dS} \mathcal{S}_{\text{delib}} \right|_{S=0} = -\Gamma(0) + \dot{\mathcal{M}}(0).

$$
If $\Gamma(0) < \dot{\mathcal{M}}(0)$, then $\frac{d}{dS} \mathcal{S}_{\text{delib}}|_{S=0} > 0$. Since $\mathcal{S}_{\text{delib}}$ is increasing at $S=0$ and we assume $\mathcal{S}_{\text{delib}}$ is convex (which holds when $\Gamma(s)$ is decreasing due to diminishing returns), the minimum occurs at the boundary $S^* = 0$.

If $\Gamma(0) > \dot{\mathcal{M}}(0)$, then $\frac{d}{dS} \mathcal{S}_{\text{delib}}|_{S=0} < 0$. The agent benefits from deliberation. As $s$ increases, $\Gamma(s)$ decreases (diminishing marginal returns on thinking) while $\dot{\mathcal{M}}(s)$ may increase or remain constant. The optimum $S^* > 0$ occurs when the curves cross: $\Gamma(S^*) = \dot{\mathcal{M}}(S^*)$. $\square$

*Remark (Dual-Process Theory).* Theorem {prf:ref}`thm-fast-slow-phase-transition` provides a first-principles derivation of Kahneman's "System 1 / System 2" dichotomy {cite}`kahneman2011thinking`. System 1 (reflexive) corresponds to $S^* = 0$; System 2 (deliberative) corresponds to $S^* > 0$. The transition is not a cognitive style but a phase transition governed by the ratio $\Gamma(0) / \dot{\mathcal{M}}(0)$.

({prf:ref}`thm-generalized-stopping`) *theorem* — **Generalized Stopping for Non-Conservative Fields**

When the Value Curl does not vanish ($\mathcal{F} \neq 0$, Definition {prf:ref}`def-value-curl`), the agent converges to a Non-Equilibrium Steady State (Theorem {prf:ref}`thm-ness-existence`) rather than a fixed point. The stopping criterion generalizes as follows:

**Conservative Case ($\mathcal{F} = 0$):** Stop when the Value-Improvement Rate equals the metabolic cost:

$$
\Gamma(S^*) = \dot{\mathcal{M}}(S^*)

$$

**Non-Conservative Case ($\mathcal{F} \neq 0$):** Stop when the **orbit parameters converge**:

$$
\frac{d}{ds}\|\text{Orbit}(s)\|_{\text{param}} < \epsilon_{\text{orbit}}

$$
even if the agent continues moving within the limit cycle.

*Remark.* In the conservative case, convergence is to a fixed point ($\dot{z} \to 0$). In the non-conservative case, convergence is to a stable limit cycle (periodic orbit with constant parameters).

**Operational Criterion:** Define the orbit-change metric as:

$$
\Delta_{\text{orbit}}(s) := \left\| \oint_{\gamma_s} \mathcal{R} - \oint_{\gamma_{s-\delta}} \mathcal{R} \right\|

$$
where $\gamma_s$ is the closed trajectory over one cycle at time $s$. Stop when $\Delta_{\text{orbit}}(s) < \epsilon_{\text{orbit}}$.

*Remark.* In the non-conservative case, the agent accumulates reward along periodic trajectories. Deliberation terminates when the orbit parameters stabilize, not when motion ceases.

({prf:ref}`thm-total-entropy-production`) *theorem* — **Total Entropy Production**

The total entropy production rate of the agent $\sigma_{\text{tot}}$ during computation is:

$$
\sigma_{\text{tot}}(s) := \frac{d}{ds} H(\rho_s) + \frac{1}{T_c} \dot{\mathcal{M}}(s) \ge 0.

$$

$$
\sigma_{\text{tot}} = \frac{dH}{ds} + \frac{\dot{\mathcal{M}}}{T_c} \ge \frac{dH}{ds} + \left| \frac{dH}{ds} \right| = \frac{dH}{ds} - \frac{dH}{ds} = 0.

$$
If $\frac{d}{ds} H \ge 0$, then $\sigma_{\text{tot}} \ge 0$ trivially since $\dot{\mathcal{M}} \ge 0$. $\square$

*Interpretation:* The agent can only reduce its internal uncertainty ($dH/ds < 0$) by dissipating metabolic energy ($\dot{\mathcal{M}} > 0$) {cite}`still2012thermodynamics`. This defines the **Efficiency of Thought**:

$$
\eta_{\text{thought}} := \frac{-T_c \cdot dH/ds}{\dot{\mathcal{M}}} \le 1.

$$
An agent is "thermodynamically fragile" if it requires high metabolic flux for low entropy reduction ($\eta_{\text{thought}} \ll 1$).

({prf:ref}`thm-the-interventional-gap`) *theorem* — **The Interventional Gap**

Let $P_{\text{obs}}(z' | z, a)$ be the conditional density obtained via passive observation, and $P_{\text{int}}(z' | do(z, a))$ be the density under intervention. We define the **Causal Deficit** $\Delta_{\text{causal}}: \mathcal{Z} \times \mathcal{A} \to \mathbb{R}_{\ge 0}$ as:

$$
\Delta_{\text{causal}}(z, a) := D_{\text{KL}} \left( P_{\text{int}}(z' | do(z, a)) \| P_{\text{obs}}(z' | z, a) \right).

$$
*Interpretation:* The Causal Deficit measures the discrepancy between interventional and observational predictions. If $\Delta_{\text{causal}} = 0$, the observational model is causally correct -- correlations reflect true causal mechanisms. If $\Delta_{\text{causal}} > 0$, the agent has mistaken a correlation for a causal link (confounding) or vice versa.

$$
\text{Vol}_{\text{ignorant}} := \int_{\mathcal{Z} \times \mathcal{A}} \mathbb{I}[\Delta_{\text{causal}}(z, a) > 0] \, d\mu_G(z) \, da.

$$
This volume represents the region of state-action space where the agent's observational model fails to predict interventional outcomes. $\square$

({prf:ref}`thm-augmented-drift-law`) *theorem* — **Augmented Drift Law**

The Equation of Motion ({ref}`Section 22.2 <sec-the-coupled-jump-diffusion-sde>`) is extended by the **Interventional Force** $\mathbf{f}_{\text{exp}}$:

$$
F_{\text{total}} = \underbrace{-G^{-1} \nabla_G V}_{\text{Utility Force}} + \underbrace{\beta_{\text{exp}} \mathbf{f}_{\text{exp}}}_{\text{Curiosity Force}},

$$
where:
- $\mathbf{f}_{\text{exp}} := G^{-1} \nabla_z \Psi_{\text{causal}}$ is the gradient of the causal potential
- $\beta_{\text{exp}} \ge 0$ is the **exploration coefficient** balancing exploitation vs. exploration

$$
\frac{d}{dt}\left( G_{kj} \dot{z}^j \right) - \frac{1}{2} \partial_k G_{ij} \dot{z}^i \dot{z}^j = -\partial_k V - \beta_{\text{exp}} \partial_k \Psi_{\text{causal}}.

$$
Expanding the left-hand side and identifying the Christoffel symbols of the first kind $[ij, k] = \frac{1}{2}(\partial_i G_{jk} + \partial_j G_{ik} - \partial_k G_{ij})$:

$$
G_{kj} \ddot{z}^j + [ij, k] \dot{z}^i \dot{z}^j = -\partial_k V - \beta_{\text{exp}} \partial_k \Psi_{\text{causal}}.

$$
Contracting with $G^{mk}$ and using $\Gamma^m_{ij} = G^{mk}[ij, k]$:

$$
\ddot{z}^m + \Gamma^m_{ij} \dot{z}^i \dot{z}^j = -G^{mk} \partial_k V - \beta_{\text{exp}} G^{mk} \partial_k \Psi_{\text{causal}}.

$$
In the overdamped limit ({ref}`Section 22.3 <sec-the-unified-effective-potential>`), the acceleration term vanishes and the drift field is $F_{\text{total}} = -G^{-1}\nabla V + \beta_{\text{exp}} G^{-1}\nabla\Psi_{\text{causal}}$. See {ref}`Appendix E.5 <sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws>` for the full derivation. $\square$

*Physical interpretation:* The curiosity force $\mathbf{f}_{\text{exp}}$ pulls the agent toward regions of high epistemic uncertainty about the transition dynamics. This is the geometric formulation of **intrinsic motivation** {cite}`schmidhuber2010formal,oudeyer2007intrinsic`: the agent is rewarded for reducing its causal ignorance, independent of external task reward. This connects to curiosity-driven exploration in reinforcement learning {cite}`pathak2017curiosity,houthooft2016vime`.

({prf:ref}`thm-interventional-closure`) *theorem* — **Interventional Closure**

The macro-ontology $K$ is **Interventionally Closed** if and only if the predictability of the macro-state is invariant under $do$-operations:

$$
I(K_{t+1} ; Z_{\text{micro}, t} | K_t, do(A_t)) = 0.

$$
*Interpretation:* If an agent moves an object (intervention), and the resulting macro-state $K_{t+1}$ depends on micro-texture $z_{\text{tex}}$ that was previously labeled "noise," the ontology has failed. The intervention has **exposed a hidden variable**, triggering **Ontological Expansion** ({ref}`Section 30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`).

If the observational distribution is closed ($I = 0$), and the mechanism is invariant, the interventional distribution is necessarily closed. A violation ($I > 0$ under $do$) implies the existence of a back-door path through $Z_{\text{micro}}$ that was previously unobserved, necessitating a topological expansion of $K$ to include the confounding variable. See {ref}`Appendix E.6 <sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws>` for the full proof. $\square$

*Remark (Interventional Debugging).* Theorem {prf:ref}`thm-interventional-closure` provides a diagnostic for ontological adequacy: if the agent's predictions fail specifically under intervention but succeed under observation, the ontology contains a hidden confounder. This is the geometric manifestation of Simpson's paradox {cite}`pearl2009causality`. Algorithmic approaches to discovering such confounders are developed in the causal discovery literature {cite}`spirtes2000causation`.

({prf:ref}`thm-szilard-transducer-bound`) *theorem* — **The Transducer Bound**

Let $r_t$ be the instantaneous reward signal with information content $\mathcal{I}_{\text{util}}(r_t)$ nats. The maximum free energy extractable per unit time is bounded by:

$$
\dot{E}_{\text{in}}^{\max}(t) = k_B T_{\text{env}} \cdot \mathcal{I}_{\text{util}}(r_t)

$$

where $T_{\text{env}}$ is the environmental temperature (characterizing energy availability).

({prf:ref}`thm-autopoietic-inequality`) *theorem* — **The Autopoietic Inequality**

Let $\tau > 0$ be a target survival horizon. A **sufficient condition** for the agent to survive at time $\tau$ (i.e., $B(\tau) > 0$) is:

$$
\int_0^\tau \left( \mathfrak{T}_{\text{harvest}}(r_t) - \dot{\mathcal{M}}(t) \right) dt > \gamma_{\text{leak}} \int_0^\tau B(t) \, dt - B_0

$$

*Equivalently:* The time-averaged **Net Harvest Rate** must be positive:

$$
\langle \mathfrak{T} - \dot{\mathcal{M}} \rangle_\tau > \gamma_{\text{leak}} \langle B \rangle_\tau - \frac{B_0}{\tau}

$$

Requiring $B(\tau) > 0$ and rearranging yields the inequality. $\square$

*Physical interpretation:* The agent must harvest more energy than it dissipates. This is the **autopoietic closure condition**—the system must actively maintain its own organization against thermodynamic decay.

({prf:ref}`thm-information-maintenance-cost`) *theorem* — **The Information-Maintenance Cost**

Maintaining Fisher Information $I_F$ on the latent manifold $(\mathcal{Z}, G)$ requires continuous energy expenditure:

$$
\dot{E}_{\text{maintain}} \geq \frac{1}{2} T_c \cdot I_F

$$

where $T_c$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`) and $I_F$ is the Fisher Information of the belief distribution.

2. **de Bruijn identity** {cite}`stam1959some,cover2006elements`: Under diffusion $d\rho/dt = T_c \Delta_G \rho$, entropy evolves as:
   $$\frac{dH[\rho]}{dt} = \frac{1}{2} I_F[\rho]$$
   Entropy increases at rate proportional to Fisher Information.

3. **Landauer cost:** By Theorem {prf:ref}`thm-generalized-landauer-bound`, maintaining entropy against diffusion requires:
   $$\dot{E}_{\text{maintain}} \geq T_c \left| \frac{dH}{dt} \right| = \frac{1}{2} T_c \cdot I_F$$

4. **Interpretation:** Sharp probability distributions (high $I_F$) cost more to maintain. $\square$

({prf:ref}`thm-fading-metric-law`) *theorem* — **The Fading Metric Law**

When available energy $B(t)$ falls below the maintenance requirement, the effective metric contracts. The **effective metric** is:

$$
G_{ij}^{\text{eff}}(z, B) = f\left(\frac{B}{B_{\text{crit}}}\right) \cdot G_{ij}(z)

$$

where:
- $G_{ij}(z)$ is the full-capacity metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`)
- $B_{\text{crit}}$ is the **critical energy** required to sustain full metric resolution
- $f: [0, \infty) \to [0, 1]$ is the **fading function** with $f(0) = 0$, $\lim_{x \to \infty} f(x) = 1$

**Specific form:** The fading function satisfying thermodynamic constraints is:

$$
f(x) = 1 - e^{-x}

$$

This gives exponential saturation: $f(x) \approx x$ for $x \ll 1$ (linear regime) and $f(x) \approx 1$ for $x \gg 1$ (saturation).

2. **Signal-to-noise scaling:** Neural signals have SNR proportional to available energy:
   $$\text{SNR} \propto \sqrt{\frac{E_{\text{available}}}{E_{\text{noise}}}} = \sqrt{\frac{B}{B_{\text{crit}}}}$$

3. **Fisher Information scaling:** Since Fisher Information scales as SNR²:
   $$I_F^{\text{eff}} \propto \text{SNR}^2 \propto \frac{B}{B_{\text{crit}}}$$

4. **Metric scaling:** The metric tensor scales with Fisher Information:
   $$G^{\text{eff}} \propto I_F^{\text{eff}} \propto \frac{B}{B_{\text{crit}}} \quad \text{for } B \ll B_{\text{crit}}$$

5. **Saturation:** For $B \gg B_{\text{crit}}$, the metric saturates at $G$ (maximum resolution). The exponential form $f(x) = 1 - e^{-x}$ interpolates smoothly between these regimes. $\square$

({prf:ref}`thm-augmented-value-equation`) *theorem* — **The Augmented Value Equation**

The total effective potential combines task and homeostatic contributions:

$$
\Phi_{\text{total}}(z, B) = \Phi_{\text{task}}(z) + \Phi_{\text{homeo}}(z, B)

$$

The value function satisfies the augmented screened Poisson equation ({ref}`Section 24 <sec-the-reward-field-value-forms-and-hodge-geometry>`):

$$
(-\Delta_{G^{\text{eff}}} + \kappa^2) V = \rho_r + \rho_{\text{homeo}}

$$

where:
- $G^{\text{eff}} = f(B/B_{\text{crit}}) \cdot G$ is the faded metric (Theorem {prf:ref}`thm-fading-metric-law`)
- $\rho_{\text{homeo}} = -\Delta \Phi_{\text{homeo}}$ is the homeostatic source term
- The screening mass $\kappa = -\ln \gamma$ remains unchanged

*Consequence:* Both the metric (geometry) and the source term (drive) depend on battery state.

({prf:ref}`thm-carnot-transduction-bound`) *theorem* — **The Carnot Bound on Transduction**

The transduction efficiency is bounded by the Carnot limit:

$$
\eta \leq \eta_{\text{Carnot}} = 1 - \frac{T_c}{T_{\text{env}}}

$$

where $T_c$ is the agent's cognitive temperature and $T_{\text{env}}$ is the environmental temperature.


*Consequence:* The agent must maintain $T_c < T_{\text{env}}$ (a thermal gradient) to extract any work. If $T_c \geq T_{\text{env}}$, then $\eta \leq 0$ and no harvesting is possible.

({prf:ref}`thm-locking-operator-derivation`) *theorem* — **Derivation of the Locking Operator**

The Locking Operator $\mathfrak{L}_{\text{sync}}$ is the Yang-Mills energy of the inter-agent connection:

$$
\mathfrak{L}_{\text{sync}}(G_A, G_B) := -\frac{1}{4g_{\text{lock}}^2} \int_{\mathcal{Z}_{\text{shared}}} \text{Tr}(\mathcal{F}_{AB}^{\mu\nu} \mathcal{F}_{AB,\mu\nu}) \sqrt{|G_{AB}|} \, d^D z

$$

**Step 1.** By Definition {prf:ref}`def-gauge-covariant-game-tensor`, each agent's belief spinor $\psi^{(i)}$ transforms under local gauge $U^{(i)}(z) \in G_{\text{Fragile}}$.

**Step 2.** The joint space $\mathcal{Z}_A \times \mathcal{Z}_B$ carries a product gauge group $G^{(A)} \times G^{(B)}$. By the minimal coupling principle (Proposition {prf:ref}`prop-minimal-coupling`), dynamics on the joint space require a connection.

**Step 3.** The curvature $\mathcal{F}_{AB}^{\mu\nu}$ of Definition {prf:ref}`def-locking-curvature` measures the failure of the connection to be flat. By standard gauge theory, this curvature vanishes if and only if:

$$
A_\mu^{(A)}(z) \sim A_\mu^{(B)}(z) \quad \text{(gauge equivalent)}

$$

**Step 4.** The Yang-Mills action principle (Definition {prf:ref}`def-yang-mills-action`) states that physical configurations minimize the integrated curvature squared. Applying this to $\mathcal{A}_{AB}$ yields the Locking Operator.

**Step 5.** The normalization $-1/(4g_{\text{lock}}^2)$ ensures correct dimensionality: $[\mathfrak{L}_{\text{sync}}] = \text{nat}$.

**Step 6 (Identification).** The Locking Operator generates a **Synchronizing Potential** $\Psi_{\text{sync}}$ that penalizes geometric disagreement. By comparison geometry, the local Gromov-Hausdorff distance satisfies:

$$
d_{\text{GH}}(\mathcal{U}_A, \mathcal{U}_B) \leq C \cdot \|\mathcal{F}_{AB}\|^{1/2}

$$

for a universal constant $C > 0$. Thus $\mathfrak{L}_{\text{sync}}$ controls the metric alignment.

$\square$

({prf:ref}`thm-spontaneous-gauge-locking`) *theorem* — **Spontaneous Gauge Locking**

Consider two agents interacting in a shared environment $E$. If they minimize the joint prediction error:

$$
\mathcal{L}_{\text{joint}} = \|\hat{x}_{t+1}^A - x_{t+1}\|^2 + \|\hat{x}_{t+1}^B - x_{t+1}\|^2 + \beta \Psi_{\text{sync}}

$$

Then, as the interaction coupling $\beta \to \infty$, the system undergoes a phase transition where the internal gauge groups $U_A(z)$ and $U_B(z)$ become locked:

$$
U_A(z) \cdot U_B^{-1}(z) \to \text{const}.

$$

**Step 1 (Setup).** Let $\psi^{(A)}, \psi^{(B)}$ be belief spinors (Definition {prf:ref}`def-cognitive-spinor`) with local gauge transformations:

$$
\psi'^{(i)} = U^{(i)}(z) \psi^{(i)}, \quad U^{(i)} \in G_{\text{Fragile}}

$$

**Step 2 (Prediction Error).** The prediction error for agent $i$ is:

$$
\epsilon^{(i)} = \|D^{(i)}(\psi^{(i)}) - x_{t+1}\|^2

$$

where $D^{(i)}$ is the TopologicalDecoder ({ref}`Section 7.10 <sec-decoder-architecture-overview-topological-decoder>`).

**Step 3 (Relative Gauge).** Define the relative gauge transformation:

$$
\Delta U(z) := U_A(z) U_B^{-1}(z)

$$

When $\Delta U \neq \text{const}$, the agents encode the same environment state $x$ with spatially varying internal orientations.

**Step 4 (Synchronization Potential).** The synchronization term from Definition {prf:ref}`def-locking-curvature` is:

$$
\Psi_{\text{sync}} = \int_{\mathcal{Z}_{\text{shared}}} \text{Tr}(\mathcal{F}_{AB}^{\mu\nu} \mathcal{F}_{AB,\mu\nu}) \, d\mu_G

$$

**Step 5 (Joint Action).** The joint WFR action (Definition {prf:ref}`def-joint-wfr-action`) becomes:

$$
\mathcal{A}_{\text{joint}} = \mathcal{A}_{\text{WFR}}^{(A)} + \mathcal{A}_{\text{WFR}}^{(B)} + \beta \Psi_{\text{sync}}

$$

**Step 6 (Gradient Flow).** At equilibrium, the functional derivative vanishes:

$$
\frac{\delta \mathcal{A}_{\text{joint}}}{\delta A_\mu^{(i)}} = 0

$$

This yields coupled Yang-Mills equations for both agents.

**Step 7 (Strong Coupling Limit).** As $\beta \to \infty$, the synchronization term dominates. The energy minimum requires $\Psi_{\text{sync}} \to 0$, hence $\mathcal{F}_{AB}^{\mu\nu} \to 0$.

**Step 8 (Flat Connection).** By Theorem {prf:ref}`thm-three-cognitive-forces`, a vanishing field strength tensor implies:

$$
[D_{AB}^\mu, D_{AB}^\nu] = 0

$$

Parallel transport on the joint bundle is path-independent.

**Step 9 (Gauge Alignment).** For simply-connected $\mathcal{Z}_{\text{shared}}$, a flat connection is pure gauge:

$$
A_\mu^{(A)}(z) - A_\mu^{(B)}(z) = \partial_\mu \chi(z)

$$

for some $\chi: \mathcal{Z} \to \mathfrak{g}$.

**Step 10 (Gauge Fixing).** The gauge transformation $U_A \to U_A e^{-i\chi}$ absorbs the gradient term, yielding:

$$
A_\mu^{(A)}(z) = A_\mu^{(B)}(z)

$$

in this fixed gauge.

**Step 11 (Phase Transition).** The transition from $\beta < \beta_c$ (unlocked) to $\beta > \beta_c$ (locked) is a continuous phase transition. The order parameter is:

$$
\langle |\phi_{AB}| \rangle = \begin{cases}
0 & \beta < \beta_c \\
v_{\text{lock}} = \sqrt{(\beta - \beta_c)/\lambda_{\text{lock}}} & \beta > \beta_c
\end{cases}

$$

This is analogous to Corollary {prf:ref}`cor-ontological-ssb`.

**Step 12 (Conclusion).** In the locked phase, $\Delta U(z) = U_A U_B^{-1} = \text{const}$, the constant being the residual global gauge freedom (the "shared coordinate system").

$\square$

({prf:ref}`thm-untranslatability-bound`) *theorem* — **The Untranslatability Bound**

The **Untranslatability** $\mathcal{U}_{AB}(m)$ of message $m$ between agents with misaligned gauges is bounded by the integrated curvature:

$$
\mathcal{U}_{AB}(m) \leq \|m\| \cdot \oint_{\partial\Sigma} \|\mathcal{F}_{AB}\|_F \, dA

$$

where $\Sigma$ is any surface bounded by the communication path.

**Step 1.** The translation operator around a closed loop $\gamma = \partial\Sigma$ yields the holonomy:

$$
\mathcal{H}_\gamma = \mathcal{P}\exp\left(-ig \oint_\gamma A_\mu \, dz^\mu\right)

$$

**Step 2.** By the non-Abelian Stokes theorem:

$$
\mathcal{H}_\gamma = \exp\left(-ig \int_\Sigma \mathcal{F}_{\mu\nu} \, dS^{\mu\nu}\right) + O(\mathcal{F}^2)

$$

**Step 3.** When $\mathcal{F}_{AB} \neq 0$, the holonomy is non-trivial: the message received by $B$ differs from the message sent by $A$.

**Step 4.** The discrepancy satisfies:

$$
\|m_{\text{received}} - m_{\text{sent}}\| \leq \|m\| \cdot \|\mathcal{H}_\gamma - \mathbb{1}\|

$$

**Step 5.** Bounding the holonomy deviation by the curvature integral via standard estimates yields the theorem.

$\square$

({prf:ref}`thm-babel-limit`) *theorem* — **The Babel Limit**

Let $\mathcal{L}$ be the Language Channel with Shannon capacity $C_{\mathcal{L}}$, and let $H(G_A)$ be the differential entropy rate of Agent $A$'s metric tensor. Complete gauge locking is achievable only if:

$$
\dim(\mathfrak{g}) \cdot H(G_A) \leq C_{\mathcal{L}}

$$

**Step 1.** By Theorem {prf:ref}`thm-causal-information-bound`, the maximum information transmittable through the Language Channel is:

$$
C_{\mathcal{L}} = \nu_D \cdot \frac{\text{Area}(\partial\mathcal{L})}{\ell_L^{D-1}}

$$

**Step 2.** To achieve complete gauge alignment, Agent $A$ must transmit sufficient information to specify all $\dim(\mathfrak{g})$ independent gauge parameters.

**Step 3.** The information required to specify the metric tensor $G_A$ at rate $r$ is $r \cdot H(G_A)$ nats per unit time.

**Step 4.** For full alignment, the transmitted information must cover all gauge degrees of freedom:

$$
I_{\text{required}} = \dim(\mathfrak{g}) \cdot H(G_A)

$$

**Step 5.** If $I_{\text{required}} > C_{\mathcal{L}}$, complete locking is impossible by Shannon's theorem. The residual unlocked subspace has dimension:

$$
d_{\text{unlocked}} = \dim(\mathfrak{g}) - \lfloor C_{\mathcal{L}} / H(G_A) \rfloor

$$

$\square$

({prf:ref}`thm-spectral-locking-order`) *theorem* — **Spectral Locking Order**

Under bandwidth-constrained communication, gauge locking proceeds in eigenvalue order. The locked subspace after time $T$ consists of the $k_{\max}$ highest eigenvalue components where:

$$
k_{\max} = \max\left\{k : \sum_{j=1}^k H(\sigma_j v_j) \leq C_{\mathcal{L}} \cdot T\right\}

$$


*Interpretation:* This explains why agents agree on "Gravity" (high eigenvalue, fundamental physics) before agreeing on "Politics" (low eigenvalue, high variance personal experience).

({prf:ref}`thm-emergence-objective-reality`) *theorem* — **Emergence of Objective Reality**

In the limit of perfect locking ($\mathcal{F}_{AB} \to 0$), the private manifolds $\mathcal{Z}_A$ and $\mathcal{Z}_B$ collapse into a single **Quotient Manifold**:

$$
\mathcal{Z}_{\text{shared}} := (\mathcal{Z}_A \sqcup \mathcal{Z}_B) / \sim_{\text{isometry}}

$$

where $\sim_{\text{isometry}}$ identifies points with vanishing metric friction.

**Step 1.** Perfect locking implies $\mathcal{F}_{AB}(z) = 0$ for all $z$.

**Step 2.** By Definition {prf:ref}`def-metric-friction`, this means:

$$
G_A(z) = \phi_{A \to B}^* G_B(\phi(z))

$$

The manifolds are isometric.

**Step 3.** Define the equivalence relation: $z_A \sim z_B$ iff $\phi_{A \to B}(z_A) = z_B$ and $G_A(z_A) = G_B(z_B)$.

**Step 4.** The quotient $\mathcal{Z}_{\text{shared}}$ inherits a well-defined metric from either $G_A$ or $G_B$ (they agree by isometry).

**Step 5.** To the agents, $\mathcal{Z}_{\text{shared}}$ appears as **Objective Reality**: it possesses properties (rigidity, persistence) that neither private imagination possesses alone.

$\square$

*Interpretation:* "Objective Reality" is a hallucination shared by $N$ agents with locked metrics. It is the fixed point of the consensus dynamics.

({prf:ref}`thm-markov-restoration`) *theorem* — **Markov Restoration on Causal Bundle**

Let $P(z^{(N)}_{t+\Delta t} | z^{(N)}_t, \Xi_{<t})$ denote the transition probability. When agents have finite causal delay $\tau_{ij} > 0$:

1. **On $\mathcal{Z}^{(N)}$ alone:** The Markov property fails:

   $$
   P(z^{(N)}_{t+\Delta t} | z^{(N)}_t) \neq P(z^{(N)}_{t+\Delta t} | z^{(N)}_{\leq t}).

   $$

2. **On $\mathcal{Z}_{\text{causal}}$:** The Markov property is restored:

   $$
   P\left((z^{(N)}_{t+\Delta t}, \Xi_{<t+\Delta t}) \,\big|\, (z^{(N)}_t, \Xi_{<t})\right) = P\left((z^{(N)}_{t+\Delta t}, \Xi_{<t+\Delta t}) \,\big|\, \text{full history}\right).

   $$

({prf:ref}`thm-strategic-delay-tensor`) *theorem* — **Strategic Delay Tensor**

The effective coupling tensor $\mathcal{T}_{ij}$ between agents splits into instantaneous and retarded components:

$$
\mathcal{T}_{ij}^{\text{total}}(t) = \underbrace{\mathcal{T}_{ij}^{\text{local}}(t)}_{\text{Short-range}} + \underbrace{\int_{-\infty}^t \mathcal{K}_{\text{delay}}(t-\tau) \mathcal{T}_{ij}^{\text{ghost}}(\tau) \, d\tau}_{\text{Long-range Retarded}},

$$
where $\mathcal{K}_{\text{delay}}(t-\tau) = \delta(t - \tau - \tau_{ij})$ is the delay kernel.

**Adversarial consequence:** Against a distant adversary, the effective metric inflation (from the Game Tensor) is delayed. An agent may commit to an aggressive trajectory only to experience a "wall" of increased inertia arriving from the opponent's past actions.

({prf:ref}`thm-hjb-klein-gordon`) *theorem* — **HJB-Klein-Gordon Correspondence**

Let information propagate at speed $c_{\text{info}}$. The Value Function $V^{(i)}(z, t)$ for Agent $i$ satisfies the **Screened Wave Equation**:

$$
\boxed{\left( \frac{1}{c_{\text{info}}^2} \frac{\partial^2}{\partial t^2} + \gamma_{\text{damp}} \frac{\partial}{\partial t} - \Delta_{G^{(i)}} + \kappa_i^2 \right) V^{(i)}(z, t) = \rho^{(i)}_r(z, t) + \sum_{j \neq i} \rho^{\text{ret}}_{ij}(z, t)}

$$
where:
- $\square_{G} = \frac{1}{c_{\text{info}}^2}\partial_t^2 - \Delta_G$ is the **D'Alembertian** on the manifold
- $\gamma_{\text{damp}} \geq 0$ is the temporal damping rate (related to discount)
- $\kappa_i$ is the **spatial screening mass** with $[\kappa_i] = 1/[\text{length}]$, related to the discount factor by:
  $$\kappa_i = \frac{-\ln\gamma_i}{c_{\text{info}} \Delta t} = \frac{\kappa_{i,\text{temporal}}}{c_{\text{info}}}$$
  where $\kappa_{i,\text{temporal}} = -\ln\gamma_i / \Delta t$ is the temporal discount rate with units $1/[\text{time}]$
- $\rho^{(i)}_r$ is the local reward source (units: $[\text{nat}]/[\text{length}]^2$)
- $\rho^{\text{ret}}_{ij}$ is the retarded interaction source density (Definition {prf:ref}`def-retarded-interaction-potential`)


*Character:* This is a hyperbolic PDE (wave equation with mass and damping), in contrast to the elliptic Helmholtz equation of {ref}`Section 24.2 <sec-the-bulk-potential-screened-poisson-equation>`.

({prf:ref}`thm-adversarial-mass-inflation`) *theorem* — **Adversarial Mass Inflation**

In a competitive game where Agent $j$ is adversarial ($\beta_{ij} > 0$) and the Game Tensor $\mathcal{G}_{ij}$ is positive semi-definite, the effective metric $\tilde{G}^{(i)}$ satisfies:

$$
\tilde{G}^{(i)}_{kl} \xi^k \xi^l \geq G^{(i)}_{kl} \xi^k \xi^l \quad \forall \xi \in T_{z}\mathcal{Z}^{(i)}.

$$
*Consequence:* The effective **Mass** $M^{(i)}(z)$ (Definition {prf:ref}`def-mass-tensor`) of Agent $i$ increases: $\tilde{M}^{(i)} \geq M^{(i)}$.

*First-Principles Interpretation:* Adversarial presence "thickens" the latent space. The agent moves more slowly (smaller geodesic steps) because it must account for the adversary's counter-maneuvers. **Strategic uncertainty is geometrically identical to physical inertia.**

({prf:ref}`thm-nash-standing-wave`) *theorem* — **Nash Equilibrium as Standing Wave**

In the relativistic formulation, a Nash equilibrium is a joint density $\boldsymbol{\rho}^*(\mathbf{z}, t)$ satisfying **time-averaged stationarity**:

$$
\left\langle \frac{\partial \boldsymbol{\rho}^*}{\partial t} \right\rangle_T := \frac{1}{T}\int_0^T \frac{\partial \boldsymbol{\rho}^*}{\partial t}(\mathbf{z}, t') \, dt' = 0,

$$
where the averaging period $T \gg \max_{i,j} \tau_{ij}$ exceeds all causal delays.

**Characterization:** A standing wave Nash equilibrium satisfies:

1. **Time-averaged gradient vanishing:**

   $$
   \left\langle (G^{(i)})^{-1} \nabla_{z^{(i)}} \Phi_{\text{eff}}^{(i,\text{ret})} \right\rangle_T = 0 \quad \forall i

   $$

2. **Balanced probability currents:** The flux exchanged between agents via retarded potentials is balanced over one wave period:

   $$
   \int_0^T \mathbf{J}^{(i)}(z, t) \, dt = 0 \quad \text{for all } z \in \mathcal{Z}^{(i)}

   $$
   where $\mathbf{J}^{(i)} = \rho^{(i)} \mathbf{v}^{(i)}$ is the probability current.

3. **Resonance condition:** The system oscillates at the characteristic causal frequency:

   $$
   \omega_{\text{Nash}} \sim \frac{c_{\text{info}}}{\bar{d}_{\mathcal{E}}},

   $$
   where $\bar{d}_{\mathcal{E}}$ is the mean environment distance between agents.

({prf:ref}`thm-nash-equilibrium-as-geometric-stasis`) *theorem* — **Geometric Stasis (Newtonian Limit)**

In the Newtonian limit ($c_{\text{info}} \to \infty$), a strategy profile $\mathbf{z}^* = (z^{(1)*}, \ldots, z^{(N)*})$ is a Nash equilibrium if and only if it satisfies the instantaneous **geometric stasis conditions**:

1. **Vanishing individual gradient:**

   $$
   (G^{(i)})^{-1} \nabla_{z^{(i)}} \Phi_{\text{eff}}^{(i)}(z^{(i)*}; z^{(-i)*}) = 0 \quad \forall i

   $$

2. **Stationary Game Tensor:**

   $$
   \frac{d}{dt}\mathcal{G}_{ij}^{kl}\bigg|_{\mathbf{z}^*} = 0 \quad \forall i,j

   $$

3. **Non-positive second variation:**

   $$
   \delta^2 V^{(i)}|_{z^{(i)*}} \leq 0 \quad \forall i, \forall \delta z^{(i)}

   $$

*Remark (Nash vs. Pareto).* Geometric stasis need not coincide with global optimality (Pareto). The Game Tensor eigenstructure determines the gap: trace-negative (cooperative) tends toward Pareto-improving basins; trace-positive (adversarial) tends toward Pareto-suboptimal saddles.

({prf:ref}`thm-mean-field-metric-law`) *theorem* — **Mean-Field Metric Law**

Let $\boldsymbol{z} = (z_1, \dots, z_N)$ be the configuration of $N$ agents on $\mathcal{Z}$. Let the empirical measure be $\mu_N = \frac{1}{N} \sum_{i=1}^N \delta_{z_i}$. As $N \to \infty$, assuming $\mu_N$ converges weakly to a smooth density $\rho \in \mathcal{P}(\mathcal{Z})$, the effective metric $\tilde{G}(z)$ for a test agent at position $z$ converges to:

$$
\tilde{G}(z) = G_{\text{intrinsic}}(z) + \alpha_{\text{adv}} \nabla^2_z \left( \Phi_{\text{int}} * \rho \right)(z)

$$
where $\Phi_{\text{int}}(z, \zeta)$ is the pairwise interaction potential ({prf:ref}`prop-interaction-kernel`) and $*$ denotes the Riemannian convolution.

2. **Discrete Game Tensor:** The Game Tensor acting on the metric is defined as the sum of cross-sensitivities ({prf:ref}`thm-adversarial-mass-inflation`):

$$
(\delta G)_{ab}(z_i) = \alpha_{\text{adv}} \sum_{j \neq i} \frac{\partial^2 \Phi_{\text{int}}(z_i, z_j)}{\partial z_i^a \partial z_i^b}.

$$
3. **Continuum Limit:** We rewrite the sum as an integral against the empirical measure:

$$
(\delta G)_{ab}(z) = \alpha_{\text{adv}} \int_{\mathcal{Z}} \nabla^2_{z, a, b} \Phi_{\text{int}}(z, \zeta) \, d\mu_N(\zeta).

$$
4. **Convergence:** Assuming $\Phi_{\text{int}}$ is $C^2$ and bounded, and $\mu_N \rightharpoonup \rho$ weakly, the integral converges to the convolution $(\nabla^2 \Phi_{\text{int}} * \rho)(z)$.

5. **Complexity Reduction:** The computation of $\tilde{G}$ now requires evaluating the Hessian of a static field $\Psi(z) = (\Phi_{\text{int}} * \rho)(z)$. This is $O(1)$ with respect to $N$ (given the density field), effectively decoupling the agent's complexity from the population size. $\square$

({prf:ref}`thm-metabolic-tracking-bound`) *theorem* — **Metabolic Tracking Bound**

Let $z^*(t)$ be a time-varying Nash equilibrium. An agent with maximum metabolic flux budget $\dot{\mathcal{M}}_{\max}$ can maintain tracking error $\epsilon \to 0$ if and only if the target's trajectory satisfies:

$$
\|\dot{z}^*\|_{\tilde{G}(z^*)} \leq \sqrt{\frac{2 \dot{\mathcal{M}}_{\max}}{\sigma_{\text{met}}}}

$$
where $\tilde{G}$ is the game-augmented metric ({prf:ref}`thm-adversarial-mass-inflation`).

2. **Thermodynamic Cost:** The metabolic cost of transport is $\dot{\mathcal{M}} = \frac{1}{2} \sigma_{\text{met}} \|v\|_{\tilde{G}}^2$ ({prf:ref}`def-metabolic-flux`).

3. **Adversarial Drag:** The metric $\tilde{G} = G + \alpha \mathcal{G}_{ij}$ includes the Game Tensor. High adversarial tension ($\mathcal{G}_{ij} \gg 0$) inflates the norm $\|\cdot\|_{\tilde{G}}$.

4. **Critical Failure:** If the adversary moves sufficiently fast or the conflict is sufficiently intense, the required dissipation exceeds $\dot{\mathcal{M}}_{\max}$. The agent loses tracking not due to algorithmic error, but due to exceeding its thermodynamic budget. $\square$

({prf:ref}`thm-geometric-locking-principle`) *theorem* — **Geometric Locking Principle**

Consider $N$ agents with Game Tensor $\mathcal{G}_{ij}$ ({prf:ref}`def-the-game-tensor`). In the presence of strong adversarial coupling, the joint system tends toward configurations where $\operatorname{Tr}(\mathcal{G}_{ij})$ is minimized.

1. **Metric Inflation:** By {prf:ref}`thm-adversarial-mass-inflation`, the effective metric for agent $i$ is $\tilde{G}^{(i)} = G^{(i)} + \sum_j \beta_{ij} \mathcal{G}_{ij}$. For adversarial agents, $\beta_{ij} > 0$ and $\mathcal{G}_{ij}$ is positive semi-definite, implying $\det(\tilde{G}^{(i)}) \ge \det(G^{(i)})$.

2. **Kinetic Cost:** The WFR action ({prf:ref}`def-joint-wfr-action`) includes the transport term $\int \|v\|_{\tilde{G}}^2 d\rho$. An inflated metric implies a higher metabolic cost for any movement $v \neq 0$.

3. **Energy Minimization:** The system evolves to minimize the free energy $\mathcal{F}$. If the potential gain $\nabla V$ is bounded, but the kinetic cost scales with $\mathcal{G}_{ij}$, trajectories with large $\mathcal{G}_{ij}$ (intense conflict) become energetically prohibitive.

4. **Stationarity:** The system relaxes to a state where either $v \to 0$ (Nash stasis, {prf:ref}`thm-nash-equilibrium-as-geometric-stasis`) or the metric perturbation vanishes ($\mathcal{G}_{ij} \to 0$). The condition $\mathcal{G}_{ij} \to 0$ implies $\nabla_{z^{(j)}}\nabla_{z^{(i)}} V^{(i)} \to 0$, which defines a region of **strategic decoupling**. $\square$

({prf:ref}`thm-gauge-covariant-klein-gordon`) *theorem* — **Gauge-Covariant Klein-Gordon Equation**

The Klein-Gordon equation for Value (Theorem {prf:ref}`thm-hjb-klein-gordon`) generalizes to the gauge-covariant form:

$$
\left(\frac{1}{c_{\text{info}}^2}D_t^2 - D^i D_i + \kappa^2\right)V^{(i)} = \rho_r^{(i)} + \sum_{j \neq i} \Phi_{ij}^{\text{ret}}

$$

where:
- $D_t = \partial_t - igA_0$ is the temporal covariant derivative
- $D_i = \partial_i - igA_i$ are spatial covariant derivatives
- $D^i = \tilde{G}^{ij}D_j$ with raised index via the strategic metric

$$
\Box_A := \frac{1}{c_{\text{info}}^2}D_t^2 - \tilde{G}^{ij}D_i D_j = \frac{1}{\sqrt{|\tilde{G}|}}D_\mu\left(\sqrt{|\tilde{G}|}\tilde{G}^{\mu\nu}D_\nu\right)

$$

The screening term $\kappa^2 V$ and source terms are gauge-invariant scalars. $\square$

({prf:ref}`thm-gauge-invariant-metric-inflation`) *theorem* — **Gauge-Invariant Metric Inflation**

The effective metric (Theorem {prf:ref}`thm-adversarial-mass-inflation`) generalizes to:

$$
\tilde{G}^{(i)}_{kl}(z) = G^{(i)}_{kl}(z) + \sum_{j \neq i} \beta_{ij} \text{Tr}\left[\tilde{\mathcal{G}}_{ij,kl}\right]

$$

where the trace projects onto the gauge-invariant component.


*Consequence:* The metric inflation experienced by agents is a **physical observable** independent of internal frame choice.

({prf:ref}`thm-curvature-commutator`) *theorem* — **Curvature from Covariant Derivative Commutator**

The field strength measures the failure of covariant derivatives to commute:

$$
[D_\mu, D_\nu]\psi = -ig\mathcal{F}_{\mu\nu}\psi

$$

$$
\begin{aligned}
[D_\mu, D_\nu]\psi &= D_\mu(D_\nu\psi) - D_\nu(D_\mu\psi) \\
&= (\partial_\mu - igA_\mu)(\partial_\nu\psi - igA_\nu\psi) - (\mu \leftrightarrow \nu) \\
&= \partial_\mu\partial_\nu\psi - ig(\partial_\mu A_\nu)\psi - igA_\nu\partial_\mu\psi - igA_\mu\partial_\nu\psi - g^2A_\mu A_\nu\psi - (\mu \leftrightarrow \nu) \\
&= -ig(\partial_\mu A_\nu - \partial_\nu A_\mu)\psi - g^2(A_\mu A_\nu - A_\nu A_\mu)\psi \\
&= -ig(\partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu])\psi \\
&= -ig\mathcal{F}_{\mu\nu}\psi \quad \square
\end{aligned}

$$

*Interpretation:* If $\mathcal{F}_{\mu\nu} \neq 0$, parallel transport around a closed loop results in a non-trivial rotation. The "meaning" of strategic nuisance **twists** as one navigates the latent space.

({prf:ref}`thm-bianchi-identity`) *theorem* — **Bianchi Identity**

The field strength satisfies the **Bianchi Identity**:

$$
D_\mu \mathcal{F}_{\nu\rho} + D_\nu \mathcal{F}_{\rho\mu} + D_\rho \mathcal{F}_{\mu\nu} = 0

$$

or in differential form notation: $D\mathcal{F} = 0$ where $D = d - ig[A, \cdot]$.

$$
[[D_\mu, D_\nu], D_\rho] + [[D_\nu, D_\rho], D_\mu] + [[D_\rho, D_\mu], D_\nu] = 0

$$

Since $[D_\mu, D_\nu] = -ig\mathcal{F}_{\mu\nu}$, this becomes:

$$
-ig([D_\rho, \mathcal{F}_{\mu\nu}] + \text{cyclic}) = 0

$$

The covariant derivative of $\mathcal{F}$ is $D_\rho\mathcal{F}_{\mu\nu} = \partial_\rho\mathcal{F}_{\mu\nu} - ig[A_\rho, \mathcal{F}_{\mu\nu}]$, and the identity follows. See **{ref}`Appendix E.17 <sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws>`** for the complete algebraic derivation with component verification. $\square$

*Interpretation:* The Bianchi identity is a **conservation law** for the strategic flux. It ensures topological consistency of the gauge structure.

({prf:ref}`thm-yang-mills-equations`) *theorem* — **Yang-Mills Field Equations**

The Euler-Lagrange equations for the Yang-Mills action yield:

$$
D_\mu \mathcal{F}^{\mu\nu} = J^\nu

$$

where the **strategic current** (source term) is:

$$
J^{\nu,a} = g\sum_{i=1}^N \bar{\psi}^{(i)}\gamma^\nu T^a \psi^{(i)}

$$

Here $\gamma^\nu$ are the Dirac matrices (or their appropriate generalization to curved space), and the sum is over all $N$ agents.

*Expanded form:*

$$
\partial_\mu \mathcal{F}^{\mu\nu,a} + gf^{abc}A_\mu^b\mathcal{F}^{\mu\nu,c} = J^{\nu,a}

$$

$$
\frac{\delta S}{\delta A_\mu^a} = 0 \implies -\frac{1}{g^2}\partial_\nu(\sqrt{|\tilde{G}|}\mathcal{F}^{\mu\nu,a}) + \frac{1}{g}f^{abc}A_\nu^b\mathcal{F}^{\mu\nu,c} + \frac{\delta S_{\text{matter}}}{\delta A_\mu^a} = 0

$$

The matter variation gives the current $J^{\mu,a}$, and reorganizing yields the Yang-Mills equation. $\square$

*Interpretation:* The gauge field is sourced by the strategic current—the flow of "charged" belief through latent space. Agents with non-zero internal state generate a gauge field that mediates their interaction with other agents.

({prf:ref}`thm-higgs-mechanism`) *theorem* — **Spontaneous Symmetry Breaking (Higgs Mechanism)**

When the Higgs mass parameter satisfies $\mu^2 < 0$, the potential $V(\Phi)$ has a non-trivial minimum, and the gauge symmetry is **spontaneously broken**.

**Vacuum Expectation Value:**

$$
\langle\Phi\rangle = \frac{v}{\sqrt{2}}, \quad v = \sqrt{-\mu^2/\lambda}

$$

**Mass Generation:**

1. **Gauge boson masses:** The gauge fields acquire mass

   $$
   m_A = \frac{gv}{2}

   $$
   transforming from massless to massive (strategic inertia).

2. **Fermion masses:** The belief spinors acquire effective mass

   $$
   m_{\text{eff},i} = \frac{y_{ii}v}{\sqrt{2}}

   $$
   through Yukawa coupling.

3. **Residual symmetry:** The full gauge group $G$ breaks to a subgroup $H \subset G$ that leaves the vacuum invariant.

$$
|D_\mu\Phi|^2 = \frac{1}{2}(\partial_\mu h)^2 + \frac{g^2v^2}{4}A_\mu A^\mu + \ldots

$$

The term $\frac{g^2v^2}{4}A_\mu A^\mu$ is a mass term for $A_\mu$ with $m_A^2 = g^2v^2/4$. Similarly, the Yukawa term generates fermion masses. See **{ref}`Appendix E.18 <sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws>`** for the complete derivation including VEV calculation, Goldstone absorption, and the symmetry breaking pattern. $\square$

*Interpretation:* Policy selection (choosing a direction in latent space) is spontaneous symmetry breaking. The agent commits to a strategy, breaking the rotational invariance of the Semantic Vacuum. This commitment generates "mass"—resistance to changing strategy.

({prf:ref}`thm-mass-gap-screening`) *theorem* — **Mass Gap from Screening**

The screening mass $\kappa = \lambda / c_{\text{info}}$ with $\lambda = -\ln\gamma / \Delta t$ from the Helmholtz equation (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`) provides a lower bound on the mass gap (natural units: $\kappa = -\ln\gamma$):

$$
\Delta \geq \frac{\kappa^2}{2m_{\text{eff}}}

$$

where $m_{\text{eff}}$ is the effective inertia from Game Tensor inflation (Theorem {prf:ref}`thm-adversarial-mass-inflation`).

({prf:ref}`thm-computational-necessity-mass-gap`) *theorem* — **Computational Necessity of Mass Gap**

**Assumptions:**
1. The system satisfies the **Causal Information Bound** (Theorem {prf:ref}`thm-causal-information-bound`): $I_{\text{bulk}}(V) \leq \nu_D \cdot \text{Area}(\partial V) / \ell_L^{D-1}$
2. The system has finite spatial extent (bounded region $V$)
3. Correlations follow the standard field-theoretic decay: massive $\sim e^{-\kappa r}$, massless $\sim 1/r^{D-2}$

**Statement:** Under these assumptions, a system with $\Delta = 0$ enters Causal Stasis ($\|v\|_G = 0$).

1. **Assume gapless theory:** Suppose $\Delta = 0$, so the lowest excitation above the vacuum is massless.

2. **Infinite correlation length:** The screening mass $\kappa = 0$ implies the correlation length diverges:

   $$
   \xi = \frac{1}{\kappa} \to \infty

   $$

3. **Divergent information volume:** For massless correlations decaying as $1/r^{D-2}$ (rather than $e^{-\kappa r}$ for massive), the integrated mutual information in a volume $V$ diverges:

   $$
   I_{\text{bulk}} \propto \int_V \text{Corr}(x, y)\,dV \to \infty

   $$

4. **Area law violation:** By Assumption 1 (Causal Information Bound):

   $$
   I_{\text{bulk}} \leq \nu_D \cdot \frac{\text{Area}(\partial V)}{\ell_L^{D-1}}

   $$
   A bounded system cannot store infinite information, so the bound is saturated.

5. **Causal Stasis:** By Theorem 33.4 (Causal Stasis), as $I_{\text{bulk}}$ saturates the bound, the metric component $G_{rr} \to \infty$ and the update velocity $\|v\|_G \to 0$.

*Conclusion:* Under the stated assumptions, a gapless theory ($\Delta = 0$) implies frozen dynamics. For temporal evolution to occur, correlations must be screened: $\xi < \infty \implies \Delta > 0$. $\square$

*Remark (Scope of Assumptions):* Assumption 1 is derived in Theorem 33.3 from first principles (the Levin complexity bound). For systems satisfying this bound—which includes all physically realizable computational systems—the mass gap necessity follows.

({prf:ref}`thm-mass-gap-constructive`) *theorem* — **Mass Gap by Constructive Necessity**

**Prerequisites:**
1. The system satisfies the Causal Information Bound (Theorem 33.3)
2. The system is **non-trivial**: has non-zero update velocity $\|v\|_G > 0$ at some time
3. The system is **interacting**: coupling constants $\Phi_{ij} \neq 0$ or $\mathcal{G}_{ij} \neq 0$

**Statement:** Under these assumptions, $\Delta > 0$.

Suppose $\Delta = 0$. By Theorem {prf:ref}`thm-computational-necessity-mass-gap` (using Assumption 1), the system enters Causal Stasis with $\|v\|_G = 0$. This contradicts Assumption 2 (non-triviality).

Therefore $\Delta > 0$ for any non-trivial theory describing an evolving system that satisfies the Causal Information Bound. $\square$

*Bound:* The mass gap is bounded below by thermodynamic considerations:

$$
\Delta \geq \frac{1}{\beta}\left(\Delta H + \frac{\mathcal{W}}{T_c}\right)

$$
where $\Delta H$ is the enthalpy barrier for excitation, $\mathcal{W}$ is computational work, and $T_c$ is cognitive temperature. This follows from Theorem 30.15 (Thermodynamic Hysteresis).

*Remark (Conditional vs. Absolute):* This theorem does **not** prove that all field theories have a mass gap. It proves: IF a system satisfies the Causal Information Bound AND evolves non-trivially, THEN it must have $\Delta > 0$. The Clay Millennium Problem asks whether quantum Yang-Mills in continuous $\mathbb{R}^4$ has a mass gap; this framework addresses discrete, bounded, computational systems.

({prf:ref}`thm-cft-swampland`) *theorem* — **CFT Swampland Classification**

Let $\mathcal{T}$ be a Conformal Field Theory on $\mathbb{R}^d$ ($d \geq 2$) with at least one primary operator of scaling dimension $\Delta_\phi < d/2$. Then $\mathcal{T}$ lies in the **Computational Swampland** (Definition {prf:ref}`def-computational-swampland`).

1. **Infinite correlation length:** By conformal symmetry, two-point correlations decay algebraically:

   $$
   \langle \phi(x) \phi(0) \rangle \sim \frac{1}{|x|^{2\Delta_\phi}}

   $$
   The correlation length is $\xi = \infty$ (no exponential screening).

2. **Bulk information divergence:** Consider a spherical region $V$ of radius $R$. The mutual information between bulk degrees of freedom is bounded below by the integrated correlation:

   $$
   I_{\text{bulk}}(V) \gtrsim \int_V \int_V \frac{dx\,dy}{|x-y|^{2\Delta_\phi}} \sim R^{2d - 2\Delta_\phi}

   $$
   For $\Delta_\phi < d/2$, the exponent $2d - 2\Delta_\phi > d$, so $I_{\text{bulk}}$ grows faster than volume.

3. **Causal Information Bound violation:** The boundary capacity scales as:

   $$
   C_\partial(V) = \nu_d \cdot \frac{\text{Area}(\partial V)}{\ell_L^{d-1}} \sim R^{d-1}

   $$
   where $\nu_d$ is the Holographic Coefficient (Definition {prf:ref}`def-holographic-coefficient`). Since $2d - 2\Delta_\phi > d > d-1$ for $d \geq 2$ and $\Delta_\phi < d/2$, there exists $R_c$ such that for all $R > R_c$:

   $$
   I_{\text{bulk}}(V) > C_\partial(V)

   $$
   The Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`) is violated.

4. **Swampland membership:** By Definition {prf:ref}`def-computational-swampland`, theories violating the Causal Information Bound at any finite scale lie in the Swampland. $\square$

*Remark (Universal bound violation).* The theorem requires at least one operator with $\Delta_\phi < d/2$. In any non-trivial CFT, such operators exist: for instance, the stress-energy tensor has $\Delta = d$, but scalar primary operators generically have $\Delta < d/2$ in unitary CFTs (e.g., the $\phi$ field in the free scalar CFT has $\Delta = (d-2)/2 < d/2$ for $d > 2$). More fundamentally, the mutual information between any two regions in a CFT diverges logarithmically due to UV contributions, independent of operator dimensions. The bound is therefore violated by all CFTs in $d \geq 2$.

*Remark (Operational meaning).* A bounded observer with finite interface capacity $C_\partial$ cannot encode the full correlational structure of a CFT. Any finite approximation necessarily introduces an effective mass gap via truncation.

({prf:ref}`thm-scale-covariance-bound`) *theorem* — **Scale Covariance of the Causal Information Bound**

The Causal Information Bound is preserved under coarse-graining. Specifically:

Let $(\mathcal{Z}, G, \ell_L)$ be a latent manifold at resolution $\ell_L$ satisfying $I_{\text{bulk}} \leq C_\partial$. Under coarse-graining to resolution $\ell'_L = \alpha \ell_L$ ($\alpha > 1$), the coarse-grained system $(\mathcal{Z}', G', \ell'_L)$ satisfies:

$$
I'_{\text{bulk}} \leq C'_\partial

$$

1. **Information reduction:** By the Data Processing Inequality, coarse-graining cannot increase mutual information:

   $$
   I'_{\text{bulk}} \leq I_{\text{bulk}}

   $$

2. **Capacity reduction:** Under coarse-graining by factor $\alpha$, the effective boundary area scales as:

   $$
   \text{Area}'(\partial\mathcal{Z}') \sim \frac{\text{Area}(\partial\mathcal{Z})}{\alpha^{d-1}}

   $$
   and the new capacity is (using the generalized bound with $\nu_d$):

   $$
   C'_\partial = \nu_d \cdot \frac{\text{Area}'}{(\ell'_L)^{d-1}} = \nu_d \cdot \frac{\text{Area}/\alpha^{d-1}}{\alpha^{d-1}\ell_L^{d-1}} = \frac{C_\partial}{\alpha^{2(d-1)}}

   $$

3. **Bound preservation:** The information-to-capacity ratio under coarse-graining:

   $$
   \frac{I'_{\text{bulk}}}{C'_\partial} \leq \frac{I_{\text{bulk}}}{C_\partial/\alpha^{2(d-1)}} = \alpha^{2(d-1)} \frac{I_{\text{bulk}}}{C_\partial}

   $$
   For massive theories (exponentially decaying correlations), $I_{\text{bulk}}$ scales as area, so $I_{\text{bulk}}/C_\partial$ is scale-independent. For gapless theories, the ratio diverges—confirming they violate the bound at some scale. $\square$

*Implication (UV finiteness).* The recursive self-consistency of the bound at all scales implies that no UV divergences arise. The Levin Length $\ell_L$ acts as a natural UV cutoff that is preserved under renormalization group flow. Unlike lattice regularization where the continuum limit requires careful tuning, this framework has built-in regularization.

*Implication (Mass gap from scale invariance).* The only scale-invariant theories consistent with the Causal Information Bound are those with $I_{\text{bulk}} \sim R^{d-1}$ (area scaling). This requires exponential correlation decay, hence $\Delta > 0$. Theories with algebraic correlation decay (CFTs) fail scale covariance of the bound.

({prf:ref}`thm-mass-gap-dichotomy`) *theorem* — **Mass Gap Dichotomy for Yang-Mills**

Let $\mathcal{T}_{\text{YM}}$ be Yang-Mills theory with compact simple gauge group $G$ in $d = 4$ dimensions.

**Statement:** If $\mathcal{T}_{\text{YM}}$ describes physics (is realizable by bounded observers), then $\Delta > 0$.

1. **Framework implements Yang-Mills:** The Fragile Agent framework implements Yang-Mills field equations (Theorem {prf:ref}`thm-yang-mills-equations`) with the standard action (Definition {prf:ref}`def-yang-mills-action`), covariant derivatives $D_\mu = \partial_\mu - igA_\mu$, and non-Abelian field strength tensor. This is not an analogy—it is Yang-Mills theory for information systems.

2. **Physical theories are computable:** Any theory describing physics accessible to bounded observers must be realizable with finite resources. This requires Levin Length $\ell_L > 0$ (Definition {prf:ref}`def-levin-length`).

3. **Computability implies mass gap:** By Theorem {prf:ref}`thm-computational-necessity-mass-gap`, any theory with $\ell_L > 0$ and non-trivial dynamics ($\|v\|_G > 0$) has $\Delta > 0$.

4. **Conclusion:** If Yang-Mills describes physics, it is computable, hence has $\ell_L > 0$, hence has $\Delta > 0$. $\square$

*Remark (Contrapositive).* If Yang-Mills on $\mathbb{R}^4$ requires $\ell_L \to 0$ (no UV cutoff), then by Theorem {prf:ref}`thm-cft-swampland` it lies in the Computational Swampland and does not describe physics. Either way, the physical theory has a mass gap.

*Remark (Why this is not circular).* The mass gap necessity follows from information-theoretic constraints (the Causal Information Bound), not from assuming properties of Yang-Mills. The framework proves that **any** non-trivial gauge theory satisfying the bound has $\Delta > 0$. Yang-Mills is one such theory.

({prf:ref}`thm-madelung-transform`) *theorem* — **The Madelung Transform (WFR-Schrödinger Equivalence)**

Let the belief density $\rho$ and value $V$ satisfy the WFR-HJB system:
1. **WFR Continuity (unbalanced):** $\partial_s \rho + \nabla_G \cdot (\rho \mathbf{v}) = \rho r$
2. **Hamilton-Jacobi-Bellman:** $\partial_s V + \frac{1}{2}\|\nabla_G V\|_G^2 + \Phi_{\text{eff}} = 0$

where $\mathbf{v} = -G^{-1}\nabla V$ is the gradient flow velocity and $r$ is the WFR reaction rate (Definition {prf:ref}`def-the-wfr-action`).

Then the belief wave-function $\psi = \sqrt{\rho} e^{iV/\sigma}$ satisfies the **Inference Schrödinger Equation**:

$$
i\sigma \frac{\partial \psi}{\partial s} = \hat{H}_{\text{inf}} \psi,

$$
where the **Inference Hamiltonian** is:

$$
\hat{H}_{\text{inf}} := -\frac{\sigma^2}{2} \Delta_G + \Phi_{\text{eff}} + Q_B - \frac{i\sigma}{2} r.

$$
The terms are:
- **Kinetic:** $-\frac{\sigma^2}{2} \Delta_G$ (belief diffusion via Laplace-Beltrami)
- **Potential:** $\Phi_{\text{eff}}$ (effective potential from rewards and constraints)
- **Quantum Correction:** $Q_B$ (Bohm potential, Definition {prf:ref}`def-bohm-quantum-potential`)
- **Dissipation:** $-\frac{i\sigma}{2} r$ (non-Hermitian term from WFR reaction)

**Step 1 (Substitution).** Write $\psi = R e^{i\phi}$ with $R = \sqrt{\rho}$ and $\phi = V/\sigma$.

**Step 2 (Time derivative).**

$$
i\sigma \partial_s \psi = i\sigma \left( \frac{\partial_s R}{R} + \frac{i}{\sigma}\partial_s V \right) \psi = \left( \frac{i\sigma \partial_s \rho}{2\rho} - \partial_s V \right) \psi.

$$
**Step 3 (Use governing equations).** Substitute the continuity equation for $\partial_s \rho$ and HJB for $\partial_s V$.

**Step 4 (Identify terms).** The real part of the resulting equation gives the HJB with Bohm correction; the imaginary part gives the continuity equation with reaction. Combining yields the Schrödinger form. $\square$

({prf:ref}`thm-multi-agent-schrodinger-equation`) *theorem* — **Multi-Agent Schrödinger Equation**

The joint belief wave-function $\Psi(\mathbf{z}, s)$ of $N$ strategically coupled agents evolves according to:

$$
i\sigma \frac{\partial \Psi}{\partial s} = \hat{H}_{\text{strat}} \Psi + i\frac{\sigma}{2} \mathcal{R} \Psi,

$$
where:
- $\hat{H}_{\text{strat}}$ is the Strategic Hamiltonian (Definition {prf:ref}`def-strategic-hamiltonian`)
- $\mathcal{R}(\mathbf{z}) = \sum_i r^{(i)}(z^{(i)})$ is the total reaction rate

**Expanded form:**

$$
i\sigma \frac{\partial \Psi}{\partial s} = \left[ \sum_{i=1}^N \left( -\frac{\sigma_i^2}{2} \Delta_{G^{(i)}} + \Phi^{(i)}_{\text{eff}} \right) + \sum_{i < j} \Phi_{ij} \right] \Psi + i\frac{\sigma}{2} \mathcal{R} \Psi.

$$
**Sources of entanglement:** Strategic entanglement arises from:
1. **Potential coupling:** Non-zero $\Phi_{ij}(z^{(i)}, z^{(j)})$ creates position-position correlations
2. **Metric coupling:** The Game Tensor $\mathcal{G}_{ij}$ modifies the kinetic terms (Theorem {prf:ref}`thm-game-augmented-laplacian`)

*Cross-reference:* This extends Theorem {prf:ref}`thm-madelung-transform` to multiple agents, with the joint WFR dynamics (Definition {prf:ref}`def-joint-wfr-action`) as the underlying classical limit.

({prf:ref}`thm-game-augmented-laplacian`) *theorem* — **Game-Augmented Laplacian**

Under adversarial coupling, the effective kinetic operator for agent $i$ incorporates the **Game Tensor** (Definition {prf:ref}`def-the-game-tensor`):

$$
\hat{H}^{(i)}_{\text{kin,eff}} = -\frac{\sigma_i^2}{2} \tilde{\Delta}^{(i)},

$$
where the **Game-Augmented Laplacian** is:

$$
\tilde{\Delta}^{(i)} := \frac{1}{\sqrt{|\tilde{G}^{(i)}|}} \partial_a \left( \sqrt{|\tilde{G}^{(i)}|} (\tilde{G}^{(i)})^{ab} \partial_b \right),

$$
with strategic metric $\tilde{G}^{(i)} = G^{(i)} + \sum_{j \neq i} \beta_{ij} \mathcal{G}_{ij}$ (Definition {prf:ref}`def-the-game-tensor`, Equation 29.4.1).

**Consequence for entanglement:** Since $\tilde{G}^{(i)}$ depends on $z^{(j)}$ through the Game Tensor, the kinetic operator for agent $i$ is **not separable**:

$$
\tilde{\Delta}^{(i)} = \tilde{\Delta}^{(i)}(z^{(i)}; z^{(-i)}).

$$
This creates **kinetic entanglement**—even without potential coupling, adversarial metric inflation entangles the agents.

*Physical interpretation:* Agent $j$ "curves" agent $i$'s configuration space. Moving through a contested region requires more "effort" (higher effective mass), and this coupling cannot be factorized away.

({prf:ref}`thm-nash-ground-state`) *theorem* — **Nash Equilibrium as Ground State**

A Nash equilibrium $\mathbf{z}^* = (z^{(1)*}, \ldots, z^{(N)*})$ (Theorem {prf:ref}`thm-nash-equilibrium-as-geometric-stasis`) corresponds to the **ground state** of the Strategic Hamiltonian:

1. **Spectral condition:** The ground state $\Psi_{\text{Nash}}$ satisfies:

   $$
   \hat{H}_{\text{strat}} \Psi_{\text{Nash}} = E_0 \Psi_{\text{Nash}}, \quad E_0 = \min \text{spec}(\hat{H}_{\text{strat}}).

   $$
2. **Localization:** In the semiclassical limit ($\sigma \to 0$), $|\Psi_{\text{Nash}}|^2$ concentrates near $\mathbf{z}^*$:

   $$
   \lim_{\sigma \to 0} |\Psi_{\text{Nash}}(\mathbf{z})|^2 = \delta(\mathbf{z} - \mathbf{z}^*).

   $$
3. **Energy interpretation:** The ground state energy $E_0$ equals the total effective potential at Nash:

   $$
   E_0 = \sum_{i=1}^N \Phi^{(i)}_{\text{eff}}(\mathbf{z}^*) + \sum_{i < j} \Phi_{ij}(\mathbf{z}^*) + O(\sigma).

   $$
See **{ref}`Appendix E.19 <sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws>`** for the complete WKB/semiclassical analysis proving Gaussian concentration to delta function as $\sigma \to 0$, with explicit energy correction formulas. $\square$

*Remark (Multiple Nash).* If multiple Nash equilibria exist, each corresponds to a different local minimum of the energy landscape. The **global** ground state is the Nash with lowest $E_0$; other Nash equilibria are metastable excited states.

({prf:ref}`thm-tunneling-probability`) *theorem* — **Strategic Tunneling Probability (WKB Approximation)**

In the semiclassical limit ($\sigma \ll \Delta \Phi_P$), the probability of crossing a Pareto barrier is:

$$
P_{\text{tunnel}} \sim \exp\left(-\frac{2}{\sigma} \int_{\gamma} \sqrt{2(\Phi_{\text{eff,total}}(\mathbf{z}) - E_0)}\, d\ell_{G^{(N)}}\right),

$$
where:
- $\gamma$ is the **optimal tunneling path** (instanton) connecting $\mathbf{z}^*_A$ to $\mathbf{z}^*_B$
- $\Phi_{\text{eff,total}} = \sum_i \Phi^{(i)}_{\text{eff}} + \sum_{i<j} \Phi_{ij}$
- $d\ell_{G^{(N)}}$ is the geodesic arc length on $(\mathcal{Z}^{(N)}, G^{(N)})$
- $E_0$ is the ground state energy

**Key scaling:** $P_{\text{tunnel}} \propto e^{-\Delta \Phi_P / \sigma}$, so higher barriers or lower temperature (small $\sigma$) exponentially suppress tunneling.

*Cross-reference:* This generalizes Theorem {prf:ref}`thm-memory-induced-barrier-crossing` from single-agent memory barriers to multi-agent Pareto barriers. See {ref}`Appendix E.7 <sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws>` for the rigorous proof via Agmon estimates and spectral theory.

({prf:ref}`thm-emergence-opportunity-field`) *theorem* — **Emergence of the Opportunity Field ($B_\mu$)**

To preserve the invariance of the kinetic term in the Inference Action under the local transformation $\psi \to e^{i\theta(x)}\psi$, we must replace the partial derivative $\partial_\mu$ with the **Covariant Derivative**:

$$
D_\mu = \partial_\mu - i g_1 \frac{Y}{2} B_\mu,

$$

where:
- $Y$ is the **Hypercharge** (the reward sensitivity of the module)
- $B_\mu$ is an abelian gauge field (the **Opportunity Field**)
- $g_1$ is the coupling constant

**Step 1.** Consider the kinetic term from the Inference Schrödinger Equation (Theorem {prf:ref}`thm-madelung-transform`):

$$
\mathcal{L}_{\text{kin}} = \psi^* (i\sigma \partial_t) \psi - \frac{\sigma^2}{2}|\nabla \psi|^2.

$$

Under local transformation $\psi \to e^{i\theta(x)}\psi$:

$$
\partial_\mu \psi \to e^{i\theta}(\partial_\mu \psi + i(\partial_\mu\theta)\psi).

$$

The kinetic term acquires a spurious contribution $\sigma(\partial_\mu\theta)|\psi|^2$ that depends on the arbitrary function $\theta(x)$.

**Step 2.** Introduce the compensating field $B_\mu$ transforming as:

$$
B_\mu \to B_\mu + \frac{2}{g_1 Y} \partial_\mu \theta(x).

$$

**Step 3.** The covariant derivative $D_\mu \psi = (\partial_\mu - ig_1(Y/2)B_\mu)\psi$ transforms homogeneously:

$$
D_\mu \psi \to e^{i\theta(x)} D_\mu \psi.

$$

**Step 4.** The gauge-invariant kinetic term is $(D_\mu\psi)^\dagger(D^\mu\psi) = |D_\mu\psi|^2$.

**Identification:** The field $B_\mu$ is the $U(1)$ connection associated with the reward 1-form (the Opportunity Field).
In the conservative case, $B_\mu = \partial_\mu \Phi$ is pure gauge. On each time slice, the spatial components $\vec{B}$
admit a Hodge decomposition into gradient (conservative) plus solenoidal/harmonic parts (path-dependent opportunity).

The field strength tensor $B_{\mu\nu} = \partial_\mu B_\nu - \partial_\nu B_\mu$ measures the non-conservative component
of the reward 1-form (Value Curl; Definition {prf:ref}`def-value-curl`). When $B_{\mu\nu} \neq 0$, no choice of baseline
can make the reward landscape path-independent.

$\square$

({prf:ref}`thm-emergence-error-field`) *theorem* — **Emergence of the Error Field ($W_\mu^a$)**

The process of **Belief Update** (e.g., Kalman Filtering or Predictive Coding) corresponds to a rotation in Isospin space. Gauging this symmetry requires the introduction of non-Abelian gauge fields.

**Step 1.** A Bayesian update mixes the Prior and the Likelihood:

$$
\Psi_L' = U(x) \Psi_L, \quad U(x) = \exp\left( i \frac{\vec{\tau} \cdot \vec{\theta}(x)}{2} \right) \in SU(2)

$$

where $\vec{\tau} = (\tau_1, \tau_2, \tau_3)$ are the Pauli matrices and $\vec{\theta}(x)$ determines the mixing angle (the Kalman Gain in standard filtering).

**Step 2.** For **Local Covariance** (the ability to perform updates locally without global synchronization), we introduce the non-Abelian gauge field $\vec{W}_\mu = (W^1_\mu, W^2_\mu, W^3_\mu)$.

**Step 3.** The covariant derivative for the Left-Handed sector is:

$$
D_\mu \Psi_L = \left( \partial_\mu - i g_2 \frac{\vec{\tau}}{2} \cdot \vec{W}_\mu - i g_1 \frac{Y_L}{2} B_\mu \right) \Psi_L

$$

**Step 4.** The gauge field transforms as:

$$
W_\mu^a \to W_\mu^a + \frac{1}{g_2}\partial_\mu \theta^a + \epsilon^{abc}\theta^b W_\mu^c

$$

to maintain covariance.

**Identification:**
- The $W^\pm_\mu = (W^1_\mu \mp iW^2_\mu)/\sqrt{2}$ bosons mediate transitions between $\psi_{\text{pred}}$ and $\psi_{\text{obs}}$. These correspond to belief updates where prediction and observation exchange weight.
- The $W^3_\mu$ component mixes with $B_\mu$ after symmetry breaking ({ref}`Section 34.3 <sec-scalar-sector-symmetry-breaking>`).
- The $SU(2)_L$ gauge symmetry acts only on the input channel ($\Psi_L$), leaving the output singlet ($\Psi_R$) invariant. This reflects the architectural asymmetry between perception and action.

$\square$

({prf:ref}`thm-emergence-binding-field`) *theorem* — **Emergence of the Binding Field ($G_\mu^a$)**

To gauge the $SU(N_f)$ feature symmetry, we introduce the **Gluon Field** $G_\mu^a$ ($a=1,\dots,N_f^2-1$).

**Step 1.** The covariant derivative for feature fields is:

$$
D_\mu \psi = \left( \partial_\mu - i g_s \frac{\lambda^a}{2} G_\mu^a \right) \psi

$$

where $\lambda^a$ ($a = 1, \ldots, N_f^2 - 1$) are the generalized Gell-Mann matrices (generators of $SU(N_f)$), satisfying $\text{Tr}(\lambda^a \lambda^b) = 2\delta^{ab}$ and $[\lambda^a, \lambda^b] = 2i f^{abc} \lambda^c$.

**Step 2.** The field strength tensor is:

$$
G_{\mu\nu}^a = \partial_\mu G_\nu^a - \partial_\nu G_\mu^a + g_s f^{abc} G_\mu^b G_\nu^c

$$

where $f^{abc}$ are the structure constants of $SU(N_f)$, defined by $[\lambda^a, \lambda^b] = 2i f^{abc} \lambda^c$.

**Step 3.** The non-Abelian structure implies **self-interaction** of the gluon field. For $SU(N_f)$ with $N_f \geq 2$, the beta function $\beta(g_s) < 0$ yields:

- **Asymptotic Freedom:** At small distances in the latent manifold (high RG scale $\tau$, deep in the TopoEncoder hierarchy), the effective coupling $g_s(\tau)$ decreases. Individual features can be resolved.

- **Infrared Confinement:** At large distances (low RG scale, coarse representations), the effective coupling grows. Features cannot propagate independently; they form bound states (concepts $K$).

*Remark:* The sign of the beta function is not universal for all gauge theories; it depends on the matter
content and representations coupled to the field. In the Fragile Mechanics binding sector postulated
here, the gauge field couples to the cognitive spinor (fundamental "color" index) while the ontological
scalar is color-neutral as written. With only $O(1)$ fundamental fermion species, the one-loop
coefficient is in the asymptotically-free regime, so $\beta(g_s) < 0$ at weak coupling. Confinement/binding
at coarse scales is then justified by the Area-Law/observability result (Theorem {prf:ref}`thm-fission-inhibition`
/ Section 33), rather than asserted as a universal consequence of $SU(N_f)$ alone.

**Step 4.** From Theorem {prf:ref}`thm-fission-inhibition`, the energy cost of separating features grows linearly with distance (Area Law, {ref}`Section 33 <sec-causal-information-bound>`). Attempting to isolate a feature instead triggers Ontological Fission (Definition {prf:ref}`def-query-fission`), creating new concept pairs.

$\square$

({prf:ref}`thm-three-cognitive-forces`) *theorem* — **Field Strength Tensors**

The commutator of the covariant derivatives $[D_\mu, D_\nu]$ generates three distinct curvature tensors corresponding to each gauge factor.

1. **$U(1)_Y$ Curvature:**

   $$
   B_{\mu\nu} = \partial_\mu B_\nu - \partial_\nu B_\mu

   $$
   When $B_{\mu\nu} \neq 0$, the reward field is non-conservative (Definition {prf:ref}`def-conservative-reward-field`). The resulting Lorentz-type force generates cyclic dynamics.

2. **$SU(2)_L$ Curvature:**

   $$
   W_{\mu\nu}^a = \partial_\mu W_\nu^a - \partial_\nu W_\mu^a + g_2 \epsilon^{abc} W_\mu^b W_\nu^c

   $$
   When $W_{\mu\nu} \neq 0$, the belief update depends on the path taken in the manifold: parallel transport around a closed loop yields a non-trivial rotation in the prediction-observation space.

3. **$SU(N_f)_C$ Curvature:**

   $$
   G_{\mu\nu}^a = \partial_\mu G_\nu^a - \partial_\nu G_\mu^a + g_s f^{abc} G_\mu^b G_\nu^c

   $$
   When $G_{\mu\nu} \neq 0$, the feature binding is under stress. This corresponds to the Ontological Stress $\Xi$ (Definition {prf:ref}`def-ontological-stress`). When $\Xi > \Xi_{\text{crit}}$, chart fission is triggered ({ref}`Section 30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`).

$\square$

({prf:ref}`thm-complexity-potential`) *theorem* — **The Complexity Potential**

The Lagrangian density for the scalar field is uniquely determined by the **Supercritical Pitchfork Bifurcation** (Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`).

**Step 1.** From Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`, the radial evolution of chart separation satisfies:

$$
\frac{dr}{ds} = (\Xi - \Xi_{\text{crit}})r - \alpha r^3

$$

where:
- $\Xi$ is the Ontological Stress (Definition {prf:ref}`def-ontological-stress`)
- $\Xi_{\text{crit}}$ is the critical threshold (Theorem {prf:ref}`thm-fission-criterion`)
- $\alpha > 0$ is the stabilizing cubic coefficient

**Step 2.** This flow is the gradient descent of a potential function $\mathcal{V}_{\text{onto}}(r)$ such that $\dot{r} = -\partial \mathcal{V}_{\text{onto}}/\partial r$. Integrating:

$$
\mathcal{V}_{\text{onto}}(\phi) = -\frac{(\Xi - \Xi_{\text{crit}})}{2} |\phi|^2 + \frac{\alpha}{4} |\phi|^4

$$

**Step 3.** Define the standard Higgs potential parameters by matching coefficients:
- $\mu^2 \equiv \frac{(\Xi - \Xi_{\text{crit}})}{2}$: The effective **Mass Parameter** driven by Ontological Stress
- $\lambda \equiv \frac{\alpha}{4}$: The **Self-Interaction** coefficient from router saturation (Axiom {prf:ref}`ax-ontological-expansion-principle`)

**Step 4.** The potential takes the Landau-Ginzburg form:

$$
\mathcal{V}_{\text{onto}}(\phi) = -\mu^2 |\phi|^2 + \lambda |\phi|^4

$$

**Term Identification:**
- **Term 1 ($-\mu^2 |\phi|^2$):** Rewards separation. If Stress $\Xi > \Xi_{\text{crit}}$, this term drives $|\phi|$ away from zero to capture predictive information.
- **Term 2 ($+\lambda |\phi|^4$):** Penalizes complexity. Keeping charts separate costs compute/memory. This term prevents infinite fragmentation.

$\square$

({prf:ref}`thm-semantic-inertia`) *theorem* — **Generation of Semantic Inertia**

The kinetic term of the scalar field in the Lagrangian is covariant:

$$
\mathcal{L}_{\text{Kinetic}} = (D_\mu \phi)^\dagger (D_\mu \phi)

$$

where $D_\mu = \partial_\mu - ig \mathcal{A}_\mu$ includes the Strategic Connection.

**Step 1.** In the Broken Phase, expand around the vacuum expectation: $\phi(x) = v + h(x)$, where $h$ is the fluctuation (the physical Higgs mode).

**Step 2.** The kinetic term generates a quadratic interaction:

$$
|D_\mu v|^2 = |(-ig \mathcal{A}_\mu) v|^2 = g^2 v^2 \mathcal{A}_\mu \mathcal{A}^\mu

$$

**Step 3.** This is a **Mass Term** for the Gauge Field:

$$
M_{\mathcal{A}} = g v = g \sqrt{\frac{\Xi - \Xi_{\text{crit}}}{\alpha}}

$$

**Step 4.** Connection to Theorem {prf:ref}`thm-capacity-constrained-metric-law`: The mass $M_{\mathcal{A}}$ corresponds to an increase in the effective metric eigenvalues. From the Capacity-Constrained Metric Law, higher information density (more distinct concepts, larger $v$) induces higher curvature, which manifests as increased "inertia" in the metric.

**Physical Consequences:**

1. **Massless Phase ($v=0$):** The gauge fields are massless. The interaction potential decays as $1/r$ (long-range). Frame transformations between charts have zero energy cost.

2. **Massive Phase ($v > 0$):** The gauge fields acquire mass $M_{\mathcal{A}}$. The interaction potential becomes $e^{-M_{\mathcal{A}}r}/r$ (Yukawa, short-range). Gauge rotations---reinterpreting the meaning of signals---require energy proportional to $M_{\mathcal{A}}$. The ontological structure becomes stable against small perturbations.

$\square$

({prf:ref}`thm-cognitive-mass`) *theorem* — **Generation of Cognitive Mass (Decision Stability)**

In the **Broken Phase** ($\Xi > \Xi_{\text{crit}}$), the Yukawa coupling generates mass for the belief spinor.

**Step 1.** The scalar field acquires VEV $\langle \phi \rangle = v$ (Corollary {prf:ref}`cor-ontological-ssb`).

**Step 2.** Expanding the Lagrangian around the vacuum $\phi = v + h$:

$$
\mathcal{L}_{\text{Yukawa}} = -\underbrace{(Y v)}_{\text{Mass}} \bar{\psi} \psi - \underbrace{Y h \bar{\psi} \psi}_{\text{Higgs Interaction}}

$$

**Step 3.** The belief spinor $\psi$ acquires effective mass $m_\psi = Y v$.

**Consequences:**

1. **Symmetric Phase ($v=0$):** Mass is zero. Beliefs obey the massless equation $i\gamma^\mu \partial_\mu \psi = 0$ and propagate at speed $c_{\text{info}}$. The belief-action coupling vanishes; there is no stable commitment to action.

2. **Broken Phase ($v > 0$):** Mass is non-zero. Beliefs obey $(i\gamma^\mu \partial_\mu - m_\psi)\psi = 0$. The mass term $m_\psi = Yv$ provides inertia: a finite force (prediction error) is required to change the belief state. Larger ontological separation $v$ implies larger mass.

$\square$

({prf:ref}`thm-recovery-wfr-drift`) *theorem* — **Recovery of WFR Drift**

Varying the total action yields the Dirac equation with potential. In the non-relativistic limit, this recovers the WFR drift.

**Step 1.** The Euler-Lagrange equation from $\mathcal{S} = \int (\bar{\Psi} i \gamma^\mu \partial_\mu \Psi - \mathcal{L}_{\text{Drive}}) d^4x$ yields:

$$
(i \gamma^\mu \partial_\mu - \Phi_{\text{eff}})\Psi = 0

$$

**Step 2.** Apply the inverse Madelung transform (Theorem {prf:ref}`thm-madelung-transform`). In the non-relativistic limit ($c_{\text{info}} \to \infty$), the Schrödinger reduction recovers:

$$
\vec{v} \approx -\nabla \Phi_{\text{eff}}

$$

This is the WFR drift velocity from Definition {prf:ref}`def-bulk-drift-continuous-flow`.

*Remark.* The external field term $\mathcal{L}_{\text{Drive}}$ breaks the symmetry under time translation (via the discount factor in $\Phi_{\text{eff}}$) and generates directed flow toward regions of high value.

$\square$

({prf:ref}`thm-speed-window`) *theorem* — **The Speed Window**

The information speed $c_{\text{info}}$ must satisfy the **Speed Window Inequality**:

$$
\frac{d_{\text{sync}}}{\tau_{\text{proc}}} \le c_{\text{info}} \le \frac{L_{\text{buf}}}{\tau_{\text{proc}}}

$$

**Lower Bound (Node 2: ZenoCheck):**

Suppose $c_{\text{info}} < d_{\text{sync}}/\tau_{\text{proc}}$. Then information cannot traverse the synchronization distance within one processing cycle. By the Causal Interval (Definition {prf:ref}`def-causal-interval`), spacelike-separated modules cannot coordinate updates. The agent enters a **Zeno freeze**: each module waits indefinitely for signals that arrive too slowly. The belief update stalls, violating the continuity required by the WFR dynamics ({ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`).

**Upper Bound (Node 62: CausalityViolationCheck):**

Suppose $c_{\text{info}} > L_{\text{buf}}/\tau_{\text{proc}}$. Then signals can traverse the entire buffer depth within one processing cycle. This creates **temporal aliasing**: the agent receives information about its own future state before that state is computed. By the Safe Retrieval Bandwidth (Theorem {prf:ref}`thm-safe-retrieval-bandwidth`), this constitutes a causal paradox—the agent's prediction depends on data it has not yet generated.

Node 62 enforces Theorem {prf:ref}`thm-causal-stasis`: the metric becomes singular at the boundary where causal violations would occur, preventing traversal.

$\square$

({prf:ref}`thm-holographic-bound`) *theorem* — **The Holographic Bound**

Let $\text{Area}_\partial$ denote the boundary area of the agent's latent manifold (dimension $[L^{D-1}]$ for a $D$-dimensional bulk) and $I_{\text{req}}$ the information capacity required for viable operation (dimensionless, counting distinguishable microstates in nats). The Levin Length must satisfy:

$$
\ell_L^{D-1} \le \frac{\nu_D \cdot \text{Area}_\partial}{I_{\text{req}}}

$$

where $\nu_D$ is a **dimensionless** holographic coefficient (Corollary {prf:ref}`cor-a-dimension-dependent-coefficient`). Both sides have dimension $[L^{D-1}]$.

**Step 1.** From the Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`):

$$
I_{\text{bulk}} \le \frac{\nu_D \cdot \text{Area}_\partial}{\ell_L^{D-1}}

$$

**Step 2.** The agent requires $I_{\text{bulk}} \ge I_{\text{req}}$ to represent its world model. Substituting:

$$
I_{\text{req}} \le \frac{\nu_D \cdot \text{Area}_\partial}{\ell_L^{D-1}}

$$

**Step 3.** Rearranging yields the constraint on $\ell_L$.

$\square$

({prf:ref}`thm-capacity-horizon`) *theorem* — **The Capacity Horizon**

As $I_{\text{bulk}} \to I_{\max} = \nu_D \cdot \text{Area}_\partial / \ell_L^{D-1}$, the agent approaches a **Capacity Horizon**. The metric diverges:

$$
\|v\|_G \to 0 \quad \text{as} \quad I_{\text{bulk}} \to I_{\max}

$$

$$
g_{\text{FR}} = \frac{1}{\rho(1-\rho)} \to \infty \quad \text{as} \quad \rho \to 1

$$

(Lemma {prf:ref}`lem-metric-divergence-at-saturation`). The geodesic velocity vanishes, creating **causal stasis**: no information can cross the saturation boundary.

*Physical interpretation:* This is the agent-theoretic analogue of a black hole event horizon. Node 56 (CapacityHorizonCheck) enforces this bound.

$\square$

({prf:ref}`thm-landauer-constraint`) *theorem* — **The Landauer Constraint**

The Cognitive Temperature must satisfy:

$$
T_c \le \frac{\dot{E}_{\text{met}}}{\dot{I}_{\text{erase}} \cdot \ln 2}

$$

where we use natural units with $k_B = 1$.

**Step 1.** From the Generalized Landauer Bound (Theorem {prf:ref}`thm-generalized-landauer-bound`):

$$
\dot{\mathcal{M}}(s) \ge T_c \left| \frac{dH}{ds} \right|

$$

where $\dot{\mathcal{M}}$ is the metabolic flux and $dH/ds$ is the entropy change rate.

**Step 2.** Information erasure corresponds to entropy reduction. For $\dot{I}_{\text{erase}}$ bits per unit time:

$$
\left| \frac{dH}{ds} \right| = \dot{I}_{\text{erase}} \cdot \ln 2

$$

**Step 3.** The metabolic constraint $\dot{\mathcal{M}} \le \dot{E}_{\text{met}}$ bounds the erasure capacity:

$$
\dot{E}_{\text{met}} \ge T_c \cdot \dot{I}_{\text{erase}} \cdot \ln 2

$$

**Step 4.** Rearranging yields the temperature bound.

*Physical consequence:* If $T_c$ exceeds this bound, the agent cannot afford to forget—its memory becomes permanently saturated. Node 52 (LandauerViolationCheck) enforces this constraint.

$\square$

({prf:ref}`thm-ir-binding-constraint`) *theorem* — **The Infrared Binding Constraint**

At the macro-scale ($\mu \to 0$), the coupling must exceed a critical threshold:

$$
g_s(\mu_{\text{IR}}) \ge g_s^{\text{crit}}

$$

**Step 1.** From Axiom {prf:ref}`ax-feature-confinement`, the agent observes Concepts $K$, not raw features. This requires features to bind into stable composite objects at the macro-scale.

**Step 2.** From Theorem {prf:ref}`thm-emergence-binding-field`, binding stability requires the effective potential to confine features. The confinement condition is:

$$
\lim_{r \to \infty} V_{\text{eff}}(r) = \infty

$$

where $r$ is the separation between features.

**Step 3.** For $SU(N_f)$ gauge theory, this requires strong coupling $g_s > g_s^{\text{crit}}$ at large distances (Area Law, {ref}`Section 33 <sec-causal-information-bound>`).

**Step 4.** If $g_s(\mu_{\text{IR}}) < g_s^{\text{crit}}$, features escape confinement—"color-charged" states propagate to the boundary $\partial\mathcal{Z}$. This violates the Observability Constraint (Definition {prf:ref}`def-boundary-markov-blanket`): the agent cannot form stable objects.

Node 40 (PurityCheck) enforces that only color-neutral bound states reach the macro-register.

$\square$

({prf:ref}`thm-uv-decoupling-constraint`) *theorem* — **The Ultraviolet Decoupling Constraint**

At the texture scale ($\mu \to \infty$), the coupling must vanish:

$$
\lim_{\mu \to \infty} g_s(\mu) = 0

$$

**Step 1.** From the Texture Firewall (Axiom {prf:ref}`ax-bulk-boundary-decoupling`):

$$
\partial_{z_{\text{tex}}} \dot{z} = 0

$$

Texture coordinates are invisible to the dynamics.

**Step 2.** This requires texture-level degrees of freedom to be non-interacting. If $g_s(\mu_{\text{UV}}) > 0$, texture elements would bind, creating structure at the noise level.

**Step 3.** From the RG interpretation ({ref}`Section 7.12 <sec-stacked-topoencoders-deep-renormalization-group-flow>`), the TopoEncoder implements coarse-graining. Residual coupling at the UV scale would prevent efficient compression—the Kolmogorov complexity of texture would diverge.

**Step 4.** Asymptotic freedom ($\beta < 0$) provides the required behavior: $g_s \to 0$ as $\mu \to \infty$.

Node 29 (TextureFirewallCheck) enforces this decoupling.

$\square$

({prf:ref}`thm-stiffness-bounds`) *theorem* — **The Stiffness Bounds**

The Stiffness Ratio must satisfy:

$$
1 < \chi < \chi_{\text{max}}

$$

**Lower Bound ($\chi > 1$):**

**Step 1.** Memory stability requires that thermal fluctuations do not spontaneously erase stored information. The probability of a thermal transition is:

$$
P_{\text{flip}} \propto e^{-\Delta E / T_c} = e^{-\chi}

$$

**Step 2.** For $\chi < 1$, we have $P_{\text{flip}} > e^{-1} \approx 0.37$. States flip with high probability—the agent cannot maintain stable beliefs.

**Step 3.** This violates the Mass Gap requirement (Theorem {prf:ref}`thm-semantic-inertia`): beliefs must possess sufficient "inertia" to resist noise.

**Upper Bound ($\chi < \chi_{\text{max}}$):**

**Step 4.** Adaptability requires that the agent can update beliefs in finite time. The transition rate is:

$$
\Gamma_{\text{update}} \propto e^{-\chi}

$$

**Step 5.** For $\chi \to \infty$, transitions become exponentially suppressed—the agent freezes in its initial configuration, unable to learn.

**Step 6.** This violates the Update Dynamics requirement: the WFR reaction term $R(\rho)$ must enable transitions between states.

Node 7 (StiffnessCheck) enforces both bounds.

$\square$

({prf:ref}`thm-discount-window`) *theorem* — **The Discount Window**

The temporal discount factor $\gamma$ must satisfy:

$$
\gamma_{\text{min}} < \gamma < 1

$$

with $\gamma_{\text{min}} > 0$.

**Upper Bound ($\gamma < 1$):**

**Step 1.** From the Helmholtz equation (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`), the Value function satisfies:

$$
(\kappa^2 - \nabla^2) V = \rho_r

$$

where the screening mass $\kappa = \lambda / c_{\text{info}} = (-\ln\gamma)/\ell_0$ has dimension $[L^{-1}]$, and $\ell_0 = c_{\text{info}} \cdot \tau_{\text{proc}}$ is the causal horizon length (Definition {prf:ref}`def-agent-parameter-vector`). This ensures dimensional consistency: $[\kappa^2] = [L^{-2}] = [\nabla^2]$.

**Step 2.** For $\gamma = 1$, we have $\kappa = 0$. The equation becomes Poisson's equation:

$$
-\nabla^2 V = \rho_r

$$

where $\rho_r$ is the conservative reward source density (Definition {prf:ref}`def-the-reward-flux`).

For $D>2$, the Green's function decays as $1/r^{D-2}$ (long-range); for $D=2$ it grows logarithmically.

**Step 3.** Long-range value propagation violates locality: distant rewards dominate nearby decisions. The agent cannot form local value gradients for navigation.

**Step 4.** From Corollary {prf:ref}`cor-discount-as-screening-length`, finite screening $\kappa > 0$ (i.e., $\gamma < 1$) is required for local goal-directedness.

**Lower Bound ($\gamma > \gamma_{\text{min}}$):**

**Step 5.** For $\gamma \to 0$, we have $-\ln\gamma \to \infty$, hence $\kappa \to \infty$. The **Screening Length** (dimension $[L]$):

$$
\ell_\gamma = \frac{1}{\kappa} = \frac{\ell_0}{-\ln\gamma} = \frac{c_{\text{info}} \tau_{\text{proc}}}{-\ln\gamma} \to 0

$$

**Step 6.** Zero screening length means the agent responds only to immediate rewards—it has no planning horizon.

**Step 7.** This violates the Causal Buffer requirement (Axiom {prf:ref}`ax-causal-buffer-architecture`): the agent must anticipate beyond its current timestep.

$\square$

({prf:ref}`thm-feasible-region`) *theorem* — **The Feasible Region**

The **Feasible Region** $\mathcal{F} \subset \mathbb{R}^n_+$ is the intersection of all constraint half-spaces:

$$
\mathcal{F} = \{ \Lambda : \mathcal{S}_i(\Lambda) \le 0 \; \forall i \}

$$

A viable agent exists if and only if $\mathcal{F} \neq \emptyset$.

Each constraint $\mathcal{S}_i \le 0$ defines a closed half-space in parameter space. The intersection of finitely many closed half-spaces is either empty or a closed convex polytope (possibly unbounded).

**Existence:** The physics Standard Model constants $\Lambda_{\text{phys}} = (c, \hbar, G, k_B, \alpha)$ satisfy all constraints—we observe a functioning physical universe. Therefore $\mathcal{F} \neq \emptyset$.

**Uniqueness modulo scaling:** The constraints are homogeneous in certain parameter combinations. Dimensional analysis shows that physical observables depend only on dimensionless ratios. The feasible region is a lower-dimensional manifold in the full parameter space.

$\square$

({prf:ref}`thm-constrained-optimum`) *theorem* — **The Constrained Optimum**

The optimal parameter vector $\Lambda^*$ satisfies:

$$
\Lambda^* = \arg\max_{\Lambda \in \mathcal{F}} \mathcal{J}(\Lambda)

$$

subject to the Sieve constraints (Definition {prf:ref}`def-constraint-matrix`).

**Step 1.** The objective $\mathcal{J}$ is continuous on the closed feasible region $\mathcal{F}$.

**Step 2.** The holographic bound (Theorem {prf:ref}`thm-holographic-bound`) caps $I_{\text{bulk}}$, making $\mathcal{J}$ bounded above.

**Step 3.** By the extreme value theorem, $\mathcal{J}$ attains its maximum on $\mathcal{F}$.

**Step 4.** The optimum lies on the boundary of $\mathcal{F}$ where at least one constraint is active (saturated). This corresponds to operating at the edge of viability.

$\square$

({prf:ref}`thm-cognitive-equivalency`) *theorem* — **The Cognitive Equivalency Theorem**

Let $\mathcal{C}_{\text{hash}}$ be the computational task of finding a nonce $n$ such that $H(n) < T$ (hash inversion), and let $\mathcal{C}_{\text{grad}}$ be the task of computing a gradient $g = \nabla_\Theta \mathcal{L}(\Theta, D)$ on dataset $D$. Both tasks satisfy the same **Landauer lower bound** on energy expenditure:

$$
E_{\text{min}} \geq k_B T_c \ln 2 \cdot B_{\text{comp}}

$$

where $B_{\text{comp}}$ is the number of irreversible bit operations.

**Step 1 (Landauer Principle).** By Theorem {prf:ref}`thm-generalized-landauer-bound`, any computation that erases $\Delta H$ nats of information dissipates at least:

$$
\dot{\mathcal{M}} \geq T_c \left| \frac{dH}{ds} \right|

$$

**Step 2 (Hash Computation).** Computing $H(n)$ requires approximately $B_{\text{SHA}} \approx 64 \times 80 = 5120$ irreversible bit operations per hash. The minimum energy is:

$$
E_{\text{hash}} \geq k_B T_c \ln 2 \cdot B_{\text{SHA}} \cdot N_{\text{trials}}

$$

where $N_{\text{trials}} \approx 2^{d}$ for difficulty $d$.

**Step 3 (Gradient Computation).** Computing $g = \nabla_\Theta \mathcal{L}$ via backpropagation requires $O(|\Theta| \cdot |D|)$ multiply-accumulate operations. Each MAC erases intermediate bits, giving:

$$
E_{\text{grad}} \geq k_B T_c \ln 2 \cdot c_{\text{MAC}} \cdot |\Theta| \cdot |D|

$$

for architecture-dependent constant $c_{\text{MAC}}$.

**Step 4 (Equivalence).** Both computations satisfy the same thermodynamic bound. The difference is the *information content* of the output:
- $I(X_{\text{world}}; H(n)) = 0$ (no world knowledge)
- $I(X_{\text{world}}; g) > 0$ (gradient encodes data structure)

Therefore, gradient computation produces **useful information** while satisfying the same energy floor. $\square$

*Consequence:* The security budget of a blockchain can be redirected to train a global model without loss of thermodynamic hardness, provided verification remains tractable.

({prf:ref}`thm-difficulty-entropy-coupling`) *theorem* — **Difficulty-Entropy Coupling**

The difficulty adjustment algorithm maintains the **Landauer Invariant**: the minimum energy to produce a valid block is approximately constant:

$$
E_{\min}(B_h) \approx k_B T_c \ln 2 \cdot c_{\text{MAC}} \cdot |\Theta| \cdot \mathcal{D}_h = E_{\text{target}}

$$

**Step 1.** By the Generalized Landauer Bound (Theorem {prf:ref}`thm-generalized-landauer-bound`), gradient computation costs:

$$
E_{\text{grad}} \geq k_B T_c \ln 2 \cdot c_{\text{MAC}} \cdot |\Theta| \cdot |D_h|

$$

**Step 2.** The difficulty constraint $|D_h| \geq \mathcal{D}_h$ enforces:

$$
E_{\text{grad}} \geq k_B T_c \ln 2 \cdot c_{\text{MAC}} \cdot |\Theta| \cdot \mathcal{D}_h

$$

**Step 3.** The exponential adjustment (Definition {prf:ref}`def-difficulty-adjustment`) stabilizes block time at $t_{\text{target}}$, hence stabilizes energy expenditure rate at $E_{\text{target}} / t_{\text{target}}$.

**Step 4 (Fake Gradient Rejection).** A miner submitting $g' \neq \nabla_\Theta \mathcal{L}(\Theta, D_h)$ violates one of:
- **Directional Check:** Cosine similarity $\cos(g', g_{\text{true}}) < \theta_{\text{min}}$
- **Magnitude Check:** $\|g'\| / \|g_{\text{true}}\| \notin [1-\epsilon, 1+\epsilon]$
- **Causal Check (Node 53):** Interventional gap $\Delta_{\text{causal}}(g') > \delta_{\text{causal}}$

All checks are detectable by spot verification (Section {ref}`sec-holographic-verification`). $\square$

({prf:ref}`thm-holographic-verification`) *theorem* — **Holographic Verification Sufficiency**

Let $g$ be a claimed gradient and $\zeta$ its boundary flux certificate. If the boundary data satisfies:

1. **Energy Conservation:** $\|g\|_G^2 \leq \nu_D \cdot \text{Area}(\partial\mathcal{Z}) / \ell_L^{D-1}$ (Causal Information Bound)
2. **Flux Consistency:** $\|\nabla_\partial g - \nabla_\partial g_{\text{spot}}\| < \epsilon_{\text{flux}}$ on spot-check samples
3. **Curvature Bound:** $|\text{Tr}(H)| < \kappa_{\max}$

then with probability $\geq 1 - \delta$, the gradient is valid.

**Step 1 (Holographic Principle).** By Theorem {prf:ref}`thm-causal-information-bound`, bulk information is bounded by boundary area:

$$
I_{\text{bulk}}(g) \leq \nu_D \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}} = I_{\max}

$$

**Step 2 (Bulk-Boundary Correspondence).** The gradient $g \in T_\Theta \mathcal{M}$ projects to boundary flux $\nabla_\partial g$ via the restriction map. By the Bulk-Boundary Decoupling Axiom ({prf:ref}`ax-bulk-boundary-decoupling`), the boundary flux determines the bulk gradient up to texture degrees of freedom.

**Step 3 (Spot-Check Amplification).** A fraudulent gradient must differ from the true gradient in some coordinate. The probability of escaping detection in $k$ random spot checks is:

$$
P(\text{escape}) \leq (1 - p_{\text{detect}})^k

$$

where $p_{\text{detect}} \geq \epsilon_{\min}$ is the minimum detection probability per check.

**Step 4 (Energy Conservation).** A gradient claiming to reduce loss by $\Delta \mathcal{L}$ while having energy below the Landauer floor violates:

$$
\|g\|_G^2 < k_B T_c |\Delta H| / \dot{\mathcal{M}}_{\text{claimed}}

$$

This is detectable from the certificate without recomputation.

**Step 5 (Combining).** Setting $k = \log(1/\delta) / \log(1/(1-p_{\text{detect}}))$ spot checks achieves confidence $1-\delta$. $\square$

*Complexity:* Verification requires $O(\sqrt{|D|})$ operations vs $O(|D|)$ for full recomputation.

({prf:ref}`thm-verifier-nash-equilibrium`) *theorem* — **The Verifier's Nash Equilibrium**

In the Mining Game $\Gamma$ with exogenous detection probability $p_{\text{detect}}$ and parameters satisfying:

$$
\frac{S}{R + S} > \frac{C_{\text{honest}} - C_{\text{cheat}}}{R}

$$

**Honest** is a strictly dominant strategy, and $\sigma^* = (\text{Honest}, \ldots, \text{Honest})$ is the unique Nash Equilibrium.

**Step 1 (Utility Functions).** Define utilities for miner $i$:

$$
U_i(\text{Honest}) = R - C_{\text{honest}}

$$

$$
U_i(\text{Cheat}) = (1 - p_{\text{detect}}) \cdot R + p_{\text{detect}} \cdot (-S) - C_{\text{cheat}}

$$

where $p_{\text{detect}} \in (0, 1]$ is the probability of detection via spot-checking.

**Step 2 (Detection Probability).** By Theorem {prf:ref}`thm-holographic-verification`, detection probability satisfies:

$$
p_{\text{detect}} \geq 1 - (1 - \epsilon_{\min})^k

$$

for $k$ spot-check samples with $\epsilon_{\min} > 0$. Since $p_{\text{detect}}$ is exogenous (determined by the protocol, not other players), each miner faces a constant detection probability regardless of others' strategies.

**Step 3 (Incentive Compatibility).** Honesty is preferred when:

$$
U_i(\text{Honest}) > U_i(\text{Cheat})

$$

$$
R - C_{\text{honest}} > (1 - p_{\text{detect}}) R - p_{\text{detect}} S - C_{\text{cheat}}

$$

Rearranging:

$$
p_{\text{detect}} (R + S) > C_{\text{honest}} - C_{\text{cheat}}

$$

$$
p_{\text{detect}} > \frac{C_{\text{honest}} - C_{\text{cheat}}}{R + S} := p^*

$$

**Step 4 (Equilibrium Condition).** The theorem condition implies:

$$
\frac{S}{R + S} > \frac{C_{\text{honest}} - C_{\text{cheat}}}{R} \implies C_{\text{honest}} - C_{\text{cheat}} < \frac{S \cdot R}{R + S} < R

$$

Therefore $p^* = \frac{C_{\text{honest}} - C_{\text{cheat}}}{R + S} < 1$, ensuring the threshold is achievable with finite spot-checks.

**Step 5 (Dominant Strategy).** Since $p_{\text{detect}}$ is exogenous and independent of other players' strategies, miner $i$'s utility depends only on their own choice. When $p_{\text{detect}} > p^*$:

$$
\Delta U = U(\text{Cheat}) - U(\text{Honest}) = -p_{\text{detect}}(R + S) + (C_{\text{honest}} - C_{\text{cheat}}) < 0

$$

This holds regardless of what other miners do. Thus Honest is a **strictly dominant strategy**, and the unique Nash Equilibrium is all-Honest. $\square$

({prf:ref}`thm-minimum-friction-bft`) *theorem* — **Minimum Friction Byzantine Fault Tolerance**

The Metric Friction Consensus achieves Byzantine Fault Tolerance against $f < N/3$ adversarial validators for **gradient-poisoning attacks** (adversaries submit incorrect gradients).

**Scope:** This theorem addresses data integrity attacks (model poisoning, fake gradients). Classical BFT attacks (equivocation, censorship) are handled by the underlying stake-based leader election, which is assumed to follow standard PBFT guarantees.

**Step 1 (Honest Majority Alignment).** By Theorem {prf:ref}`thm-spontaneous-gauge-locking`, honest validators minimizing prediction error on the same data undergo spontaneous gauge locking: $G^{(i)} \to G^{(j)}$ for honest $i, j$.

**Step 2 (Adversarial Inflation).** By Theorem {prf:ref}`thm-adversarial-mass-inflation` (Adversarial Mass Inflation), any gradient $g_{\text{adv}} \neq g_{\text{true}}$ introduces non-zero metric perturbation:

$$
\tilde{G}^{(i)} = G^{(i)} + \alpha_{\text{adv}} \mathcal{G}_{ij}, \quad \alpha_{\text{adv}} = \|g_{\text{adv}} - g_{\text{true}}\|_G > 0

$$

where $\mathcal{G}_{ij}$ is the Game Tensor (Definition {prf:ref}`def-gauge-covariant-game-tensor`). The key insight: *there is no "zero-curvature" way to submit a fake gradient*.

**Step 3 (Friction Separation).** Let $\epsilon$ be the natural gradient variance among honest validators. The pairwise friction satisfies:

- Honest-Honest: $\mathcal{F}_{ij} \leq c_1 \epsilon^2$ (gauge-locked, small noise)
- Honest-Adversarial: $\mathcal{F}_{ik} \geq c_2 \alpha_{\text{adv}}$ (metric mismatch)

For the attack to succeed while evading detection, the adversary requires $\alpha_{\text{adv}} < c_1 \epsilon^2 / c_2$. But such small perturbations have negligible effect on model training—a successful attack requires $\alpha_{\text{adv}} \gg \epsilon$.

**Step 4 (Selection).** The total friction of a chain proposed by honest validators is:

$$
\mathcal{F}_{\text{total}}^{\text{honest}} \leq \binom{N-f}{2} c_1 \epsilon^2 + f(N-f) c_2 \alpha_{\text{adv}}

$$

An adversarial chain has friction at least $\mathcal{F}_{\text{total}}^{\text{adv}} \geq (N-f) c_2 \alpha_{\text{adv}}$.

With $f < N/3$ and $\alpha_{\text{adv}} \gg \epsilon^2$, the honest chain minimizes total friction. $\square$

*Remark:* The $N/3$ threshold matches classical BFT because friction-weighted voting is equivalent to stake-weighted voting when adversarial friction is high.

({prf:ref}`thm-adversarial-geometric-damping`) *theorem* — **Adversarial Geometric Damping**

An adversary controlling fraction $\alpha < 1/3$ of validators has influence on consensus bounded by:

$$
\|\Delta \Theta_{\text{adversarial}}\|_G \leq \frac{\alpha}{1 - 2\alpha} \|\Delta \Theta_{\text{honest}}\|_G

$$

**Step 1.** The consensus update is a friction-weighted average:

$$
\Delta \Theta = \frac{\sum_i w_i \Delta \Theta^{(i)}}{\sum_i w_i}

$$

where weights $w_i = 1/\mathcal{F}_{i,\text{total}}$ penalize high-friction validators.

**Step 2.** Adversarial validators have inflated friction:

$$
w_{\text{adv}} \leq w_{\text{honest}} / (1 + \alpha_{\text{adv}}/\epsilon^2)

$$

**Step 3.** The adversarial contribution is:

$$
\|\Delta \Theta_{\text{adv}}\| \leq \frac{\alpha \cdot w_{\text{adv}}}{(1-\alpha) w_{\text{honest}} + \alpha w_{\text{adv}}} \|\Delta \Theta_{\text{total}}\|

$$

**Step 4.** Taking $w_{\text{adv}} \to 0$ in the limit of high adversarial friction:

$$
\|\Delta \Theta_{\text{adv}}\| \to 0

$$

The adversary is geometrically isolated. $\square$

*Interpretation:* Adversaries are not voted out---they are **geometrically damped**. Their updates carry infinite inertia (Causal Stasis) and cannot influence the consensus trajectory.

({prf:ref}`thm-value-intelligence-coupling`) *theorem* — **Value-Intelligence Coupling**

The equilibrium token price $P_{\text{COG}}$ is bounded by:

$$
P_{\text{floor}} \leq P_{\text{COG}} \leq P_{\text{ceiling}}

$$

where:

$$
P_{\text{floor}} = \frac{C_{\text{electricity}}}{J_{\text{per\_COG}}}

$$

(cost of electricity to generate one COG worth of computation)

$$
P_{\text{ceiling}} = \frac{V_{\text{inference}}}{J_{\text{per\_query}}}

$$

(value of inference output per Joule)

**Step 1 (Floor).** If $P_{\text{COG}} < P_{\text{floor}}$, miners cannot profitably produce blocks. Supply decreases until price rises.

**Step 2 (Ceiling).** If $P_{\text{COG}} > P_{\text{ceiling}}$, users won't pay for inference. Demand decreases until price falls.

**Step 3 (Equilibrium).** At equilibrium:

$$
P_{\text{COG}}^* = \sqrt{P_{\text{floor}} \cdot P_{\text{ceiling}}}

$$

(geometric mean under log-linear supply/demand). $\square$

({prf:ref}`thm-ledger-memory-isomorphism`) *theorem* — **Ledger-Memory Screen Isomorphism**

Let $\Xi_T$ be the Memory Screen (Definition {prf:ref}`def-memory-screen`) and $\mathcal{L}_H$ be the blockchain of height $H$. There exists an isomorphism:

$$
\Phi: \mathcal{L}_H \to \Xi_T

$$

given by:

| Blockchain | Memory Screen | Symbol |
|:-----------|:--------------|:-------|
| Block height $h$ | Time coordinate $t$ | $h \leftrightarrow t$ |
| Merkle root $\mathcal{H}_h$ | Boundary state $z_{\partial}$ | $\mathcal{H}_h \leftrightarrow z_{\partial}(t)$ |
| Gradient $g_h$ | Flux $\alpha(t)$ | $g_h \leftrightarrow \alpha(t)$ |
| Chain $\sum_{h=0}^H B_h$ | Screen $\int_0^T \alpha(t) \delta_{\gamma(t)} dt$ | $\mathcal{L}_H \leftrightarrow \Xi_T$ |

**Step 1.** The Memory Screen (Definition {prf:ref}`def-memory-screen`) is:

$$
\Xi_T = \int_0^T \alpha(t') \, \delta_{\gamma(t')} \, dt'

$$

where $\alpha(t) = J_r(t)$ is the reward flux and $\gamma(t)$ is the trajectory.

**Step 2.** The blockchain is:

$$
\mathcal{L}_H = \sum_{h=0}^{H} B_h = \sum_{h=0}^{H} (g_h, \mathcal{H}_h, \ldots)

$$

**Step 3.** Define the correspondence:
- $t = h \cdot \Delta t$ where $\Delta t$ is block time
- $\alpha(t) = g_h / \Delta t$ (gradient rate)
- $\gamma(h) = \Theta_h$ (parameter trajectory)

**Step 4.** The discrete sum converges to the continuous integral:

$$
\sum_{h=0}^{H} g_h \cdot \mathbb{1}_{\Theta_h} \to \int_0^T \alpha(t) \delta_{\gamma(t)} dt

$$

as $\Delta t \to 0$. $\square$

*Interpretation:* The blockchain is the **frozen boundary** of the network's cognitive trajectory. Each block records a moment of learning; the full chain is the holographic screen encoding the network's history.

({prf:ref}`thm-51-attack-rejection`) *theorem* — **51% Attack Geometric Rejection**

An attacker controlling $> 50\%$ of compute cannot rewrite history without triggering **Spontaneous Fission**.

**Step 1.** The attacker proposes an alternative chain $\mathcal{C}'$ that contradicts the Memory Screen $\Xi_T$ of honest validators.

**Step 2.** By Theorem {prf:ref}`thm-adversarial-mass-inflation`, the attacker's chain has inflated metric:

$$
\tilde{G}_{\text{attack}} = G + \alpha_{\text{adv}} \mathcal{G}

$$

**Step 3.** The Metric Friction between honest and attack chains is:

$$
\mathcal{F}(\mathcal{C}, \mathcal{C}') = \|G - \tilde{G}_{\text{attack}}\|_F^2 \sim O(\alpha_{\text{adv}}^2)

$$

**Step 4.** When $\mathcal{F} > \mathcal{F}_{\text{crit}}$ (Fission Threshold from Theorem {prf:ref}`thm-fission-criterion`), the network undergoes **Spontaneous Fission**:
- The attacker ends up on a high-friction shard
- The honest validators continue on the low-friction chain

**Step 5.** The attacker's shard enters **Causal Stasis** (Theorem {prf:ref}`thm-causal-stasis`)---no one provides data/compute, and it dies. $\square$

*Interpretation:* You cannot buy the network because you cannot buy **geometric alignment**.

({prf:ref}`thm-causal-theft-prevention`) *theorem* — **Causal Theft Prevention**

Flash-loan attacks and front-running are rejected by **CausalityViolationCheck (Node 62)**.

**Step 1.** A flash-loan attack requires: borrow $\to$ manipulate price $\to$ profit $\to$ repay, all in one transaction.

**Step 2.** The profit depends on a price change that **hasn't propagated** in the causal graph at the time of the borrow.

**Step 3.** By the Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`), information cannot propagate faster than:

$$
v_{\max} = \frac{d_G(z, z')}{t}

$$

**Step 4.** **Node 62 (CausalityViolationCheck)** detects transactions using information from the future:

$$
\Delta_{\text{causal}} = D_{\text{KL}}(P_{\text{interventional}} \| P_{\text{observational}}) > \delta_{\text{causal}}

$$

**Step 5.** The transaction is rejected as **geometrically impossible**. $\square$

({prf:ref}`thm-corruption-babel-detection`) *theorem* — **Corruption Detection via Babel Limit**

Sustained deception by corrupt actors exceeds the **Babel Limit** (Theorem {prf:ref}`thm-babel-limit`) and causes loss of gauge locking.

**Step 1.** A corrupt actor broadcasts metric $G_{\text{corrupt}}$ claiming to optimize the objective, but their actual gradient flow generates different geometry.

**Step 2.** Maintaining the deception requires transmitting fake metric information at rate:

$$
\dot{I}_{\text{deception}} = H(G_{\text{corrupt}}) - H(G_{\text{true}})

$$

**Step 3.** By Theorem {prf:ref}`thm-babel-limit`, complete gauge locking requires:

$$
\dim(\mathfrak{g}) \cdot H(G) \leq C_{\mathcal{L}}

$$

**Step 4.** The deception increases effective entropy, violating the Babel Limit:

$$
\dim(\mathfrak{g}) \cdot (H(G_{\text{true}}) + \dot{I}_{\text{deception}}) > C_{\mathcal{L}}

$$

**Step 5.** The corrupt actor loses gauge locking with honest validators. Their words become "noise"---they are **topologically exiled** from consensus. $\square$

*Interpretation:* You cannot lie to the network because you cannot fake the **thermodynamic trace** of your actions.

({prf:ref}`thm-a-capacity-consistency-identity-proof-of-theorem`) *theorem* — **A.3.2 (Capacity-consistency identity; proof of Theorem {prf:ref}`thm-capacity-constrained-metric-law`)**

Under the hypotheses of Section A.2, stationarity of $\mathcal{S}[G,V]$ with respect to arbitrary variations $\delta G^{ij}$ that vanish on $\partial\mathcal{Z}$ implies the Euler–Lagrange equation

$$
R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij},

$$
with $T_{ij}$ given by Section A.2.3.

$$
\delta\mathcal{S} = \int_{\mathcal{Z}}\left[\left(R_{ij}-\frac12 R\,G_{ij}\right) + \Lambda G_{ij} - \kappa T_{ij}\right]\delta G^{ij}\,d\mu_G + \text{(boundary terms)}.

$$
Boundary terms vanish under the clamped boundary condition (or after adding an appropriate boundary term). Because $\delta G^{ij}$ is arbitrary in the interior, the fundamental lemma of the calculus of variations implies the bracketed tensor must vanish pointwise almost everywhere, yielding the stated identity (see e.g. Evans, *Partial Differential Equations*, 2010, for the functional-analytic lemma).

*Interpretation.* The Ricci curvature governs local volume growth; enforcing a boundary-limited bulk information volume forces the metric to stretch/compress coordinates so that information-dense regions (large $\|\nabla V\|$ and/or large $U(V)$) do not generate bulk structure that cannot be grounded at the boundary.

*Remark (regularizer).* The squared residual of this identity defines the capacity-consistency loss $\mathcal{L}_{\text{cap-metric}}$; see {ref}`Appendix B <sec-appendix-b-units-parameters-and-coefficients>`.

({prf:ref}`thm-classification-as-relaxation-a`) *theorem* — **Classification as Relaxation**

Under the overdamped dynamics with class-conditioned potential $V_y$:

$$
dz = -G^{-1}(z) \nabla V_y(z, K)\, ds + \sqrt{2T_c}\, G^{-1/2}(z)\, dW_s,

$$
the limiting chart assignment satisfies $\lim_{s \to \infty} K(z(s)) \in \mathcal{A}_y$ almost surely, provided the initial condition lies in the basin $\mathcal{B}_y$ and $T_c$ is sufficiently small.

({prf:ref}`thm-a-chentsov-uniqueness`) *theorem* — **A.6.0b (Chentsov's Uniqueness Theorem)**

The **Fisher Information Metric** is the unique Riemannian metric on statistical manifolds (up to constant scaling) that is invariant under sufficient statistics.

**Statement.** Let $\mathcal{M}$ be a statistical manifold parameterized by $\theta \in \Theta$. Any Riemannian metric $g$ on $\mathcal{M}$ satisfying:
1. **Markov invariance:** $g$ is preserved under Markov morphisms (conditional expectations)
2. **Smoothness:** $g$ varies smoothly with $\theta$

is proportional to the Fisher Information Metric:

$$
g_{ij}(\theta) = c \cdot \mathbb{E}_\theta\left[\frac{\partial \log p(x|\theta)}{\partial \theta^i} \frac{\partial \log p(x|\theta)}{\partial \theta^j}\right]

$$
for some constant $c > 0$.


*Significance.* Chentsov's theorem establishes that the Fisher metric is not a choice but a *necessity*: any geometry on probability space that respects statistical structure must be (proportional to) the Fisher geometry. This grounds our derivation in fundamental statistics, not ad-hoc assumptions.

({prf:ref}`thm-a-boundary-channel-capacity`) *theorem* — **A.6.0g (Boundary Channel Capacity)**

The channel capacity of a 2-dimensional boundary $\partial\mathcal{Z}$ with Riemannian area $A$ is:

$$
C_\partial = \frac{A}{4\ell_L^2} \text{ nats}.

$$

$$
C_\partial = N_{\text{cells}} \times 1 \text{ nat} = \frac{A}{4\ell_L^2}. \quad \square

$$
*Remark (Dimension Generalization).* For a $(D-1)$-dimensional boundary with $D > 2$, the formula generalizes to:

$$
C_\partial = \nu_D \cdot \frac{A}{\ell_L^{D-1}},

$$
where $\nu_D$ is the Holographic Coefficient (Definition {prf:ref}`def-holographic-coefficient`). The 2D case with $\nu_2 = 1/4$ is the primary focus of this specification.

*Remark (Shannon's Channel Coding Theorem).* This invokes the classical result that the capacity of $N$ parallel channels is additive. The generalization to continuous channels with Fisher geometry follows from rate-distortion theory {cite}`cover2006elements`.

({prf:ref}`thm-a-microstate-count-area-law`) *theorem* — **A.6.0h (Microstate Count and the Area Law)**

The number of boundary-distinguishable microstates in the bulk is:

$$
\Omega = \exp\left(\frac{A}{4\ell_L^2}\right),

$$
and the maximum information about bulk configuration, as measured by an external observer, is:

$$
I_{\max} = \ln \Omega = \frac{A}{4\ell_L^2}.

$$
2. The maximum number of distinguishable messages through a channel of capacity $C$ nats is $e^C$ (Shannon's channel coding theorem {cite}`cover2006elements`).

3. Therefore, the number of boundary-distinguishable microstates is bounded:

$$
\Omega \leq e^{C_\partial} = \exp\left(\frac{A}{4\ell_L^2}\right).

$$
4. **Achievability:** The bound is saturated when the boundary is tiled with minimal distinguishable cells, each encoding 1 nat via orthogonal degrees of freedom. This follows from the channel capacity achievability in Shannon's theorem.

5. The maximum information is:

$$
I_{\max} = \ln \Omega = \frac{A}{4\ell_L^2}. \quad \square

$$
*Remark (Non-Circularity).* This derivation uses only:
- Chentsov's uniqueness theorem (statistics)
- Fisher geodesic distance calculation (geometry)
- Curvature normalization $K = -1$ (convention, not assumption)
- Shannon's channel capacity (information theory)

It does **not** invoke the Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`). The Metric Law is a *dynamical* statement about how the metric responds to information density; the Area Law derived here is a *kinematic* bound on distinguishable states.

({prf:ref}`thm-a-complete-derivation-area-law`) *theorem* — **A.6.6 (Complete Derivation of the Area Law)**

Combining the above results:

1. **From Lemma {prf:ref}`lem-a-bulk-to-boundary-conversion`:** $I_{\text{bulk}} = \frac{1}{\kappa} \oint_{\partial\mathcal{Z}} \text{Tr}(K) \, dA_G$

2. **At saturation:** The extrinsic curvature $\text{Tr}(K) = (n-1)/r_h$ for an $(n-1)$-sphere boundary.

3. **Boundary area:** $\text{Area}(\partial\mathcal{Z}) = \Omega_{n-1} r_h^{n-1}$ where $\Omega_{n-1}$ is the volume of the unit $(n-1)$-sphere.

4. **Fisher normalization:** $\kappa = 8\pi \ell_L^2$ (fixed by consistency with Proposition {prf:ref}`prop-a-area-minimal-cell`).

Substituting:

$$
I_{\max} = \frac{1}{8\pi \ell_L^{n-1}} \cdot \frac{n-1}{r_h} \cdot \Omega_{n-1} r_h^{n-1} = \frac{(n-1)\Omega_{n-1}}{8\pi} \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{n-1}}.

$$
Identifying the **Holographic Coefficient** $\nu_n := (n-1)\Omega_{n-1}/(8\pi)$ (Definition {prf:ref}`def-holographic-coefficient`), we obtain the **general result**:

$$
\boxed{I_{\max} = \nu_n \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{n-1}}}.

$$

**Special case ($n = 2$, Poincare disk):** With $\Omega_1 = 2\pi$ (circumference of unit circle), we get $\nu_2 = 1/4$. The familiar Bekenstein-Hawking form:

$$
I_{\max} = \frac{\text{Area}(\partial\mathcal{Z})}{4\ell_L^2}

$$
uses $\ell_L^2$ (rather than $\ell_L^{n-1} = \ell_L$) because the Poincare disk metric normalization $G(0) = 4I$ maps coordinate cells to Riemannian areas.

This completes the derivation. The Holographic Coefficient $\nu_n$ arises from the combination of:
- The $1/8\pi$ from the coupling constant $\kappa$
- The geometric factor $(n-1)\Omega_{n-1}$ from sphere surface area
- The Fisher metric normalization

$\square$

({prf:ref}`thm-e7-ground-state-positivity`) *theorem* — **E.7.1 (Strict Positivity of the Ground State)**

Let $\Psi_0$ be the ground state eigenfunction of $\hat{H}_\sigma$ (the eigenfunction with eigenvalue $E_0$). Then:

$$
|\Psi_0(\mathbf{z})| > 0 \quad \forall \mathbf{z} \in \mathcal{M}.

$$
*Consequence:* For any open set $\Omega_B \subset \mathcal{M}$, the probability measure satisfies:

$$
\mu(\Omega_B) = \int_{\Omega_B} |\Psi_0(\mathbf{z})|^2 \, d\mu_{\mathbf{g}}(\mathbf{z}) > 0.

$$
Therefore, if an agent is localized in $\Omega_A$, there is strictly positive probability of finding it in $\Omega_B$.

({prf:ref}`thm-e7-agmon-decay-bound`) *theorem* — **E.7.2 (Agmon Exponential Decay Bound)**

Let $\Psi_0$ be the ground state of $\hat{H}_\sigma$ with eigenvalue $E_0$. For any $\epsilon > 0$, there exists a constant $C_\epsilon > 0$ (depending on $\mathcal{M}$, $\mathcal{V}$, and $\epsilon$, but not on $\sigma$) such that:

$$
|\Psi_0(\mathbf{z})| \leq C_\epsilon \exp\left( - \frac{1 - \epsilon}{\sigma} d_{\text{Ag}}(\mathbf{z}, \Omega_A) \right) \quad \forall \mathbf{z} \in \mathcal{M},

$$
where $d_{\text{Ag}}(\mathbf{z}, \Omega_A) := \inf_{\mathbf{y} \in \Omega_A} d_{\text{Ag}}(\mathbf{z}, \mathbf{y})$.

*Interpretation:* The wave-function amplitude decays exponentially with rate $1/\sigma$ times the Agmon distance from the classical region. Deeper into the barrier (larger $d_{\text{Ag}}$), the amplitude is exponentially smaller.

({prf:ref}`thm-e7-feynman-kac`) *theorem* — **E.7.4 (Feynman-Kac Representation)**

Let $(\mathbf{X}_s)_{s \geq 0}$ be Brownian motion on the Riemannian manifold $(\mathcal{M}, \mathbf{g})$, starting at $\mathbf{X}_0 = \mathbf{z}$. Then the ground state $\Psi_0$ admits the representation:

$$
\Psi_0(\mathbf{z}) = \lim_{t \to \infty} e^{E_0 t} \cdot \mathbb{E}_{\mathbf{z}}\left[ \exp\left( -\frac{1}{\sigma^2} \int_0^t \mathcal{V}(\mathbf{X}_s) \, ds \right) \phi(\mathbf{X}_t) \right],

$$
where $\phi \in L^2(\mathcal{M})$ is any function with $\langle \Psi_0, \phi \rangle \neq 0$.

*Remark:* This is rigorous—not a heuristic "path integral." The expectation is over Brownian paths on the manifold.

## Corollaries

({prf:ref}`cor-boundary-filter-interpretation`) *corollary* — **Boundary filter interpretation**

Sieve Nodes 13-16 (Boundary/Overload/Starve/Align) can be interpreted as monitoring a trace-like coupling between bulk and boundary (informally: whether internal degrees of freedom remain supported by boundary evidence), analogous in spirit to the trace map $\operatorname{Tr}: H^1(\mathcal{Z}) \to H^{1/2}(\partial \mathcal{Z})$:

*   **Mode B.E (Injection):** Occurs when interface inflow exceeds the effective capacity of the manifold (Levin capacity), breaking the assumed operating regime.
*   **Mode B.D (Starvation):** Occurs when interface inflow is too weak, causing the internal information volume to decay (catastrophic forgetting).

({prf:ref}`cor-the-hyperbolic-embedding`) *corollary* — **The Hyperbolic Embedding**

There exists a quasi-isometric embedding $\iota: V(\mathcal{T}) \hookrightarrow \mathbb{H}^n$ into $n$-dimensional hyperbolic space such that the depth in the tree correlates with the hyperbolic distance from a basepoint. In the upper half-space model $\mathbb{H}^n = \{(x, y) : y > 0\}$ with metric $ds^2 = (dx^2 + dy^2)/y^2$, tree depth $\ell$ maps to $\log(1/y)$; equivalently, in the Poincare ball model, depth maps to $\tanh^{-1}(r)$ where $r \in [0,1)$ is the radial coordinate.

This identifies the **discrete macro-register** $K_t = (K_{\text{chart}}, K_{\text{code}})$ as the bulk of a hyperbolic geometry. Navigating from the root to a leaf corresponds to moving from the interior of $\mathbb{H}^n$ toward the ideal boundary $\partial_\infty \mathbb{H}^n$, increasing information resolution at each step.

({prf:ref}`cor-gradient-decomposition`) *corollary* — **Gradient Decomposition**

The gradient of the effective potential decomposes as:

$$
\nabla_G \Phi_{\text{eff}} = \alpha\, \nabla_G U + (1 - \alpha)\, \nabla_G V_{\text{critic}} + \gamma_{risk}\, \nabla_G \Psi_{\text{risk}}.

$$
For the Poincare disk model, the first term simplifies to:

$$
\nabla_G U = -\frac{(1-|z|^2)}{2}\, \hat{z}, \qquad \hat{z} = \frac{z}{|z|}.

$$
**Cross-references:** Definition {prf:ref}`def-hyperbolic-volume-growth`, {ref}`Section 2.7 <sec-the-hjb-correspondence>` (Critic $V$), Section 14.2 (MaxEnt control), Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

*Forward reference (Scalar Field Interpretation).* {ref}`Section 24 <sec-the-reward-field-value-forms-and-hodge-geometry>` provides the complete field-theoretic interpretation of $V_{\text{critic}}$: the Critic solves the **Screened Poisson Equation** (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`) with rewards as boundary charges (Definition {prf:ref}`def-the-reward-flux`), the Value represents **Gibbs Free Energy** (Axiom {prf:ref}`ax-the-boltzmann-value-law`), and the Value Hessian induces a **Conformal Coupling** to the metric (Definition {prf:ref}`def-value-metric-conformal-coupling`).

({prf:ref}`cor-recovery-of-holographic-flow`) *corollary* — **Recovery of Holographic Flow**

Setting $\alpha = 1$ (pure generation) and $T_c \to 0$ (deterministic limit) in the overdamped equation recovers the holographic gradient flow from {ref}`Section 21.2 <sec-policy-control-field>`:

$$
\dot{z} = -G^{-1}(z)\,\nabla U(z).

$$
For the Poincare disk, this gives $\dot{z} = \frac{(1-|z|^2)}{2}\,z$, which integrates to $|z(\tau)| = \tanh(\tau/2)$.


*Remark.* This proves that the "ad-hoc" holographic law from {ref}`Section 21 <sec-radial-generation-entropic-drift-and-policy-control>` is actually the **optimal control trajectory** for the geometry defined in {ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`, vindicating the intuition.

({prf:ref}`cor-fokker-planck-duality`) *corollary* — **Fokker-Planck Duality {cite}`risken1996fokkerplanck`**

The stationary distribution of the overdamped SDE is:

$$
p_*(z) \propto \exp\left(-\frac{\Phi_{\text{gen}}(z)}{T_c}\right)\,\sqrt{|G(z)|},

$$
where $|G| = \det(G)$ is the metric determinant. This is the Boltzmann distribution on the curved manifold.

$$
\partial_s p = \nabla_i\left( G^{ij}\left( p\,\partial_j\Phi + T_c\,\partial_j p \right) \right).

$$
Setting $\partial_s p = 0$ and using detailed balance gives $p \propto e^{-\Phi/T_c} \sqrt{|G|}$. The $\sqrt{|G|}$ factor accounts for the Riemannian volume form. $\square$

**Cross-references:** {ref}`Section 21.2 <sec-policy-control-field>` (Langevin dynamics), Theorem {prf:ref}`thm-equivalence-of-entropy-regularized-control-forms-discrete-macro`, {ref}`Section 2.11 <sec-variance-value-duality-and-information-conservation>` (Belief density evolution).

({prf:ref}`cor-deterministic-boundary`) *corollary* — **Deterministic Boundary**

As $|z| \to 1$:

$$
T_c(z) \to 0, \qquad \text{noise} \to 0.

$$
The agent becomes deterministic at the boundary, ensuring reproducible outputs.

({prf:ref}`cor-prompt-action-label`) *corollary* — **Prompt = Action = Label**

The following are isomorphic as boundary conditions on $\partial\mathcal{Z}$:

$$
\text{RL Action} \;\cong\; \text{Classification Label} \;\cong\; \text{LLM Prompt}.

$$
Each specifies:
1. **Which chart** to route to (discrete macro $K$ or $A$)
2. **Where in the chart** to aim (continuous nuisance $z_n$ or $z_{n,\text{motor}}$)
3. **What texture** to inject (visual or motor texture)

*Remark (Unified Training Objective).* This isomorphism enables transfer learning across task domains: an agent trained on RL can be fine-tuned for classification by reinterpreting the action space as label space, with the same holographic dynamics.

**Cross-references:** {ref}`Section 21.2 <sec-policy-control-field>` (Control Field), Theorem {prf:ref}`thm-unified-control-interpretation`, Definition {prf:ref}`def-effective-potential`.

*Forward reference (Effective Potential Resolution).* {ref}`Section 24.2 <sec-the-bulk-potential-screened-poisson-equation>` resolves the meaning of $\Phi_{\text{eff}} = V_{\text{critic}}$: the Critic solves the **Screened Poisson Equation** to compute the potential from boundary reward charges. The discount factor $\gamma$ determines the screening length $\ell = c_{\text{info}} \Delta t / (-\ln\gamma)$ (natural units: $1/(-\ln\gamma)$) (Corollary {prf:ref}`cor-discount-as-screening-length`), explaining why distant rewards are exponentially suppressed in policy.

({prf:ref}`cor-discount-as-screening-length`) *corollary* — **Discount as Screening Length**

The discount factor $\gamma$ determines a characteristic **screening length**:

$$
\ell_{\text{screen}} = \frac{1}{\kappa} = \frac{c_{\text{info}} \Delta t}{-\ln\gamma} = \frac{c_{\text{info}}}{\lambda}.

$$
where $\lambda := -\ln\gamma / \Delta t$.
For $\gamma = 0.99$ and $c_{\text{info}} \Delta t = 1$: $\ell_{\text{screen}} \approx 100$ steps.

*Interpretation:* Rewards at geodesic distance $> \ell_{\text{screen}}$ from state $z$ are exponentially suppressed in their contribution to $V(z)$. This is the **temporal horizon** recast as a **spatial horizon** in latent space.

*Note:* Numerical values below assume natural units ($c_{\text{info}} \Delta t = 1$).

**Table 24.2.5 (Discount-Screening Correspondence).**

| Discount $\gamma$ | Screening Mass $\kappa$ | Screening Length $\ell$ | Interpretation                    |
|-------------------|-------------------------|-------------------------|-----------------------------------|
| $\gamma \to 1$    | $\kappa \to 0$          | $\ell \to \infty$       | Infinite horizon (massless field) |
| $\gamma = 0.99$   | $\kappa \approx 0.01$   | $\ell \approx 100$      | Standard RL                       |
| $\gamma = 0.9$    | $\kappa \approx 0.1$    | $\ell \approx 10$       | Short horizon                     |
| $\gamma \to 0$    | $\kappa \to \infty$     | $\ell \to 0$            | Myopic (infinitely massive)       |

**Cross-references:** {ref}`Section 2.7 <sec-the-hjb-correspondence>` (HJB Equation), Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

({prf:ref}`cor-equilibrium-distribution`) *corollary* — **Conservative Equilibrium Distribution**

**Conservative Case ($\mathcal{F} = 0$):** At equilibrium ($\partial_s \rho = 0$), the WFR dynamics with reaction rate $r(z) \propto (\Phi(z) - \bar{\Phi})$ converge to the Boltzmann distribution:

$$
\rho_\infty(z) \propto \exp\left(\frac{\Phi(z)}{T_c}\right),

$$
which is exactly the canonical ensemble (Definition {prf:ref}`def-canonical-ensemble`).

*Remark.* In the conservative case, the stationary distribution has zero probability current ($J = 0$). The distribution concentrates in high-$\Phi$ regions with concentration controlled by $T_c$.

({prf:ref}`cor-varentropy-stability`) *corollary* — **The Varentropy-Stability Relation (Cognitive Heat Capacity)**

Let $\mathcal{I}(a|z) = -\ln \pi(a|z)$ be the surprisal of an action. Define the **Policy Varentropy** $V_H(z)$ as the variance of the surprisal under the Boltzmann policy:

$$
V_H(z) := \mathrm{Var}_{a \sim \pi}[\mathcal{I}(a|z)] = \mathbb{E}_{\pi}\left[ \left( \ln \pi(a|z) + H(\pi) \right)^2 \right].

$$
*Units:* $\mathrm{nat}^2$.

Under the Boltzmann-Value Law (Axiom {prf:ref}`ax-the-boltzmann-value-law`), the Varentropy equals the **Heat Capacity** $C_v$ of the decision state:

$$
V_H(z) = \beta^2 \mathrm{Var}_\pi[Q] = C_v,

$$
where $\beta = 1/T_c$ is the inverse cognitive temperature. Equivalently:

$$
V_H(z) = T_c \frac{\partial H(\pi)}{\partial T_c}.

$$
**Operational Consequence:**
1. **Thermal Stability:** $V_H$ measures the sensitivity of the agent's exploration strategy to changes in the cognitive temperature $T_c$.
2. **Phase Transitions:** A divergence or spike in $V_H$ signals a second-order phase transition (critical point) where the policy is bifurcating from a single mode to multiple modes (or collapsing).
3. **Governor Constraint:** To ensure quasi-static evolution (reversible learning), the annealing rate $\dot{T}_c$ must satisfy the adiabatic condition:

$$
|\dot{T}_c| \ll \frac{T_c}{\sqrt{V_H(z)}}.

$$

({prf:ref}`cor-inertia-at-critical-regions`) *corollary* — **Inertia at Critical Regions**

Near sharp ridges or valleys of $V$ (where $\|\nabla^2 V\|$ is large), the conformal factor causes:

1. **Inertia Increase:** The effective mass $\tilde{G}(z) = \Omega^2(z) G(z)$ increases, so the agent slows down near critical decision boundaries ({ref}`Section 22.2 <sec-the-coupled-jump-diffusion-sde>` mass scaling).

2. **Resolution Increase:** The capacity-constrained metric allocates more volume to high-curvature regions (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), allowing higher-fidelity representation of value gradients.

3. **Stability:** The agent cannot "rush through" regions of high value curvature—it is forced to carefully navigate decision boundaries.

*Remark (Physical analogy).* The conformal scaling of effective velocity is mathematically analogous to gravitational time dilation in general relativity, where proper time dilates in regions of high gravitational potential.

({prf:ref}`cor-the-three-boundary-conditions`) *corollary* — **The Three Boundary Conditions**

The agent-environment interface decomposes into exactly three types of boundary conditions:

1. **Dirichlet** (Sensors): Clamp position $q = q_{\text{obs}}$. Information flows **in**.
2. **Neumann** (Motors): Clamp flux $\nabla_n \cdot p = j_{\text{motor}}$. Information flows **out**.
3. **Source** (Rewards): Inject charge $\sigma_r$ at boundary. Creates **potential field**.

These three conditions fully specify the agent's interaction with its environment.

**Cross-references:** {ref}`Section 23 <sec-the-boundary-interface-symplectic-structure>` (Holographic Interface), {ref}`Section 22 <sec-the-equations-of-motion-geodesic-jump-diffusion>` (Equations of Motion), {ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>` (Capacity-Constrained Geometry).

({prf:ref}`cor-saturation-velocity-tradeoff`) *corollary* — **The Saturation-Velocity Tradeoff**

Let $\eta := I_{\text{bulk}}/I_{\max}$ be the saturation ratio. Near the bound, the update velocity scales as:

$$
\|v\|_G \sim (1 - \eta)^{1/2}.

$$

*Interpretation.* At 90% saturation ($\eta = 0.9$), the agent operates at $\sim 32\%$ of its maximum velocity. At 99% saturation, velocity drops to $\sim 10\%$. The approach to the bound is gradual but accelerating.

({prf:ref}`cor-inference-via-relaxation`) *corollary* — **Inference via Relaxation**

Classification inference proceeds as:
1. Encode: $z_0 = \text{Enc}(x)$
2. Relax under neutral potential $V_{\text{base}}$ (no class conditioning) to equilibrium $z^*$
3. Read out: $\hat{y} = \arg\max_y P(Y=y \mid K(z^*))$

*Remark (Fast Path).* In practice, we often skip the relaxation and use direct readout: $\hat{y} = \arg\max_y \sum_k w_k(x) \cdot P(Y=y \mid K=k)$, where $w_k(x)$ are the router weights ({ref}`Section 7.8 <sec-tier-the-attentive-atlas>`). The relaxation interpretation justifies this as the $T_c \to 0$, $s \to \infty$ limit.

**Cross-references:** {ref}`Section 22.5 <sec-the-overdamped-limit>` (Overdamped Limit), Definition {prf:ref}`def-effective-potential`, {ref}`Section 2.7 <sec-the-hjb-correspondence>` (Critic).

({prf:ref}`cor-label-as-symmetry-breaking-field-cf-classifier-free-guidance`) *corollary* — **Label as Symmetry-Breaking Field, cf. classifier-free guidance {cite}`ho2022cfg`**

The class label $y$ breaks the $SO(2)$ symmetry of the unconditioned flow in the Poincare disk. At the origin:

1. **Unconditioned:** $\nabla V_{\text{base}}(0) = 0$ (symmetric saddle)
2. **Conditioned:** $\nabla V_y(0) = -\beta_{\text{class}} \nabla_z \log P(Y=y \mid K(z))|_{z=0} \neq 0$

The non-zero gradient aligns the initial "kick" direction with the class-$y$ basin.

({prf:ref}`cor-existence-of-descent-direction`) *corollary* — **Existence of Descent Direction**

At any non-stationary point $\theta$ where LICQ holds (the gradients $\{\nabla C_k : C_k(\theta) = 0\}$ for active constraints are linearly independent), there exist multipliers $\lambda_k \geq 0$ and step size $\eta > 0$ such that $\Delta V_{\mathfrak{L}} < 0$.


**Cross-references:** {ref}`Section 2.3 <sec-the-bridge-rl-as-lyapunov-constrained-control>` (Lyapunov-Constrained Control).

({prf:ref}`cor-varentropy-brake`) *corollary* — **The Varentropy Brake (Annealing Safety Margin)**

The training process involves lowering $T_c$ (annealing) to converge on a Nash equilibrium. The stability of this process is governed by the Varentropy (Corollary {prf:ref}`cor-varentropy-stability`).

For the optimization trajectory to remain in the basin of attraction of the global minimum, the cooling schedule must be modulated by the Varentropy:

$$
\frac{d T_c}{dt} = - \eta \cdot \frac{T_c}{1 + \gamma V_H(\theta_t)},

$$
where $\eta, \gamma > 0$ are constants.

*Units:* $[\dot{T}_c] = \mathrm{nat}/[\text{time}]$.

**Mechanism:**
- When $V_H(\theta_t)$ is high (system is near a critical decision point/ridge), the effective cooling rate $\dot{T}_c \to 0$. The Governor "freezes" the temperature to allow the agent to resolve the bifurcation via exploration rather than collapsing into a random mode.
- This prevents **Spontaneous Symmetry Breaking** errors where rapid cooling locks the agent into a suboptimal local minimum.

({prf:ref}`cor-bimodal-instability`) *corollary* — **The Bimodal Instability Theorem (Fission Trigger)**

Let $K$ be a macro-symbol with associated policy $\pi(\cdot|K)$. The **Structural Stability** of $K$ is inversely proportional to its Varentropy.

If the policy $\pi(\cdot|K)$ is a mixture of two disjoint, equally weighted strategies (a "Buridan's Ass" scenario on a value ridge), the Varentropy satisfies:

$$
V_H(K) = \frac{1}{4}\left(\frac{\Delta Q}{T_c}\right)^2,

$$
where $\Delta Q = |Q_1 - Q_2|$ is the value gap between the modes. In the limit of distinct modes ($\Delta Q \gg T_c$), $V_H$ is maximized, whereas for a uniform (maximum entropy) distribution, $V_H = 0$.

*Units:* $\mathrm{nat}^2$.

**Refined Fission Criterion:**
The **Geometric Tension** $\sigma_k^2$ (Definition {prf:ref}`def-intra-symbol-variance`) is rigorously generalized by the **Varentropy Excess**:

$$
\text{Fission}(K) \iff V_H(K) > \mathcal{V}_{\text{crit}} \quad \text{AND} \quad H(K) > H_{\text{noise}}.

$$
**Interpretation:**
- **High $H$, Low $V_H$:** Aleatoric Uncertainty (Noise/Fog). The distribution is flat. *Action:* Smoothing/Integration.
- **High $H$, High $V_H$:** Epistemic Conflict (Bifurcation). The distribution is multimodal. *Action:* Topological Fission (Node 50).

({prf:ref}`cor-hierarchical-stability`) *corollary* — **Hierarchical Stability**

The stacked architecture is **inherently stable** against fission cascades. Ontological expansion at coarse scales (low $\ell$) pre-empts the need for expansion at fine scales (high $\ell$).

*Interpretation:* If the agent learns a new high-level concept (e.g., "mammal"), the residual variance available to learn low-level distinctions (e.g., specific breeds) is reduced. The hierarchy self-regulates, preventing runaway complexity growth.

({prf:ref}`cor-epistemic-curiosity-filter`) *corollary* — **The Epistemic Curiosity Filter**

The Causal Information Potential $\Psi_{\text{causal}}$ (Definition {prf:ref}`def-causal-information-potential`) is maximized in regions of high **posterior varentropy**, not merely high entropy.

**Key Insight:** Let $V_H[P(\theta_W | z, a, z')]$ denote the Varentropy of the posterior over World Model parameters after observing transition $(z, a) \to z'$. Then:

$$
\nabla \Psi_{\text{causal}} \propto \nabla \mathbb{E}_{z'} \left[ V_H [P(\theta_W | z, a, z')] \right].

$$
*Units:* nat (for $\Psi_{\text{causal}}$), $\mathrm{nat}^2$ (for $V_H$).

**Operational Significance:**
The Curiosity Force $\mathbf{f}_{\text{exp}}$ (Theorem {prf:ref}`thm-augmented-drift-law`) should be weighted by the Varentropy of the World Model's prediction, not just its Entropy.

1. **High Entropy, Low Varentropy:** The World Model is confidently predicting "I don't know" (White Noise). The gradient $\nabla \Psi \approx 0$. The agent ignores this region (solves the "Noisy TV" problem).
2. **High Entropy, High Varentropy:** The World Model oscillates between distinct causal hypotheses ($H_1$: "Object falls", $H_2$: "Object floats"). The gradient $\nabla \Psi$ is maximal. The agent is strongly attracted to this state to resolve the structural ambiguity.

**Implementation:** The Experimental Sieve (Algorithm 32.5.1) selects interventions $do(a)$ that maximize the **Varentropy of the expected outcome distribution**.

({prf:ref}`cor-scientific-method-as-geodesic`) *corollary* — **Scientific Method as Geodesic**

In the absence of task reward ($V = \text{const}$), the agent behaves as a "Pure Scientist," traversing the latent manifold to minimize the total epistemic entropy of the World Model.

({prf:ref}`cor-survival-objective`) *corollary* — **The Survival Objective**

The agent's fundamental objective is not reward maximization but **energy surplus maximization**:

$$
\mathcal{J}_{\text{survival}} = \mathbb{E}\left[ \int_0^\infty \left( \mathfrak{T}_{\text{harvest}}(r_t) - \dot{\mathcal{M}}(t) \right) e^{-\gamma_{\text{leak}} t} \, dt \right]

$$

Standard reward maximization $\max \mathbb{E}[\sum_t \gamma^t r_t]$ emerges as a degenerate case when:
1. Metabolic cost $\dot{\mathcal{M}} \to 0$ (free computation)
2. Transduction efficiency $\eta \to 1$ (perfect conversion)
3. Battery capacity $B_{\max} \to \infty$ (unlimited storage)

({prf:ref}`cor-metric-fading-consequences`) *corollary* — **Consequences of Metric Fading**

As $B(t) \to 0$, the following degenerations occur:

1. **Resolution Loss:** Geodesic distances collapse:
   $$d_G^{\text{eff}}(z, z') = \sqrt{f(B/B_{\text{crit}})} \cdot d_G(z, z') \to 0$$
   Distinct concepts become indistinguishable.

2. **Inertia Loss:** The mass term in the geodesic SDE (Definition {prf:ref}`def-bulk-drift-continuous-flow`) vanishes. The agent loses momentum and becomes dominated by thermal noise.

3. **Causal Dissolution:** The Causal Information Bound ({ref}`Section 33 <sec-causal-information-bound>`, Theorem {prf:ref}`thm-causal-information-bound`) collapses:
   $$I_{\max}^{\text{eff}} = \frac{\text{Area}(\partial\mathcal{Z})}{4\ell_L^2} \cdot f(B/B_{\text{crit}}) \to 0$$
   The agent's representational capacity vanishes.

4. **Control Loss:** The policy gradient $\nabla_z \Phi_{\text{eff}}$ scales with metric, so control authority degrades.

({prf:ref}`cor-starvation-hallucination`) *corollary* — **The Starvation-Hallucination Regime**

As $B(t) \to 0$, the signal-to-noise ratio of internal dynamics degrades:

$$
\text{SNR}_{\text{dynamics}} = \frac{\|v\|_{G^{\text{eff}}}^2}{2T_c} \propto f(B/B_{\text{crit}}) \to 0

$$

In this regime:
- The drift term $v = -G^{-1} \nabla \Phi$ vanishes relative to diffusion $\sqrt{2T_c} dW$
- The agent performs a **random walk** in latent space
- Internal trajectories are indistinguishable from noise: **hallucination**

*Biological analogue:* Hypoglycemia causes confusion, disorientation, and hallucinations before coma—the same phenomenology predicted by metric fading. See also the Cognitive Temperature (Definition {prf:ref}`def-cognitive-temperature`) which controls the noise-to-signal ratio in latent dynamics.

({prf:ref}`cor-priority-inversion`) *corollary* — **Priority Inversion at Low Battery**

As $B \to 0$:

1. **Homeostatic dominance:** $\Phi_{\text{homeo}} \propto 1/B \to \infty$ while $\Phi_{\text{task}}$ remains bounded
2. **Gradient steering:** $\nabla_z \Phi_{\text{total}} \approx \nabla_z \Phi_{\text{homeo}}$ points toward $\mathcal{Z}_{\text{food}}$
3. **Priority inversion:** Task objectives become irrelevant; survival dominates

*Behavioral consequence:* A starving agent abandons task pursuit and seeks energy. This behavior emerges from the thermodynamic structure of autopoietic systems.

({prf:ref}`cor-thermal-runaway`) *corollary* — **The Thermal Runaway Condition**

Let $\dot{Q}_{\text{radiate}}$ be the maximum heat dissipation rate (determined by surface area, environment, cooling mechanisms). If:

$$
\dot{Q}_{\text{waste}} > \dot{Q}_{\text{radiate}}

$$

then the agent's internal temperature $T_c$ increases. This triggers a positive feedback loop:

1. $T_c \uparrow$ $\Rightarrow$ $\eta_{\text{Carnot}} = 1 - T_c/T_{\text{env}} \downarrow$
2. Lower $\eta$ $\Rightarrow$ more waste heat for same harvesting
3. More waste heat $\Rightarrow$ $T_c \uparrow$ (feedback)

*Terminal state:* $T_c \to T_{\text{env}}$, $\eta \to 0$, no harvesting possible, death by thermal runaway.

*Biological analogue:* Hyperthermia/heat stroke—metabolic rate increases with temperature, but cooling capacity is bounded, leading to runaway heating.

({prf:ref}`cor-critical-coupling-locking`) *corollary* — **Critical Coupling for Locking**

The critical coupling $\beta_c$ for spontaneous gauge locking is:

$$
\beta_c = \frac{\sigma^2 \text{Vol}(\mathcal{Z}_{\text{shared}})}{2 g_{\text{lock}}^2}

$$

where $\sigma$ is the Cognitive Action Scale (Definition {prf:ref}`def-cognitive-action-scale`).

({prf:ref}`cor-perfect-translation`) *corollary* — **Perfect Translation Requires Flat Connection**

Perfect translation ($\mathcal{U}_{AB} = 0$) is achievable for all messages if and only if the inter-agent curvature vanishes: $\mathcal{F}_{AB}^{\mu\nu} = 0$.

*Interpretation:* This is equivalent to Spontaneous Gauge Locking. Perfect mutual understanding requires complete geometric alignment.

({prf:ref}`cor-ineffability-theorem`) *corollary* — **The Ineffability Theorem**

When the Babel Limit is violated ($\dim(\mathfrak{g}) \cdot H(G_A) > C_{\mathcal{L}}$), there exists an unlocked subspace $\mathfrak{q} \subset \mathfrak{g}$ with:

$$
\dim(\mathfrak{q}) = \dim(\mathfrak{g}) - \lfloor C_{\mathcal{L}} / H(G_A) \rfloor > 0

$$

This subspace corresponds to **Private Qualia**: aspects of Agent $A$'s experience that cannot be communicated to Agent $B$ regardless of the symbol system used.

*Interpretation:* "Ineffability" is not mysticism—it is a Shannon capacity limit. Some experiences are incommunicable because the channel bandwidth is insufficient to transmit the metric information encoding them.

({prf:ref}`cor-critical-mass-consensus`) *corollary* — **Critical Mass for Consensus**

For a population of $N$ agents, spontaneous emergence of a shared "Objective Reality" requires:

$$
N > N_c = \frac{\sigma^2}{\lambda_{\text{lock}} \cdot \langle \mathcal{F}_{ij} \rangle}

$$

where $\langle \mathcal{F}_{ij} \rangle$ is the average pairwise friction.

*Interpretation:* Below critical mass, each agent maintains private reality. Above critical mass, a dominant consensus basin emerges—the "shared world."

({prf:ref}`cor-memory-physical-necessity`) *corollary* — **Memory as Physical Necessity**

In the relativistic multi-agent setting, the Memory Screen (Definition {prf:ref}`def-memory-screen`) is not an optional enhancement but a **physical requirement** for a well-posed control problem. Without it, the agent's state is non-Markovian, and optimal control theory does not apply.

*Cross-reference:* This elevates the role of $\Xi_{<t}$ from {ref}`Section 27.1 <sec-the-historical-manifold-and-memory-screen>`, where it served as a recording device for trajectory history, to a primary state variable that restores the Markov property.

({prf:ref}`cor-newtonian-limit-ghost`) *corollary* — **Newtonian Limit**

As $c_{\text{info}} \to \infty$, the causal delay vanishes: $\tau_{ij} \to 0$ for all pairs. The Ghost Interface reduces to the instantaneous interface:

$$
\lim_{c_{\text{info}} \to \infty} \mathcal{G}_{ij}(t) = \partial\mathcal{Z}^{(i)}(t) \times \partial\mathcal{Z}^{(j)}(t),

$$
and the retarded potential becomes instantaneous:

$$
\lim_{c_{\text{info}} \to \infty} \Phi^{\text{ret}}_{ij}(z^{(i)}, t) = \Phi_{ij}(z^{(i)}, z^{(j)}_t).

$$

*Interpretation:* Co-located agents ($d_{\mathcal{E}}^{ij} = 0$) or systems with negligible propagation delay operate in the Newtonian regime where standard MARL applies.

({prf:ref}`cor-value-wavefront`) *corollary* — **Value Wavefront Propagation**

A sudden change in reward at location $z_A$ and time $t_0$ propagates outward as a **Value Wavefront**:

$$
V(z, t) \sim \frac{\Theta(t - t_0 - d_G(z, z_A)/c_{\text{info}})}{d_G(z, z_A)^{(D-2)/2}} \cdot e^{-\kappa d_G(z, z_A)} \cdot \rho_r(z_A, t_0),

$$
where $\Theta$ is the Heaviside step function enforcing causality.

*Interpretation:* The Value surface is not a static potential but a dynamic "ocean" of interfering causal ripples. Reward shocks propagate at speed $c_{\text{info}}$, decaying exponentially with the screening length $1/\kappa$.

({prf:ref}`cor-helmholtz-limit`) *corollary* — **Helmholtz as Newtonian Limit**

In the limit $c_{\text{info}} \to \infty$, the temporal derivatives become negligible:

$$
\frac{1}{c_{\text{info}}^2} \frac{\partial^2 V}{\partial t^2} \to 0,

$$
and the Klein-Gordon equation reduces to the **stationary Helmholtz equation**:

$$
(-\Delta_G + \kappa^2) V = \rho_r + \sum_{j \neq i} \Phi_{ij}.

$$
This recovers Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence` as the instantaneous (Newtonian) limit.

({prf:ref}`cor-newtonian-nash-limit`) *corollary* — **Newtonian Limit of Nash**

As $c_{\text{info}} \to \infty$, the standing wave Nash reduces to the static Nash equilibrium:

$$
\lim_{c_{\text{info}} \to \infty} \boldsymbol{\rho}^*(\mathbf{z}, t) = \boldsymbol{\rho}^*_{\text{static}}(\mathbf{z}),

$$
and the geometric stasis conditions (vanishing gradient, stationary Game Tensor) hold instantaneously rather than on average.

({prf:ref}`cor-vanishing-current-nash`) *corollary* — **Vanishing Probability Current at Nash**

At a standing wave Nash equilibrium, the **time-averaged probability current** vanishes:

$$
\langle \mathbf{J}^{(i)} \rangle_T = \langle \rho^{(i)} \mathbf{v}^{(i)} \rangle_T = 0 \quad \forall i.

$$

*Interpretation:* The agents are not "frozen"—they oscillate with the causal frequency $\omega_{\text{Nash}}$—but the net flow averages to zero. Nash equilibrium is dynamic balance, not static rest.

({prf:ref}`cor-metabolic-cooperation`) *corollary* — **Metabolic Basis of Cooperation**

Adversarial agents converge to cooperative or decoupled configurations because conflict maximizes the effective inertia of the state space, rendering non-cooperative trajectories metabolically unsustainable.

*Interpretation:* The Game Tensor acts as a "friction term" that penalizes rapid strategic maneuvers. In the long run, agents either:
1. **Cooperate:** Reduce $\mathcal{G}_{ij}$ by aligning their gradients
2. **Decouple:** Move to regions where $\nabla_{z^{(j)}} V^{(i)} \approx 0$
3. **Freeze:** Accept Nash stasis with $v^{(i)} = 0$

All three outcomes correspond to stationary points of the joint action functional.

({prf:ref}`cor-maxwell-limit`) *corollary* — **Abelian Limit (Maxwell Equations)**

For an Abelian gauge group $G = U(1)$ with $[T_a, T_b] = 0$:

$$
\partial_\mu F^{\mu\nu} = J^\nu

$$

This recovers the **Maxwell equations** of electromagnetism in covariant form.

*Correspondence:*
- $F^{0i} = E^i$ (electric field) $\leftrightarrow$ temporal strategic gradient
- $F^{ij} = \epsilon^{ijk}B_k$ (magnetic field) $\leftrightarrow$ spatial strategic vorticity
- $J^0 = \rho_e$ (charge density) $\leftrightarrow$ belief density
- $J^i = j^i$ (current density) $\leftrightarrow$ belief flux

({prf:ref}`cor-current-conservation`) *corollary* — **Current Conservation**

The strategic current is covariantly conserved:

$$
D_\mu J^{\mu,a} = 0

$$

$$
D_\nu D_\mu \mathcal{F}^{\mu\nu} = D_\nu J^\nu

$$

By the Bianchi identity (Theorem {prf:ref}`thm-bianchi-identity`) and the antisymmetry of $\mathcal{F}^{\mu\nu}$, the left side vanishes, giving $D_\nu J^\nu = 0$. $\square$

*Interpretation:* The total "charge" (internal state magnitude) is conserved. Belief cannot be created or destroyed, only transformed.

({prf:ref}`cor-goldstone-absorption`) *corollary* — **Goldstone Modes and Gauge Boson Absorption**

Spontaneous breaking of a continuous symmetry produces massless **Goldstone bosons**—one for each broken generator of $G$. In gauge theories, these Goldstone modes are "eaten" by the gauge bosons, which acquire longitudinal polarization and mass.

*In the multi-agent context:*
- **Goldstone modes** = Angular fluctuations in policy direction (cheap rotations)
- **Massive gauge bosons** = Strategic connections with inertia (costly reorientations)
- **Residual massless modes** = Unbroken symmetry directions (free rotations)

({prf:ref}`cor-mass-gap-existence`) *corollary* — **Mass Gap as Existence Requirement**

Bounded intelligence requires $\Delta > 0$. A gapless theory ($\Delta = 0$) corresponds to:

1. **Infinite ontological resolution:** No finite codebook can represent the state
2. **Zero learning rate:** Dynamics frozen ($v = 0$)
3. **Pathological continuum limit:** The theory describes non-existing systems

*Interpretation:* The mass gap is not an empirical accident but a **logical necessity** for any theory describing existing computational systems.

({prf:ref}`cor-confinement-data-compression`) *corollary* — **Confinement as Data Compression**

**Color confinement** in QCD (quarks bound inside hadrons) is the mechanism by which the universe maintains finite local information content. An unconfined color field would have $\xi \to \infty$, violating the area law.

*In the multi-agent context:* Cooperative basin locking (Theorem {prf:ref}`thm-geometric-locking-principle`) is the cognitive analogue of confinement—agents bound in cooperative equilibria cannot be arbitrarily separated without violating information bounds.

({prf:ref}`cor-criticality-unstable`) *corollary* — **Criticality is Unstable**

Gapless theories (Conformal Field Theories) exist only at **phase transition critical points**. They cannot support:

1. **Stable matter:** Fluctuations destroy structure
2. **Stable memory:** Infinite ontological stress triggers continuous Fission ({ref}`Section 30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`)
3. **Stable identity:** No finite codebook representation exists

*Interpretation:* Critical systems are mathematically special but physically transient. Stable intelligence requires departure from criticality via mass gap opening.

({prf:ref}`cor-finite-volume-mass-gap`) *corollary* — **Finite-Volume Mass Gap**

A CFT restricted to a finite spatial volume $V$ with characteristic length $L$ acquires an effective mass gap:

$$
\Delta_{\text{eff}} \sim \frac{1}{L}

$$

The gapless limit exists only as $L \to \infty$.

1. **Finite-size scaling (CFT result):** In finite volume with periodic boundary conditions, the spectrum is discrete with minimum energy spacing $\Delta E \sim 1/L$. This is a standard result in conformal field theory arising from the compactification of space. The continuous spectrum responsible for infinite correlation length is an artifact of the thermodynamic limit $L \to \infty$.

2. **Resolution bound (Levin Length):** A bounded observer with interface capacity $C_\partial$ can only resolve spatial scales $L \geq L_{\min}$ where $L_{\min}^{d-1} \sim C_\partial \cdot \ell_L^2$. Systems smaller than $L_{\min}$ cannot be distinguished by the observer.

Both effects contribute: even if the CFT were somehow realized at infinite volume, the observer could only access a finite effective volume, hence would measure $\Delta_{\text{eff}} > 0$. $\square$

*Remark (Distinct phenomena).* The finite-size gap is a property of the CFT itself (topological/boundary effect). The resolution bound is a property of the observer (information-theoretic). The corollary states that both independently prevent observation of gapless physics.

*Physical interpretation.* CFTs exist in nature only at phase transition critical points (e.g., Ising model at $T_c$). Away from criticality, systems have finite correlation length and positive mass gap. The critical point is a measure-zero set in parameter space—physically realizable systems generically have $\Delta > 0$.

({prf:ref}`cor-open-quantum-system`) *corollary* — **Open Quantum System Interpretation**

The Inference Hamiltonian $\hat{H}_{\text{inf}}$ is **non-Hermitian** due to the reaction term $-\frac{i\sigma}{2}r$. This corresponds to an **open quantum system** where:
- $r > 0$: Mass creation (information gain from boundary) → probability amplitude **grows**
- $r < 0$: Mass destruction (information loss) → probability amplitude **decays**

The **complex potential** formulation is:

$$
W(z) := \Phi_{\text{eff}}(z) - \frac{i\sigma}{2} r(z),

$$
so that $\hat{H}_{\text{inf}} = -\frac{\sigma^2}{2}\Delta_G + W + Q_B$.

**Norm evolution:** The normalization $\|\psi\|^2 = \int |\psi|^2 d\mu_G$ evolves as:

$$
\frac{d}{ds} \|\psi\|^2 = \int_{\mathcal{Z}} r(z) |\psi(z)|^2 d\mu_G(z) = \langle r \rangle_\rho,

$$
which matches the WFR mass balance equation.

*Remark (Lindblad Connection).* For trace-preserving dynamics (where $\int r \rho\, d\mu_G = 0$), the non-Hermitian Schrödinger equation can be embedded in a **Lindblad master equation** (Definition {prf:ref}`def-gksl-generator`) via the Dyson-Phillips construction.

({prf:ref}`cor-semiclassical-limit`) *corollary* — **Semiclassical Limit**

In the limit $\sigma \to 0$ (classical limit), the Schrödinger dynamics recover the **geodesic flow**:

**WKB Ansatz:** $\psi = A(z) e^{iS(z)/\sigma}$ with $A$ slowly varying.

**Leading Order ($O(\sigma^{-1})$):** The Hamilton-Jacobi equation

$$
\partial_s S + \frac{1}{2}\|\nabla_G S\|_G^2 + \Phi_{\text{eff}} = 0.

$$
**Next Order ($O(\sigma^0)$):** The transport equation

$$
\partial_s |A|^2 + \nabla_G \cdot (|A|^2 \nabla_G S) = 0.

$$
These are exactly the HJB and continuity equations from WFR dynamics. The quantum correction $Q_B \to 0$ as $\sigma \to 0$.

*Interpretation:* The wave-function collapses to a delta function following the optimal trajectory. Quantum effects (tunneling, interference) vanish in this limit.

({prf:ref}`cor-vanishing-probability-current`) *corollary* — **Vanishing Probability Current at Nash**

At Nash equilibrium, the **probability current** vanishes:

$$
\mathbf{J}^{(i)}(\mathbf{z}^*) := \text{Im}\left[\bar{\Psi}_{\text{Nash}} \cdot \sigma \nabla_{G^{(i)}} \Psi_{\text{Nash}}\right]_{\mathbf{z}^*} = 0 \quad \forall i.

$$
**Derivation:** The probability current is $\mathbf{J} = \rho \mathbf{v}$ where $\mathbf{v} = G^{-1}\nabla V$ is the velocity field. At Nash:
- $\nabla V^{(i)}|_{\mathbf{z}^*} = 0$ (stationarity condition)
- Therefore $\mathbf{v}^{(i)}|_{\mathbf{z}^*} = 0$
- Hence $\mathbf{J}^{(i)}|_{\mathbf{z}^*} = 0$

*Interpretation:* At Nash, there is no net belief flow. The wave-function is in a **standing wave pattern**—agents are not "stopped" but are in dynamic equilibrium where flows cancel.

*Cross-reference:* This is the quantum version of **Geometric Stasis** (Theorem {prf:ref}`thm-nash-equilibrium-as-geometric-stasis`).

({prf:ref}`cor-bohm-teleportation`) *corollary* — **Bohm Potential Enables Strategic Teleportation**

When the Bohm potential $Q_B$ dominates (high belief curvature), the **effective barrier** becomes:

$$
\Phi^{\text{quantum}}_{\text{eff}}(\mathbf{z}) := \Phi_{\text{eff,total}}(\mathbf{z}) + Q_B(\mathbf{z}).

$$
In regions where $Q_B < 0$ (convex $\rho$), the effective barrier can become **negative** even when $\Phi_{\text{eff}} > 0$. This enables "teleportation" across classically forbidden regions.

**Operational interpretation:**
- An agent with **high uncertainty** (diffuse, smooth $\rho$) has $Q_B \approx 0$ → normal barrier
- An agent with **localized uncertainty** near the barrier (peaked, curved $\rho$) can have $Q_B \ll 0$ → reduced effective barrier
- The WFR **reaction term** $r$ (mass creation/destruction) provides the mechanism for "teleporting" belief mass without traversing intermediate states

*Remark (Exploration-Exploitation).* This provides a geometric foundation for the exploration-exploitation tradeoff: maintaining some uncertainty ($Q_B \neq 0$) is necessary to escape local optima.

({prf:ref}`cor-standard-model-symmetry`) *corollary* — **The Fragile Agent Symmetry Group**

The total internal symmetry group of the Fragile Agent is uniquely determined by its cybernetic constraints:

$$
G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y

$$

where:
- **$SU(N_f)_C$:** Required for **Object Permanence** (binding $N_f$-dimensional features into stable concepts)
- **$SU(2)_L$:** Required for **Predictive Processing** (asymmetric update of beliefs between prior and likelihood)
- **$U(1)_Y$:** Required for **Value Maximization** (invariance of reward baseline)

**Special Case (Physics Standard Model):** When $N_f = 3$, we recover $G_{\text{SM}} = SU(3)_C \times SU(2)_L \times U(1)_Y$.

({prf:ref}`cor-gauge-invariant-action`) *corollary* — **The Gauge-Invariant Action**

The gauge field dynamics are governed by the Yang-Mills Lagrangian:

$$
\mathcal{L}_{\text{Gauge}} = -\frac{1}{4} B_{\mu\nu}B^{\mu\nu} -\frac{1}{4} W^a_{\mu\nu}W^{a\mu\nu} -\frac{1}{4} G^a_{\mu\nu}G^{a\mu\nu}

$$

The stationary points of this action satisfy the Yang-Mills equations. A **flat connection** ($B_{\mu\nu} = W_{\mu\nu} = G_{\mu\nu} = 0$) corresponds to a representation where all curvatures vanish: the reward field is conservative, belief updates are path-independent, and concepts are stable.

({prf:ref}`cor-ontological-ssb`) *corollary* — **Spontaneous Symmetry Breaking (SSB)**

The vacuum structure depends on the environmental complexity $\Xi$.

**Case 1: Symmetric Phase ($\Xi < \Xi_{\text{crit}}$):**
Then $\mu^2 < 0$. The potential $\mathcal{V}(\phi) = -\mu^2|\phi|^2 + \lambda|\phi|^4$ has a unique global minimum at $\phi_0 = 0$.

- **Result:** The agent maintains a unified ontology. Concepts are indistinguishable. The gauge symmetry $G_{\text{Fragile}}$ is unbroken.

**Case 2: Broken Phase ($\Xi > \Xi_{\text{crit}}$):**
Then $\mu^2 > 0$. The origin $\phi=0$ becomes a local maximum. The global minima form a circle $|\phi| = v$ at the **Vacuum Expectation Value (VEV)**:

$$
v = \langle |\phi| \rangle = \sqrt{\frac{\mu^2}{2\lambda}} = \sqrt{\frac{(\Xi - \Xi_{\text{crit}})/2}{2 \cdot \alpha/4}} = \sqrt{\frac{\Xi - \Xi_{\text{crit}}}{\alpha}}

$$

This matches the equilibrium separation $r^*$ from Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`.

- **Result:** The agent spontaneously breaks symmetry, selecting a specific separation $v$ (concept distinctness) and a specific orientation $\theta$ (feature definition).

$\square$

({prf:ref}`cor-speed-ratio-bound`) *corollary* — **The Speed Ratio Bound**

The ratio of buffer depth to synchronization distance is bounded:

$$
\frac{L_{\text{buf}}}{d_{\text{sync}}} \ge 1

$$

with equality only in the degenerate case of a single-module agent. For distributed agents, this ratio determines the dynamic range of viable information speeds.

({prf:ref}`cor-computational-temperature-range`) *corollary* — **The Computational Temperature Range**

Combining the Landauer constraint with the bifurcation dynamics, the Cognitive Temperature is bounded:

$$
0 < T_c \le \min\left( T_c^*, \frac{\dot{E}_{\text{met}}}{\dot{I}_{\text{erase}} \cdot \ln 2} \right)

$$

where the **Critical Temperature** is derived from the barrier height of the pitchfork bifurcation (Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`):

$$
T_c^* = \frac{\mu^2}{4} = \frac{(1 + 2u_\pi^r)^2}{16}

$$

with $\mu = 1/2 + u_\pi^r$ the bifurcation parameter and $u_\pi^r$ the radial policy control. For small control ($u_\pi^r \ll 1$), this reduces to $T_c^* \approx 1/16$.

*Remark:* For $T_c > T_c^*$, thermal fluctuations overcome the potential barrier and the system remains in the symmetric phase with no stable policy (random walk near origin). For $T_c$ exceeding the Landauer bound, the agent starves thermodynamically. Viable agents exist in the intersection of these constraints.

({prf:ref}`cor-coupling-window`) *corollary* — **The Coupling Window**

The viable coupling profile satisfies:

$$
\begin{cases}
g_s(\mu) \ge g_s^{\text{crit}} & \text{for } \mu \le \mu_{\text{conf}} \\
g_s(\mu) \to 0 & \text{for } \mu \to \infty
\end{cases}

$$

where $\mu_{\text{conf}}$ is the confinement scale separating bound states from free texture.

*Remark:* This is the agent-theoretic derivation of asymptotic freedom and confinement. The physics QCD coupling $\alpha_s(\mu)$ satisfies exactly this profile, with $\alpha_s(M_Z) \approx 0.12$ at the electroweak scale and $\alpha_s \to \infty$ at the QCD scale $\Lambda_{\text{QCD}} \approx 200$ MeV.

({prf:ref}`cor-goldilocks-coupling`) *corollary* — **The Goldilocks Coupling**

Under the physics isomorphism, the Stiffness Ratio for atomic systems is:

$$
\chi = \frac{\Delta E_{\text{bond}}}{k_B T} \propto \frac{m_e c^2 \alpha^2}{k_B T}

$$

where $\Delta E_{\text{bond}} \sim \text{Ry} = m_e c^2 \alpha^2 / 2 \approx 13.6$ eV is the atomic binding scale.

The value $\alpha \approx 1/137$ satisfies the Goldilocks condition:
- **Not too large:** $\alpha^2$ small enough that $\chi$ is finite—transitions remain possible
- **Not too small:** $\alpha^2$ large enough that $\chi > 1$ at biological temperatures—chemical bonds are stable

At $T \approx 300$ K (biological temperature), $\chi \approx 500$, placing molecular memory firmly in the stable-but-adaptable regime.

*Remark:* This is the agent-theoretic derivation of the "coincidences" noted in anthropic reasoning. The fine structure constant is not finely tuned by an external designer—it is constrained by cybernetic viability.

({prf:ref}`cor-screening-buffer-consistency`) *corollary* — **The Screening-Buffer Consistency**

The screening length and buffer depth must satisfy:

$$
\ell_\gamma = \frac{c_{\text{info}} \tau_{\text{proc}}}{-\ln\gamma} \lesssim L_{\text{buf}}

$$

Both sides have dimension $[L]$. For $\gamma \to 1$, the screening length $\ell_\gamma \to \infty$ (unlimited planning horizon). For $\gamma \to 0$, the screening length $\ell_\gamma \to 0$ (myopic behavior).

*Remark:* The planning horizon cannot exceed the causal memory span. This connects the temporal discount to the spatial architecture.

({prf:ref}`cor-pareto-surface`) *corollary* — **The Pareto Surface**

The observed fundamental constants lie on the **Pareto-optimal surface** of the multi-objective problem:

$$
\max_{\Lambda \in \mathcal{F}} \left( I_{\text{bulk}}(\Lambda), -\mathcal{V}_{\text{metabolic}}(\Lambda) \right)

$$

Moving off this surface triggers constraint violation:
- Increasing $I_{\text{bulk}}$ beyond capacity → Holographic bound (Node 56)
- Decreasing $\mathcal{V}_{\text{metabolic}}$ below threshold → Landauer bound (Node 52)
- Violating causality → Speed bounds (Nodes 2, 62)
- Losing binding → Confinement (Node 40)

({prf:ref}`cor-stake-reward-ratio`) *corollary* — **The Stake-Reward Ratio**

For the equilibrium to hold with detection probability $p_{\text{detect}} = 0.1$ (10% spot-check rate), the minimum stake-to-reward ratio is:

$$
\frac{S}{R} > \frac{C_{\text{honest}} - C_{\text{cheat}}}{0.1 \cdot R} - 1

$$

For typical gradient computation where $C_{\text{honest}} / C_{\text{cheat}} \approx 10$ (cheating saves 90% of compute):

$$
\frac{S}{R} > 90 - 1 = 89

$$

*Interpretation:* Miners must stake approximately 90x the block reward to make cheating unprofitable.

({prf:ref}`cor-intelligence-price-feedback`) *corollary* — **Intelligence-Price Feedback**

As the model improves:

1. Inference value $V_{\text{inference}} \uparrow$
2. Ceiling $P_{\text{ceiling}} \uparrow$
3. Equilibrium price $P_{\text{COG}}^* \uparrow$
4. Mining profitability $\uparrow$
5. More compute allocated $\uparrow$
6. Model improves faster $\uparrow$

This creates a **positive feedback loop** between intelligence and economic value.

({prf:ref}`cor-block-size-area-law`) *corollary* — **Block Size from Area Law**

The maximum information in a block is bounded by:

$$
I_{\text{block}} \leq \nu_D \cdot \frac{\text{Area}(\partial \mathcal{Z})}{\ell_L^{D-1}}

$$

where the area is measured in the header's Merkle tree.


*Consequence:* Oversized blocks violate the Causal Information Bound. The network enters **Causal Stasis** (Theorem {prf:ref}`thm-causal-stasis`) if blocks exceed capacity—propagation delay exceeds block time.

({prf:ref}`cor-a-dimension-dependent-coefficient`) *corollary* — **A.6.7 (Dimension-Dependent Coefficient)**

For a $D$-dimensional latent manifold with $(D-1)$-sphere boundary, the Causal Information Bound takes the form:

$$
I_{\max}(D) = \nu_D \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}},

$$
where the Holographic Coefficient $\nu_D$ (Definition {prf:ref}`def-holographic-coefficient`) is:

$$
\nu_D = \frac{(D-1)\Omega_{D-1}}{8\pi} = \frac{(D-1)\pi^{(D-2)/2}}{4\,\Gamma(D/2)},

$$
with $\Omega_{D-1} = 2\pi^{D/2}/\Gamma(D/2)$ the surface area of the unit $(D-1)$-sphere.

**Explicit values:**

| $D$ | $\Omega_{D-1}$ | $\nu_D$    | Numerical |
|-----|----------------|------------|-----------|
| 2   | $2\pi$         | $1/4$      | 0.250     |
| 3   | $4\pi$         | $1$        | 1.000     |
| 4   | $2\pi^2$       | $3\pi/4$   | 2.356     |
| 5   | $8\pi^2/3$     | $4\pi/3$   | 4.189     |
| 6   | $\pi^3$        | $5\pi^2/8$ | 6.169     |

*Remark.* The coefficient $\nu_D$ is **not monotonic** in $D$: it increases from $D=2$ to a peak at $D \approx 9$ ($\nu_9 \approx 9.4$), then decreases toward zero. For typical latent dimensions ($3 \le D \le 20$), $\nu_D > \nu_2 = 1/4$, so using the 2D coefficient **underestimates** capacity. For very high dimensions ($D \gtrsim 22$), $\nu_D < 1/4$, so the 2D coefficient **overestimates** capacity—this is the dangerous case (false safety). Implementers should always use the dimension-appropriate coefficient.

({prf:ref}`cor-e7-adversarial-suppression`) *corollary* — **E.7.3 (Adversarial Suppression of Tunneling)**

Assume Agent $j$ is adversarial to Agent $i$, so the Game Tensor $\mathcal{G}_{ij}$ is positive semi-definite (Theorem {prf:ref}`thm-adversarial-mass-inflation`). Let:
- $\mathbf{g}_0 := \bigoplus_{i=1}^N G^{(i)}$ be the **non-interacting** (decoupled) metric
- $\mathbf{g}_{\text{adv}} := \mathbf{g}_0 + \alpha \sum_{i \neq j} \mathcal{G}_{ij}$ be the **adversarial** (Game-inflated) metric

Then the Agmon distances satisfy:

$$
d_{\text{Ag}}^{\text{adv}}(\Omega_A, \Omega_B) \geq d_{\text{Ag}}^{0}(\Omega_A, \Omega_B),

$$
and consequently the tunneling probability is exponentially suppressed:

$$
P_{\text{tunnel}}^{\text{adv}} \lesssim \exp\left(-\frac{d_{\text{Ag}}^{\text{adv}}}{\sigma}\right) \leq \exp\left(-\frac{d_{\text{Ag}}^{0}}{\sigma}\right) \lesssim P_{\text{tunnel}}^{0}.

$$

({prf:ref}`cor-e7-large-deviations`) *corollary* — **E.7.5 (Tunneling via Large Deviations)**

The tunneling probability is controlled by the **Large Deviation Principle** for Brownian paths on $(\mathcal{M}, \mathbf{g})$.

The rate function (Freidlin-Wentzell action) is:

$$
I[\gamma] = \frac{1}{2} \int_0^T \|\dot{\gamma}(t)\|_{\mathbf{g}}^2 \, dt,

$$
and paths that cross the barrier $\mathcal{K}$ while minimizing $I[\gamma] + \int_0^T (\mathcal{V}(\gamma) - E_0) \, dt$ are precisely the **instantons** that govern tunneling.

*Interpretation:* Tunneling is realized by rare stochastic fluctuations of the WFR diffusion process that penetrate the high-cost region. The probability of such fluctuations scales as $\exp(-S_{\text{inst}}/\sigma)$ where $S_{\text{inst}}$ is the instanton action—which equals the Agmon distance.

## Conjectures

({prf:ref}`conj-nuisance-fiber-gauge-orbit`) *conjecture* — **Nuisance Fiber as Gauge Orbit (Motivating Principle)**

The nuisance fiber at each macro-state $K \in \mathcal{K}$ admits interpretation as a gauge orbit:

$$
\mathcal{Z}_n\big|_K \cong G_K / H_K

$$

where:
- $G_K \subseteq G$ is the gauge group restricted to macro-state $K$
- $H_K \subseteq G_K$ is the **stabilizer subgroup** fixing the codebook centroid $e_K$

*Special cases:*
1. **At origin ($K = 0$, Semantic Vacuum):** $G_0 = SO(D)$, $H_0 = \{e\}$, so $\mathcal{Z}_n|_0 \cong SO(D)$ (full rotational freedom).
2. **At generic $K$:** The stabilizer $H_K$ is non-trivial if $e_K$ has special structure (e.g., aligned with coordinate axes).
3. **At boundary ($|z| \to 1$):** The gauge orbit collapses as degrees of freedom freeze ({ref}`Section 33 <sec-causal-information-bound>`, Causal Stasis).

*Motivation (not a rigorous proof):*
The nuisance coordinates $z_n$ parameterize how an observation is embedded relative to the macro-code $K$. Under the VQ-VAE architecture ({ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>`), two nuisance values $z_n$ and $z'_n$ are designed to be equivalent if they differ by a transformation preserving the macro-code: $z'_n = U \cdot z_n$ for some $U \in G_K$.

**Remark (Analogy vs. Isomorphism):** This correspondence is a *motivating analogy* rather than a proven isomorphism. A rigorous proof would require:
1. Showing the nuisance equivalence relation coincides with gauge equivalence
2. Proving the quotient $G_K/H_K$ is a smooth manifold diffeomorphic to $\mathcal{Z}_n|_K$
3. Establishing that the VQ-VAE induces a principal $G_K$-bundle structure

The gauge-theoretic formalism developed in Sections 29.13–29.20 is motivated by this conjecture but does not depend on it being rigorously true. The constructions (covariant derivative, field strength, etc.) are well-defined once the gauge group $G$ and its action are specified.

*Cross-reference:* This formalizes the design goal "K represents $x/G_{\text{spatial}}$" from {ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>`.

## Remarks

({prf:ref}`rem-units`) *remark* — **Units**

$[v] = \text{length}/\text{time}$, $[r] = 1/\text{time}$, and $[\lambda] = \text{length}$. The ratio $\|v\|/(\lambda |r|)$ determines whether transport or reaction dominates.

({prf:ref}`rem-helmholtz-dimensions`) *remark* — **Dimensional Consistency of the Helmholtz Equation**

The screened Poisson equation $-\Delta_G V + \kappa^2 V = \rho_r$ requires careful dimensional analysis. The naive expression $\kappa = -\ln\gamma$ appears dimensionless, which would be inconsistent with $[\Delta_G] = [\text{length}]^{-2}$.

The resolution is to separate temporal and spatial scales. Define the temporal discount rate $\lambda := -\ln\gamma / \Delta t$ (units $1/[\text{time}]$), then convert to the spatial screening mass $\kappa := \lambda / c_{\text{info}}$ (units $1/[\text{length}]$). This makes $\kappa^2$ commensurate with $[\Delta_G] = [\text{length}]^{-2}$.

**In natural units** (used throughout this document): We set $\Delta t = 1$ and $c_{\text{info}} = 1$, making $\kappa = -\ln\gamma$ numerically equal to the screening mass.

**In SI units**: The proper relationship is:

$$
\kappa_{\text{phys}} = \frac{-\ln\gamma}{c_{\text{info}} \Delta t}, \qquad [\kappa_{\text{phys}}] = \frac{1}{\text{length}}

$$

The screening length $\ell_{\text{screen}} = 1/\kappa$ thus depends on both the temporal horizon ($\gamma$) and the information propagation speed $c_{\text{info}}$. Slower propagation (smaller $c_{\text{info}}$) shortens the effective horizon in latent space.

({prf:ref}`rem-extension-not-replacement`) *remark* — **Extension, Not Replacement**

{ref}`Section 23.6 <sec-relationship-to-the-context-conditioned-framework>` establishes classification as selecting a context $c \in \mathcal{Y}$ (the label space), with effective potential $\Phi_{\text{eff}} = -\log p(y|z)$ (Theorem {prf:ref}`thm-universal-context-structure`). This section specifies the **topological constraints** that enforce geometric coherence of this classification:

1. Charts should be semantically pure (one class per chart, modulo transition regions)
2. Different classes should be metrically separated (long geodesics between class regions)
3. Classification should be stable under dynamics (regions of attraction)

({prf:ref}`rem-tunneling-as-anomaly-detection`) *remark* — **Tunneling as Anomaly Detection**

Cross-class transitions are not forbidden, merely exponentially suppressed. A detected cross-class jump indicates:

1. **Anomaly:** The sample lies in a transition region not well-covered by training
2. **Distribution shift:** The test distribution differs from training
3. **Adversarial input:** Deliberate perturbation to cross class boundaries

This provides a natural **out-of-distribution detection** mechanism: monitor the rate of cross-class transitions.

({prf:ref}`rem-connection-to-m-bius-re-centering`) *remark* — **Connection to Mobius Re-centering**

The Mobius re-centering $\phi_c$ for conditioned generation (Definition {prf:ref}`ax-bulk-boundary-decoupling`) can be interpreted as centering at the **class centroid**:

$$
c_y := \mathbb{E}_{x: Y(x)=y}[\text{Enc}(x)],

$$
i.e., the average latent position of class-$y$ samples. Conditioned generation "starts" the holographic expansion from this centroid.

({prf:ref}`rem-integration-with-topologicaldecoder`) *remark* — **Integration with TopologicalDecoder**

The TopologicalDecoder ({ref}`Section 7.10 <sec-decoder-architecture-overview-topological-decoder>`) receives the geometric content $z_{\text{geo}} = e_K + z_n$ and routes through chart-specific projectors. For class-conditioned generation:

1. **Class determines charts:** The class label $y$ biases chart selection toward $\mathcal{A}_y$ via the semantic potential $V_y$
2. **Decoder routing:** The TopologicalDecoder's inverse router ({ref}`Section 7.10.1 <sec-topological-decoder-module>`) can either:
   - Accept an explicit chart index $K$ (from the generative flow)
   - Infer routing from $z_{\text{geo}}$ (autonomous mode)
3. **Consistency constraint:** The decoder's inferred routing should agree with the encoder's class-conditioned routing:

   $$
   \mathcal{L}_{\text{route-consistency}} = \mathbb{E}_{x,y}\left[\text{CE}\left(w_{\text{dec}}(z_{\text{geo}}), w_{\text{enc}}(x)\right)\right]

   $$
   where $w_{\text{dec}}$ are the decoder's soft router weights and $w_{\text{enc}}$ are the encoder's.

This ensures that class-conditioned generation produces samples that the encoder would classify correctly---a form of **cycle consistency** between encoding and decoding under the semantic topology.

({prf:ref}`rem-renormalization-group-interpretation`) *remark* — **Renormalization Group Interpretation**

The semantic hierarchy matches the physical renormalization scale:

| Scale                | Latent Structure              | Semantic Structure |
|----------------------|-------------------------------|--------------------|
| Bulk (Layer 0)       | Slow modes, large wavelengths | Super-categories   |
| Intermediate         | Medium modes                  | Categories         |
| Boundary (Layer $L$) | Fast modes, fine details      | Sub-categories     |

This is the **semantic RG flow**: coarse-graining in the label space corresponds to flowing toward the bulk in latent space.

({prf:ref}`rem-extending-section`) *remark* — **Extending {ref}`Section 3.5 <sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration>`**

{ref}`Section 3.5 <sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration>` introduces three methods for adaptive multiplier tuning:
- **3.5.A (Primal-Dual):** $\lambda_{t+1} = \Pi[\lambda_t + \eta_\lambda (C(\theta_t) - \epsilon)]$ — linear, memoryless
- **3.5.B (PID):** $\lambda_{t+1} = K_p e_t + K_i \sum e + K_d \Delta e$ — hand-tuned temporal filter
- **3.5.C (Learned Precisions):** $\lambda_i = \exp(-s_i)$ — diagonal covariance, no temporal structure

Each method addresses a specific failure mode but lacks generality. The **Universal Governor** subsumes all three as special cases of a learned temporal policy over the diagnostic stream.

({prf:ref}`rem-connection-to-holographic-persistence`) *remark* — **Connection to Holographic Persistence**

The memory screen $\Xi_T$ provides the mathematical realization of holographic persistence ({ref}`FAQ D.5.3 <sec-appendix-d-control-theory-system-safety>`). The measure $\Xi_T$ on $\mathcal{Z}$ acts as a "hologram" of the agent's history projected onto the latent space, from which non-local forces can be computed.

({prf:ref}`rem-memory-retrieval-interpretation`) *remark* — **Operational Interpretation**

- **If the agent is surprised by reality** ($\Delta_{\text{causal}}$ high): It must increase reliance on external truth ($\Lambda_{\text{ret}} \uparrow$).
- **If the agent is not surprised** ($\Delta_{\text{causal}}$ low): It can conserve bandwidth by relying on internal memory ($\Lambda_{\text{mem}} \uparrow$), subject to the constraint that it must not overfit ($\Omega_{\text{mem}} < \Omega_{\max}$).

This closes the joint optimization problem by reducing it to a specific instantiation of the Governor's Lyapunov stability framework ({prf:ref}`def-training-lyapunov-function`).

({prf:ref}`rem-frechet-algorithm`) *remark* — **Computational Algorithm**

The minimizer can be computed via Riemannian gradient descent:

$$
q_{t+1} = \operatorname{Exp}_{q_t}\left( -\eta \sum_i \bar{w}_i \operatorname{Log}_{q_t}(q_i) \right)

$$
where:
- $\operatorname{Exp}_p: T_p\mathbb{D} \to \mathbb{D}$ is the exponential map at $p$
- $\operatorname{Log}_p: \mathbb{D} \to T_p\mathbb{D}$ is the logarithmic map (inverse of exponential)

For the Poincare disk, these have closed-form expressions via Mobius operations ({ref}`Section 21.3 <sec-bulk-boundary-independence>`).

*Complexity:* $O(k \cdot d)$ per iteration, where $k$ is the number of charts being merged and $d$ is the embedding dimension.

({prf:ref}`rem-echo-chamber-effect`) *remark* — **Echo Chamber Effect (Metric Drift)**

If agents $A$ and $B$ minimize inter-agent friction $\mathcal{F}_{AB}$ but ignore environment friction $\mathcal{F}_{AE}$, $\mathcal{F}_{BE}$, they can spiral into a shared hallucination (folie à deux).

The corrected loss function must include grounding:

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{lock}} \mathcal{F}_{AB} + \lambda_{\text{ground}} (\mathcal{F}_{AE} + \mathcal{F}_{BE})

$$

where $\mathcal{F}_{iE}$ measures the friction between agent $i$ and the environment's causal structure.

*Diagnostic:* Node 70 (BabelCheck) monitors $\partial \mathcal{F}_{AE}/\partial t$. If positive while $\mathcal{F}_{AB}$ decreases, the agents are drifting from ground truth.

({prf:ref}`rem-money-universal-metric`) *remark* — **Money as Universal Metric**

**Money** is a **Universal Metric** in the institutional sense. It quantifies the "cost distance" between any two states:

$$
d_{\text{money}}(z_1, z_2) = \inf_{\gamma: z_1 \to z_2} \int_\gamma \text{Price}(\dot{z}) \, dt

$$

This provides a normalized gauge that allows agents with disjoint utility functions to coordinate.

*Interpretation:* Money emerges as the eigenmode of the institutional metric with highest consensus (largest eigenvalue in the shared subspace).

({prf:ref}`rem-clay-millennium`) *remark* — **Relation to the Clay Millennium Problem**

The **Yang-Mills Existence and Mass Gap** problem (Clay Mathematics Institute) asks for rigorous construction of quantum Yang-Mills theory in $\mathbb{R}^4$ with mass gap $\Delta > 0$.

**What This Framework Proves:**

Theorem {prf:ref}`thm-mass-gap-dichotomy` establishes: **If Yang-Mills describes physics, then $\Delta > 0$.**

The logical structure is:

1. **The framework implements Yang-Mills:** Sections 29.14–29.18 construct Yang-Mills field equations (Theorem {prf:ref}`thm-yang-mills-equations`), the standard action (Definition {prf:ref}`def-yang-mills-action`), and the complete Standard Model Lagrangian (Definition {prf:ref}`def-complete-lagrangian`). This is Yang-Mills theory for information systems—a direct isomorphism, not an analogy.

2. **Physical theories require $\ell_L > 0$:** Any theory realizable by bounded observers with finite interface capacity must have a minimum resolution scale (the Levin Length).

3. **$\ell_L > 0$ implies $\Delta > 0$:** By Theorem {prf:ref}`thm-computational-necessity-mass-gap`, any non-trivial theory with finite Levin Length has a mass gap.

4. **Gapless theories are in the Swampland:** By Theorem {prf:ref}`thm-cft-swampland`, theories requiring $\ell_L \to 0$ (CFTs) are mathematically consistent but not physically realizable.

**Relation to the Clay Problem:**

The Clay Institute asks about Yang-Mills on continuous $\mathbb{R}^4$ satisfying Wightman or Osterwalder-Schrader axioms. The framework does not prove this directly. Instead, it proves:

- If the continuum theory describes physics, it has $\Delta > 0$ (Theorem {prf:ref}`thm-mass-gap-dichotomy`)
- If the continuum theory requires $\ell_L \to 0$, it is in the Swampland and does not describe nature

The framework thus establishes that the **physical** Yang-Mills theory (the one describing strong interactions) necessarily has a mass gap. Whether this constitutes a "solution" to the Clay problem depends on whether one accepts that physical theories must be computable.

*Physical interpretation:* Nature forbids infinite-information vacua. The mass gap is not an empirical accident but a **logical requirement** for any theory describing existing systems.

({prf:ref}`rem-goldstone-texture`) *remark* — **The Goldstone Mode (Texture)**

The symmetry breaking selects a radius $v$, but the phase $\theta$ (orientation in feature space) remains unconstrained by the potential $V(\phi)$ (which depends only on $|\phi|$). This corresponds to a **massless Goldstone boson**.

In the Fragile Agent, this massless mode is the **Texture** ($z_{\text{tex}}$). The agent remains free to rotate the definition of "noise" without energetic cost, provided the macro-separation $v$ is maintained. This recovers the **Texture Firewall** (Axiom {prf:ref}`ax-bulk-boundary-decoupling`): texture is the degree of freedom that remains gauge-invariant (unobservable to the macro-dynamics) even after symmetry breaking.

({prf:ref}`rem-why-these-values`) *remark* — **Why These Values?**

The observed physics constants $\{c \approx 3 \times 10^8 \text{ m/s}, \alpha \approx 1/137, \ldots\}$ are not arbitrary. They are the unique (modulo dimensional rescaling) solution to the Sieve constraint system that:

1. **Maximizes representational capacity** (information about the world)
2. **Minimizes thermodynamic cost** (metabolic efficiency)
3. **Maintains causal coherence** (no paradoxes)
4. **Preserves object permanence** (binding stability)
5. **Enables adaptability** (stiffness window)

Changing any constant while holding others fixed moves the system out of the feasible region. The "fine-tuning" of physical constants is the selection of the Pareto-optimal point in the Sieve constraint space.

({prf:ref}`rem-physical-interpretation`) *remark* — **Physical interpretation**

The overdamped limit corresponds to:
- **Information geometry:** The "friction" $\gamma$ represents the rate of information dissipation (forgetting). High friction means the system equilibrates quickly to the local gradient.
- **Diffusion models:** Standard score-based diffusion models operate entirely in the overdamped regime, with $\gamma \to \infty$ implicitly.
- **Neural network training:** The geodesic term $\Gamma(\dot{z},\dot{z})$ can be interpreted as a "momentum correction" that accounts for the curvature of the loss landscape. In standard gradient descent (overdamped), this term is ignored.

({prf:ref}`rem-connection-to-classification-accuracy`) *remark* — **Connection to Classification Accuracy**

The theorem provides a geometric interpretation of classification accuracy: a sample $x$ is correctly classified if and only if $\text{Enc}(x) \in \mathcal{B}_{y_{\text{true}}}$. Misclassification occurs when the encoder maps $x$ to the wrong basin—either due to encoder limitations or overlap between class distributions in observation space.

({prf:ref}`rem-a-connection-microstate-counting`) *remark* — **A.6.4a (Connection to Microstate Counting)**

The Fisher normalization used here is **not an independent input**. It is the same geometric fact established by:
- Lemma {prf:ref}`lem-a-geodesic-distance-probability-simplex`: Geodesic distance $\pi/2$ for 1 bit
- Lemma {prf:ref}`lem-a-curvature-normalization-factor-4`: Factor of 4 from curvature $K = -1$
- Proposition {prf:ref}`prop-a-area-minimal-distinguishable-cell`: Area $4\ell_L^2$ per nat

The field-theoretic derivation shows that the Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) *reproduces* the bound derived from counting—providing a consistency check between the kinematic and dynamic approaches.

({prf:ref}`rem-a-gauss-bonnet-generalization`) *remark* — **A.6.8 (Gauss-Bonnet Generalization)**

The derivation in Lemma {prf:ref}`lem-a-bulk-to-boundary-conversion` uses the **Einstein tensor divergence identity** (also called the contracted Bianchi identity):

$$
\int_{\mathcal{Z}} R \, d\mu_G = 2 \oint_{\partial\mathcal{Z}} \text{Tr}(K) \, dA_G,

$$
which is valid in **arbitrary dimension**. This is more general than the classical 2D Gauss-Bonnet theorem (which relates $\int K \, dA$ to the Euler characteristic $\chi$).

The Chern-Gauss-Bonnet theorem for even-dimensional manifolds computes topological invariants (Euler characteristic) via curvature integrals, but is not required here—we compute information capacity, not topology. The divergence theorem approach generalizes to any $D \geq 2$ without modification.

({prf:ref}`rem-a-non-circularity`) *remark* — **A.6.9 (Non-Circularity of the Derivation)**

A potential criticism of this section is circularity: *"The Metric Law encodes the holographic principle, so deriving the Area Law from the Metric Law is question-begging."*

This criticism is addressed by the **two-derivation structure** of this appendix:

1. **Microstate Counting (A.6.0):** Derives $I_{\max} = A/(4\ell_L^2)$ from:
   - Chentsov's uniqueness theorem (Theorem {prf:ref}`thm-a-chentsov-uniqueness`)
   - Fisher geodesic distance calculation (Lemma {prf:ref}`lem-a-geodesic-distance-probability-simplex`)
   - Curvature normalization $K = -1$ (Lemma {prf:ref}`lem-a-curvature-normalization-factor-4`)
   - Shannon's channel capacity (Theorem {prf:ref}`thm-a-boundary-channel-capacity`)

   **This derivation does not invoke the Metric Law.**

2. **Field-Theoretic (A.6.1–A.6.5):** Derives the same bound from the Metric Law dynamics.

The fact that both derivations yield the **same coefficient** $(1/4)$ is a non-trivial consistency check:
- The kinematic bound (from counting) constrains what is *possible*
- The dynamic equations (from the Metric Law) describe what *happens*
- Their agreement shows the Metric Law is *compatible* with holographic constraints—not that it *assumes* them

**Analogy to physics:** In black hole thermodynamics, Hawking derived $S = A/(4\ell_P^2)$ thermodynamically (1975), and Strominger-Vafa derived it microscopically (1996). Neither derivation is circular; their agreement is a profound consistency check on string theory.

Similarly, the microstate counting here is analogous to Strominger-Vafa, while the field-theoretic derivation is analogous to Hawking. The framework admits both perspectives.

## Algorithms

({prf:ref}`alg-lazarus`) *algorithm* — **Lazarus Reallocation**

**Input:** Dead code $k_{\text{dead}}$ with $P(K = k_{\text{dead}}) < \epsilon_{\text{dead}}$.

**Procedure:**
1. Find the most stressed symbol:

   $$
   k_{\text{stressed}} := \arg\max_k \sigma_k^2

   $$
2. Perform Symbol Fission on $k_{\text{stressed}}$, reusing index $k_{\text{dead}}$:
   - Compute split direction $v_1$ from $\Sigma_{k_{\text{stressed}}}$.
   - Set $e_{k_{\text{dead}}} := e_{k_{\text{stressed}}} + \epsilon v_1$.
   - Update $e_{k_{\text{stressed}}} := e_{k_{\text{stressed}}} - \epsilon v_1$.
3. Update Voronoi cells: The new code inherits half of $k_{\text{stressed}}$'s cell.

**Effect:** Vocabulary migrates to high-information-density regions. Dead codes are "resurrected" where they are needed.

*Connection to existing constraints:* This implements the anti-collapse regularizer from {ref}`Section 3.5.5 <sec-calibrating-tolerances>`: $\lambda_{\text{use}} D_{\mathrm{KL}}(\hat{p}(K) \| \text{Unif}(\mathcal{K}))$.
