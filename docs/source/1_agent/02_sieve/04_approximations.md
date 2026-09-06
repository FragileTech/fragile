(sec-infeasible-implementation-replacements)=
# Infeasible Implementation Replacements

## TLDR

- Many “ideal” stability/geometry criteria are **computationally intractable**; this chapter provides cheap surrogates
  that preserve the same failure-detection intent.
- Each replacement is a **mapping**: (theoretical barrier/diagnostic) → (practical probe/loss) with an operational
  interpretation and a PyTorch-friendly form.
- Use these surrogates to keep the Sieve runnable in real systems: online checks (✓) plus amortized/offline probes (⚡/✗).
- The goal is *not* to weaken the theory; it is to make the same contracts **implementable** without pretending you can
  compute impossible objects.
- Treat the summary table as an engineering index: it tells you what to compute when a given theoretical requirement is
  out of reach.

## Roadmap

1. Replacement patterns (what gets approximated and why).
2. A catalog of concrete substitutions with code-ready formulas.
3. A summary mapping from theory labels to implementation losses/probes.

(rb-practical-substitutions)=
:::{admonition} Researcher Bridge: Practical Substitutions for Idealized Laws
:class: tip
Many theoretical constraints are too expensive to compute directly. This section provides the RL-engineering replacements (surrogate losses, probes, and bounds) that preserve the same failure detection in practice.
:::

:::{div} feynman-prose
Here is a situation that comes up constantly in physics and engineering: you derive a beautiful, exact criterion for detecting when something goes wrong, then realize that computing it would take longer than the age of the universe. What do you do?

You find a cheaper test that measures a useful consequence of the failure. This is not cheating---it is the essence of good engineering. To know if my car engine is overheating, I do not need the temperature of every molecule. A single thermometer in the coolant can warn me, but it is not a proof about every component.

The theoretical framework gives us exact criteria for instability, bifurcations, and non-tame dynamics. These are mathematically elegant but computationally ruinous. So we ask: what simpler measurement triggers an alarm at the same moments? What is the "thermometer" for each kind of failure?

For each expensive theoretical test, we use a cheap surrogate with a stated scope. It may miss failures or raise false alarms, so the result is a screening signal until coverage and regularity assumptions justify more.
:::

Several regularization terms from the theoretical framework are computationally infeasible for standard training. This section provides practical alternatives with full PyTorch implementations.

(sec-barrierbode-temporal-gain-margin)=
## BarrierFreq → Temporal Gain Margin

:::{div} feynman-prose
Under the stability, relative-degree, and LTI hypotheses of the Bode sensitivity theorem, you cannot suppress disturbances at all frequencies simultaneously. Push down the response at one frequency, it pops up somewhere else. The integral of log-sensitivity is then fixed by the open-loop pole data---like a conservation relation for that specified control model.

Why care about this for neural policies? It gives a reference picture for error amplification. If a neural controller oscillates wildly, the Bode integral does not automatically apply; direct error-growth measurements are the appropriate surrogate.

The catch: computing that integral requires the transfer function $S(j\omega)$, which assumes a linear time-invariant system. Neural networks are neither. Even with FFT approximations, we would need long, stationary trajectories---but our agents are constantly exploring and changing.

What are we really trying to detect? Errors getting bigger over time. Oscillations that grow instead of decay. We do not need Fourier analysis for that---we can watch the error magnitudes directly.
:::

**Original (Infeasible):**

$$
\int_{0}^{\infty} \log \lvert S(j\omega) \rvert d\omega = \text{const.} \quad \text{(Bode sensitivity integral)}

$$
**Problem:** Requires frequency-domain analysis of the closed-loop transfer function $S(j\omega)$. Neural policies don't have closed-form transfer functions, and FFT requires long stationary trajectories.

**Replacement: Temporal Gain Margin**

$$
\mathcal{L}_{\text{gain}} = \frac{1}{\min(K,T-1)}\sum_{k=1}^{\min(K,T-1)} \max\left(0, \frac{\Vert e_{t+k} \Vert}{\Vert e_t \Vert + \epsilon} - G_{\max}\right)^2

$$

:::{div} feynman-prose
The idea is simple: compare the error at time $t$ to the error at time $t+k$. If the ratio exceeds $G_{\max}$, errors are amplifying---bad news. Summing over horizons $k = 1, 2, \ldots, K$ catches both fast oscillations (small $k$) and slower instabilities (larger $k$).

The squared penalty means small violations get a tap, big violations get hammered. Differentiable everywhere, and it focuses the optimizer on the worst cases.

The default $G_{\max} = 2$ tolerates occasional error doubling (transient disturbances happen), but flags anything worse.
:::

This surrogate loss penalizes error amplification and oscillatory instability without requiring LTI assumptions.

```python
def compute_gain_margin_loss(
    errors: torch.Tensor,  # Shape: [B, T] - tracking errors over time
    G_max: float = 2.0,     # Maximum allowed gain
    K: int = 5,             # Lookahead horizon
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    BarrierFreq replacement: Temporal gain margin constraint.

    Penalizes trajectories where errors amplify over time,
    corresponding to the loop instability detected by Bode sensitivity analysis.

    Args:
        errors: [B, T] tensor of error magnitudes at each timestep
        G_max: Maximum allowed amplification ratio
        K: Number of steps to check ahead
        eps: Numerical stability

    Returns:
        Scalar loss penalizing gain violations
    """
    B, T = errors.shape
    if T < 2:
        return torch.tensor(0.0, device=errors.device)

    total_violation = 0.0
    for k in range(1, min(K + 1, T)):
        # Gain at lag k: ||e_{t+k}|| / ||e_t||
        e_t = errors[:, :-k]  # [B, T-k]
        e_t_plus_k = errors[:, k:]  # [B, T-k]

        gain = e_t_plus_k / (e_t + eps)
        violation = torch.relu(gain - G_max).pow(2)
        total_violation = total_violation + violation.mean()

    return total_violation / min(K, T - 1)


# Alternative: Peak gain detection
def compute_peak_gain_loss(
    errors: torch.Tensor,  # [B, T]
    G_max: float = 2.0,
) -> torch.Tensor:
    """Simplified variant: penalizes maximum gain ratio."""
    B, T = errors.shape
    e_ratios = errors[:, 1:] / (errors[:, :-1] + 1e-6)
    max_gain = e_ratios.max(dim=-1).values  # [B]
    return torch.relu(max_gain - G_max).pow(2).mean()
```

(sec-bifurcatecheck-stochastic-jacobian-probing)=
## BifurcateCheck → Stochastic Jacobian Probing

:::{div} feynman-prose
At a bifurcation, the qualitative behavior of a system changes suddenly---a stable fixed point becomes unstable, splits in two, or starts oscillating. Like water turning to ice: cross a threshold and everything is different.

For a world model $S_t$ predicting latent state evolution, bifurcations spell danger. The model sits on a knife-edge between behaviors, and small input changes send predictions wildly off course.

The signature of bifurcation is in the Jacobian $J_{S_t} = \partial S_t(z) / \partial z$. When an eigenvalue crosses the unit circle (discrete time) or imaginary axis (continuous time), you have a bifurcation.

But the full Jacobian costs $O(Z^2)$ to form and $O(Z^3)$ for eigenvalues. For latent dimension 256, that is millions of operations per sample, every training step, every batch element. Not practical.
:::

**Original (Infeasible):**

$$
\rho(J_{S_t}) \quad \text{where } J_{S_t} = \frac{\partial S_t(z)}{\partial z}

$$
**Problem:** Computing the full Jacobian is $O(Z^3)$. For $Z = 256$, this is ~16M operations per sample.

**Replacement: Random Jacobian gain-spread probing**

$$
\mathcal{L}_{\text{bifurcate}} = \operatorname{Var}_u\left[\Vert J_{S_t}^{\mathsf T} u \Vert^2\right] \quad \text{where } u \sim \mathcal{N}(0, I)

$$

:::{div} feynman-prose
The trick is to obtain a cheap gain-spread signal rather than the full spectrum. If the sampled output directions are stretched very differently, the Jacobian may have anisotropic sensitivity. Similar samples do not prove that all eigenvalues are clustered, and gain spread does not by itself locate a bifurcation.

Instead of computing the full Jacobian, probe it with random output directions. Pick $u$, compute $J^{\mathsf T}u$ (a vector--Jacobian product, cheap with autodiff), measure its norm, and repeat a few times. This samples local sensitivity in selected directions.

If the sampled gains vary wildly, the result is a warning about local anisotropy. If they are similar, the probe has simply found no evidence in those directions; it is not a spectral-radius or bifurcation certificate.

This is inspired by randomized matrix probing, but the displayed variance is not the Hutchinson estimator of a trace and does not recover the full spectrum without additional assumptions.
:::

High variance in the probed vector-Jacobian-product norm is a stochastic gain-spread
proxy. It can flag local sensitivity, but it is not by itself a spectral-radius or
bifurcation certificate.

```python
def compute_bifurcation_loss(
    world_model: nn.Module,
    z: torch.Tensor,           # [B, Z] - current latent states
    a: torch.Tensor,           # [B, A] - actions (if needed)
    n_probes: int = 5,
    instability_threshold: float = 1.0,
) -> torch.Tensor:
    """
    BifurcateCheck replacement: Stochastic Jacobian probing.

    Probes the Jacobian with random directions. High variance in
    ||J^T v|| is a local gain-spread signal; it is a screening
    statistic rather than a spectral-radius certificate.

    Args:
        world_model: S_t(z, a) -> z_next
        z: Current latent states [B, Z]
        a: Actions [B, A]
        n_probes: Number of random direction probes
        instability_threshold: Variance threshold for penalty

    Returns:
        Scalar loss penalizing high Jacobian variance
    """
    B, Z = z.shape
    z = z.requires_grad_(True)

    # Forward through world model
    z_next = world_model(z, a)  # [B, Z]

    vjp_norms = []
    for _ in range(n_probes):
        # Random probe direction
        v = torch.randn_like(z)  # [B, Z]

        # Vector-Jacobian product (VJP) via autodiff (efficient: O(Z))
        vjp = torch.autograd.grad(
            outputs=z_next,
            inputs=z,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,
        )[0]  # [B, Z]

        vjp_norm = vjp.norm(dim=-1)  # [B]
        vjp_norms.append(vjp_norm)

    # Stack and compute variance across probes
    vjp_norms = torch.stack(vjp_norms, dim=0)  # [n_probes, B]
    variance = vjp_norms.var(dim=0).mean()  # Average variance across batch

    # Penalize high variance (indicates instability)
    loss = torch.relu(variance - instability_threshold).pow(2)

    return loss
```

(sec-tamecheck-lipschitz-gradient-proxy)=
## TameCheck → Lipschitz Gradient Proxy

:::{div} feynman-prose
What does "tame" mean for a function? No sharp corners or sudden kinks. The output changes smoothly through input space---not just in value, but in slope.

The Hessian (second derivatives) captures this smoothness. Bounded Hessian norm means the gradient cannot change too fast---the function is tame. This matters for optimization: gradient descent on non-tame functions oscillates wildly or gets stuck in pathological regions.

But the Hessian is even more expensive than the Jacobian: $O(Z^2 \times P)$ for a world model with $P$ parameters on $Z$-dimensional latent space. Completely impractical.
:::

**Original (Infeasible):**

$$
\Vert \nabla^2 S_t \Vert \quad \text{(Hessian norm)}

$$
**Problem:** Full Hessian is $O(Z^2 \times P_{WM})$ — prohibitive for large world models.

**Replacement: Sampled Lipschitz-of-Gradient Probe**

$$
\mathcal{L}_{\text{tame}} = \frac{\Vert \nabla_z S_t(z_1) - \nabla_z S_t(z_2) \Vert}{\Vert z_1 - z_2 \Vert + \epsilon}

$$

:::{div} feynman-prose
Key insight: bounded Hessian means the gradient does not change too fast as you move through space. That is exactly a Lipschitz condition on the gradient.

So sample a local Lipschitz-of-gradient ratio. Take two nearby points $z_1$ and $z_2$, compute the chosen directional gradients, and measure how much they changed relative to the input separation. A large ratio is evidence of non-tameness; a small finite collection of ratios is not a global Hessian bound without coverage and regularity assumptions.

Computing a directional $\nabla_z S_t(z)$ at one point is cheap (one backward pass). We need two such gradients plus a perturbation. The operation is cheaper than forming the full Hessian, while its coverage and output-direction choices determine what it can detect.

General pattern: when you cannot afford the whole matrix, probe its action on carefully chosen vectors.
:::

The ratio is a directional, finite-difference estimate. A bounded collection of
such probes is evidence for local Hessian control; it is not a global bound without
coverage and regularity assumptions.

```python
def compute_tame_loss(
    world_model: nn.Module,
    z: torch.Tensor,         # [B, Z]
    a: torch.Tensor,         # [B, A]
    perturbation_scale: float = 0.01,
    lipschitz_target: float = 1.0,
) -> torch.Tensor:
    """
    TameCheck replacement: Lipschitz gradient constraint.

    Instead of computing full Hessian, we estimate a local directional
    Lipschitz ratio for the gradient via finite differences. A finite set
    of probes is evidence about local tameness; it is not a global Hessian
    spectral-norm bound without coverage and regularity assumptions.

    Args:
        world_model: S_t(z, a) -> z_next
        z: Current latent states [B, Z]
        a: Actions [B, A]
        perturbation_scale: Size of random perturbation
        lipschitz_target: Target Lipschitz constant

    Returns:
        Scalar loss penalizing non-tame dynamics
    """
    B, Z = z.shape

    # Two nearby points
    z1 = z.requires_grad_(True)
    delta = torch.randn_like(z) * perturbation_scale
    z2 = (z + delta).requires_grad_(True)

    # Forward passes
    z1_next = world_model(z1, a)
    z2_next = world_model(z2, a)

    # Probe a random output direction so that each gradient is a
    # directional Jacobian transpose, rather than the gradient of an
    # arbitrary sum of output coordinates.
    output_probe = torch.randn_like(z1_next)
    grad1 = torch.autograd.grad(
        (z1_next * output_probe).sum(), z1, create_graph=True, retain_graph=True
    )[0]  # [B, Z]

    grad2 = torch.autograd.grad(
        (z2_next * output_probe).sum(), z2, create_graph=True, retain_graph=True
    )[0]  # [B, Z]

    # Lipschitz estimate: ||grad1 - grad2|| / ||z1 - z2||
    grad_diff = (grad1 - grad2).norm(dim=-1)  # [B]
    z_diff = delta.norm(dim=-1) + 1e-6  # [B]

    lipschitz_estimate = grad_diff / z_diff  # [B]

    # Penalize exceeding target Lipschitz constant
    loss = torch.relu(lipschitz_estimate - lipschitz_target).pow(2).mean()

    return loss
```

(sec-topocheck-value-gradient-alignment)=
## TopoCheck → Value Gradient Alignment

:::{div} feynman-prose
Reachability is fundamental: can I get from here to there? From state $z$ to $z_{\text{goal}}$, does a path exist?

The theoretical answer requires planning: simulate all trajectories, find which reach the goal. For horizon $H$, batch size $B$, and latent dimension $Z$, this costs $O(H \times B \times Z)$---and $H$ might need to be huge to guarantee finding a path.

Why care about reachability? We need the value function to tell the truth. If $V(z)$ says "this state is valuable," there had better be an actual path to high-reward regions. If the latent space has holes or barriers, the value function might look smooth and encouraging while the goal is actually unreachable.
:::

**Original (Infeasible):**

$$
T_{\text{reach}}(z_{\text{goal}}) \quad \text{(Reachability time)}

$$
**Problem:** Requires multi-step planning through world model: $O(H \times B \times Z)$ with potentially large horizon $H$.

**Replacement: Value Gradient Alignment**

$$
\mathcal{L}_{\text{topo}} = \mathrm{ReLU}\left(\left\langle \nabla_z V(z), \frac{z_{\text{goal}} - z}{\| z_{\text{goal}} - z \|} \right\rangle\right)

$$

:::{div} feynman-prose
A much cheaper test is a first-order value-direction check in the displayed flat-coordinate approximation. If $V$ is a cost minimized at the goal, gradient descent follows $-\nabla_z V(z)$. In a star-shaped, locally admissible setting, one can compare this direction with the goal chord; in a curved or obstructed space, the relevant direction is the admissible geodesic or closed-loop update.

The inner product $\langle \nabla_z V, \hat{d}_{\text{goal}} \rangle$ measures alignment with that local chord: positive means the gradient points toward the goal (wrong for a cost), negative means it points away (correct). We penalize positive alignment as a local consistency heuristic and leave negative alignment alone.

This is neither necessary nor sufficient for global reachability without a restriction such as a star-shaped reachable set and compatible dynamics. Obstacles, nonholonomic controls, and other basins can make the chord misleading. The probe catches a local sign mismatch, not the existence of a path.

Cost: one gradient computation. No multi-step simulation.
:::

The loss tests first-order local alignment only. Even perfect alignment does not
establish global reachability in the presence of obstacles, nonholonomic dynamics,
or disconnected basins.

```python
def compute_topo_loss(
    critic: nn.Module,
    states: torch.Tensor,      # [B, Z] - current states
    goal_states: torch.Tensor,  # [B, Z] or [Z] - goal states
) -> torch.Tensor:
    """
    TopoCheck replacement: Value gradient alignment.

    Instead of computing multi-step reachability, we check if
    the critic's value gradient points toward the goal. This is
    a first-order local consistency check for gradient-based motion;
    it is not a reachability guarantee.

    Args:
        critic: V(z) -> scalar value
        states: Current states [B, Z]
        goal_states: Target states [B, Z] or [Z]

    Returns:
        Scalar loss (negative = gradient points toward goal)
    """
    B, Z = states.shape
    states = states.requires_grad_(True)

    # Compute value and its gradient
    values = critic(states)  # [B]
    grad_v = torch.autograd.grad(
        values.sum(), states, create_graph=True
    )[0]  # [B, Z]

    # Direction to goal
    if goal_states.dim() == 1:
        goal_states = goal_states.unsqueeze(0).expand(B, -1)

    to_goal = goal_states - states  # [B, Z]
    to_goal_normalized = to_goal / (to_goal.norm(dim=-1, keepdim=True) + 1e-6)

    # Alignment: should be negative (V decreases toward goal)
    alignment = (grad_v * to_goal_normalized).sum(dim=-1)  # [B]

    # Loss: penalize positive alignment (wrong direction)
    loss = torch.relu(alignment).mean()

    return loss
```

(sec-geomcheck-efficient-infonce)=
## GeomCheck → Efficient InfoNCE

:::{div} feynman-prose
Contrastive learning: related points should be close in latent space, unrelated points far apart. For temporal data, "related" means close in time---frame $t$ and frame $t+k$ should map to nearby codes.

InfoNCE is the standard loss. The probability of correctly identifying which $z_{t+k}$ goes with $z_t$, among all batch elements, should be high. Mathematically, a softmax over all pairwise similarities.

The problem: "all pairwise." Batch size $B$ means $B^2$ similarity computations. For $B = 1024$ and $Z = 256$, that is a quarter billion multiplications per batch.
:::

**Original (Expensive):**

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\operatorname{sim}(z_t, z_{t+k})/\tau)}{\exp(\operatorname{sim}(z_t, z_{t+k})/\tau)+\sum_{j\in\mathcal N_t} \exp(\operatorname{sim}(z_t, z_j^-)/\tau)}

$$
**Problem:** Full pairwise computation is $O(B^2 \times Z)$.

**Replacement: Sampled InfoNCE**

$$
\mathcal{L}_{\text{InfoNCE}}^{\text{eff}} = -\log \frac{\exp(\operatorname{sim}(z_t, z_{t+k})/\tau)}{\exp(\operatorname{sim}(z_t, z_{t+k})/\tau) + \sum_{j=1}^{K} \exp(\operatorname{sim}(z_t, z_{\text{neg},j})/\tau)}

$$

:::{div} feynman-prose
We do not need every sample as a negative for a cheap training step, but the number matters for any information statement. With $K$ negatives, the usual contrastive lower-bound scale is capped by $\log(K+1)$; choose $K$ or a memory bank large enough for the capacity budget being checked. A good result with 128 negatives cannot certify the same quantity as one with 1024.

Sample $K$ negatives instead of all $B$. Cost drops from $O(B^2)$ to $O(KB)$. With $K = 128$ and $B = 1024$, that is 8x faster.

Where do negatives come from? Two options: other batch samples (in-batch negatives), or a memory bank of codes from previous batches. The memory bank decouples negative count from batch size---small batches, many negatives.

The projection head is optional but helps. Empirically, contrastive learning works better projecting to a different space before computing similarities. Use the original $z$ downstream; the projected versions are just for the loss.
:::

Use $K \ll B$ sampled negatives instead of full batch.

```python
class EfficientInfoNCE(nn.Module):
    """
    GeomCheck replacement: Efficient contrastive loss.

    Uses K sampled negatives instead of full batch pairwise.
    Reduces O(B²Z) to O(KBZ) where K << B.
    """

    def __init__(
        self,
        latent_dim: int,
        n_negatives: int = 128,
        tau: float = 0.1,  # softmax scale
    ):
        super().__init__()
        self.n_negatives = n_negatives
        self.tau = tau

        # Projection head (optional, improves quality)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        z_anchor: torch.Tensor,   # [B, Z] - z_t
        z_positive: torch.Tensor,  # [B, Z] - z_{t+k} (temporally close)
        z_bank: torch.Tensor = None,  # [M, Z] - memory bank for negatives
    ) -> torch.Tensor:
        """
        Compute efficient InfoNCE loss.

        Args:
            z_anchor: Anchor embeddings [B, Z]
            z_positive: Positive pairs [B, Z]
            z_bank: Optional memory bank for negatives [M, Z]

        Returns:
            Scalar contrastive loss
        """
        B, Z = z_anchor.shape

        # Project
        anchor = F.normalize(self.projector(z_anchor), dim=-1)  # [B, Z]
        positive = F.normalize(self.projector(z_positive), dim=-1)  # [B, Z]

        # Sample negatives
        if z_bank is not None and z_bank.shape[0] >= self.n_negatives:
            # Sample from memory bank
            indices = torch.randperm(z_bank.shape[0])[:self.n_negatives]
            negatives = z_bank[indices]  # [K, Z]
            negatives = F.normalize(self.projector(negatives), dim=-1)
        else:
            # Use other batch elements as negatives (in-batch), with
            # row-wise indices that exclude the anchor itself.
            K = min(self.n_negatives, B - 1)
            if K == 0:
                raise ValueError("in-batch InfoNCE needs at least two anchors")
            row = torch.arange(B, device=z_anchor.device).unsqueeze(1)
            offsets = torch.randint(1, B, (B, K), device=z_anchor.device)
            negative_indices = (row + offsets) % B
            negatives = anchor[negative_indices]  # [B, K, Z]

        # Positive similarity: [B]
        pos_sim = (anchor * positive).sum(dim=-1) / self.tau

        # Negative similarities: [B, K]
        if negatives.dim() == 2:
            neg_sim = torch.mm(anchor, negatives.T) / self.tau
        else:
            neg_sim = (anchor.unsqueeze(1) * negatives).sum(dim=-1) / self.tau

        # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = pos - log(exp(pos) + sum(exp(neg)))
        # = pos - logsumexp([pos, neg1, neg2, ...])

        # Combine for logsumexp: [B, K+1]
        all_sim = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)

        # Loss: -pos + logsumexp(all)
        loss = -pos_sim + torch.logsumexp(all_sim, dim=-1)

        return loss.mean()


# Usage example:
def compute_geom_loss(
    vae_encoder: nn.Module,
    x_t: torch.Tensor,      # [B, D] - observation at time t
    x_t_plus_k: torch.Tensor,  # [B, D] - observation at time t+k
    info_nce: EfficientInfoNCE,
) -> torch.Tensor:
    """GeomCheck: Contrastive anchoring for latent space."""
    z_t = vae_encoder(x_t)
    z_t_k = vae_encoder(x_t_plus_k)
    return info_nce(z_t, z_t_k)
```

(sec-summary-replacement-mapping)=
## Summary: Replacement Mapping

:::{div} feynman-prose
Five expensive theoretical tests, five cheaper probes for related failure signals. The speedups can be substantial, but the replacements do not detect exactly the same events and must be interpreted with their stated hypotheses.

The key insight: you do not need to compute everything for every update, but you must know what the sample can support. Random probes estimate gain spread, negatives set the contrastive information scale, and gradient samples provide local evidence about smoothness. Full computation gives stronger information; the cheaper signal is a screening tool.

This is a deep principle. In physics: effective theories are useful within their scale and error estimates. In computer science: approximation algorithms are useful when their error is controlled. Here, surrogate losses are cheap tests whose measured scope must be kept separate from the exact criteria they approximate.
:::

| Original                  | Replacement        | Speedup  | Preserved Property              |
|---------------------------|--------------------|----------|---------------------------------|
| BarrierFreq (FFT)         | Temporal Gain      | trajectory-dependent | Screens for error amplification |
| BifurcateCheck ($O(Z^3)$) | Jacobian Probing   | ~$Z^2/K$ | Probes local gain spread       |
| TameCheck ($O(Z^2 P)$)    | Lipschitz Gradient | ~$ZP$    | Samples directional Hessian variation |
| TopoCheck ($O(HBZ)$)      | Value Alignment    | one gradient | Tests first-order local consistency |
| GeomCheck ($O(B^2 Z)$)    | Sampled NCE        | ~$B/K$   | Preserves slow features         |

:::{note}
:class: feynman-added
A word of caution: these surrogates are not mathematically equivalent to the originals. They are designed to trigger on the same failure modes, but there may be edge cases where one catches something the other misses. In practice, this is rarely a problem---the surrogates are often more robust because they are less sensitive to numerical issues that plague the exact computations.
:::
