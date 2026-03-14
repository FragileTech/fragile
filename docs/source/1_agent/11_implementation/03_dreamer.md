(sec-geometric-dreamer)=
# Geometric Dreamer --- Model-Based RL on the Poincare Ball

## TLDR

- **Geometric Dreamer** is a model-based RL algorithm that trains an actor-critic on
  *imagined* rollouts inside the Poincare ball, using the encoder
  ({ref}`sec-shared-dynamics-encoder`) for perception and the covariant world model
  ({ref}`sec-why-physics-for-world-model`) for imagination.
- A **reward branch** inside `CovariantPotentialNet` predicts environment rewards from
  latent states via action-conditioned attention, enabling the world model to simulate
  complete experience trajectories without touching the real environment.
- A **geometric actor** (`GeometricActor`) generates Gaussian-distributed actions conditioned
  on the latent state and chart routing weights, with entropy regularization from the MaxEnt
  objective ({prf:ref}`def-maxent-rl-objective-on-macrostates`).
- The existing **critic** $V_{\text{critic}}$ inside `CovariantPotentialNet` doubles as the
  value function, trained with $\lambda$-returns from imagined trajectories. The effective
  potential $\Phi_{\text{eff}}$ unifies exploration drive, reward prediction, and risk
  avoidance into a single scalar landscape.
- A **four-phase training pipeline** extends the existing three phases: Phase 1 (encoder),
  Phase 2 (world model), Phase 3 (joint), Phase 4 (RL). Phase 4 alternates between
  real-environment data collection and imagination-based actor-critic updates.

## Roadmap

1. Why geometric model-based RL? The motivation from Dreamer and curved latent spaces.
2. Architecture overview: the perception-imagination-action loop.
3. The reward branch: predicting rewards in curved space.
4. The geometric actor: policy on the Poincare ball.
5. The critic as effective potential: $V_{\text{critic}} = V^{\pi}$.
6. Imagination training: actor-critic updates from world model rollouts.
7. Integration with DM Control: from observations to actions.
8. The four-phase training pipeline.
9. Diagnostics: knowing your agent is learning.
10. Theory-to-code correspondence.


(sec-why-geometric-mbrl)=
## 1. Why Geometric Model-Based RL

:::{div} feynman-prose
Let me tell you what we have built so far, and what is missing.

We have an encoder that maps observations into the Poincare ball, assigning discrete chart
labels and continuous coordinates. We have a world model that integrates Hamiltonian dynamics
on this curved space using Boris-BAOAB, predicting how latent states evolve under actions.
Both are trained in a supervised pipeline: give them data, they learn to compress and predict.

But neither of them can *act*. The encoder perceives, the world model imagines, but nobody
decides what to do. The agent has no policy, no value function trained on rewards, no way to
improve its behavior from experience. It is a brain without a will.

Here is the key insight. The world model already contains the machinery for decision-making.
The effective potential $\Phi_{\text{eff}}$ ({prf:ref}`def-effective-potential`) is a scalar
landscape over the latent space. The BAOAB integrator moves particles down this landscape.
The control field $u_\pi$ steers the dynamics with action-conditioned forces. All we need to
do is *connect these to rewards*.

The algorithm I am about to describe does exactly that. It is Dreamer --- Hafner et al.'s
idea of training actor-critic networks on imagined trajectories --- but on a curved space
with symplectic integration, chart routing, and Hodge-decomposed rewards. The geometry is
not decoration. It gives us CFL-stable imagination, norm-preserving curl forces for
non-conservative rewards, and a natural exploration drive from the hyperbolic metric.
:::

:::{admonition} Researcher Bridge: Dreamer-v3 and MBPO
:class: info
Dreamer-v3 (Hafner et al., 2023) learns a recurrent state-space model (RSSM) in Euclidean
latent space and trains actor-critic networks purely from imagined rollouts. MBPO (Janner
et al., 2019) uses a learned model to generate synthetic data for model-free updates.
Geometric Dreamer differs in three ways: (1) the latent space is the Poincare ball with
discrete chart structure, giving hierarchical representations; (2) imagination uses
symplectic Boris-BAOAB integration with CFL stability bounds, preventing trajectory
divergence; (3) the Hodge decomposition ({prf:ref}`thm-hodge-decomposition`) separates
conservative from solenoidal reward components, enabling the value curl to handle
non-conservative reward landscapes that break standard scalar critics.
:::


(sec-dreamer-architecture-overview)=
## 2. Architecture Overview

The full perception-imagination-action loop has four stages per training iteration:

```{mermaid}
flowchart LR
    subgraph Perception["Perception (Encoder)"]
        OBS["obs [B, D_obs]"] --> FE["Feature Extractor"]
        FE --> ENC["SharedDynAtlasEncoder"]
        ENC --> Z["z_geo [B, D]"]
        ENC --> RW["rw [B, K]"]
    end

    subgraph Imagination["Imagination (World Model)"]
        Z --> IM["imagine()"]
        RW --> IM
        ACT --> IM
        IM --> ZT["z_traj [B, H, D]"]
        IM --> RT["r_hat [B, H]"]
        IM --> PHI["phi_eff [B, H, 1]"]
    end

    subgraph Action["Action (Actor)"]
        Z --> A["GeometricActor"]
        RW --> A
        A --> ACT["a ~ pi(.|z, rw)"]
    end

    subgraph Value["Value (Critic)"]
        ZT --> V["V_critic in Phi_eff"]
        RT --> LAM["lambda-returns"]
        V --> LAM
        LAM --> LOSS["Actor-Critic Loss"]
    end

    ACT --> ENV["dm_control env"]
    ENV --> OBS
```

### Module Summary

:::{div} feynman-added
| Module | Location | Input | Output | Status |
|--------|----------|-------|--------|--------|
| `SharedDynAtlasEncoder` | `atlas.py` | obs features `[B, D_feat]` | `z_geo [B, D]`, `rw [B, K]` | Existing |
| `GeometricWorldModel` | `covariant_world_model.py` | `z_0, actions, rw` | `z_traj [B, H, D]`, `phi_eff [B, H, 1]` | Existing |
| `CovariantPotentialNet` | `covariant_world_model.py` | `z, rw` | force `[B, D]`, $\Phi_{\text{eff}}$ `[B, 1]` | Existing |
| `CovariantPotentialNet` (reward branch) | `covariant_world_model.py` | `z, a, rw` | $\hat{r}$ `[B, 1]` | Extend existing |
| `GeometricActor` (chart-conditioned policy) | **proposed** | `z, rw` | $\mu$ `[B, A]`, $\sigma$ `[B, A]` | New (reuses attention pattern) |
| `_SharedFeatureExtractor` | `atlas.py` | `obs [B, D_obs]` | `features [B, hidden_dim]` | Existing |
| `SequenceReplayBuffer` | **proposed** | `(obs, a, r, obs', done)` | mini-batches | New |
:::


(sec-reward-head)=
## 3. The Reward Branch --- Predicting Rewards in Curved Space

:::{div} feynman-prose
The world model predicts *where* the agent will be, but not *how much reward* it will
receive there. For imagination training, we need to predict rewards from latent states.

You might think: just add an MLP on top of $z$. But rewards in continuous control are
chart-dependent. A humanoid standing upright in chart 3 earns reward; the same latent
magnitude in chart 7 (where the agent is falling) does not. The reward prediction must
be chart-conditioned, just like every other module in the world model.

The good news is that we already have exactly the right machinery. `CovariantPotentialNet`
already uses `ChartTokenizer` and `CovariantAttention` to produce scalar outputs
($V_{\text{critic}}$, $\Psi_{\text{risk}}$). Adding a reward branch is a matter of
one more attention head and one more output projection --- the tokenizers, embeddings,
and `GeodesicConfig` are shared. This gives us a reward prediction that respects the
atlas structure with minimal new code.
:::

Rather than building a separate module from scratch, the reward head extends the
existing `CovariantPotentialNet` ({ref}`sec-control-field-impl`) with an
action-conditioned branch. The `ChartTokenizer`, `z_embed`, and `GeodesicConfig`
are shared infrastructure; only a new attention head and output projection are added:

```python
# PROPOSED: extend CovariantPotentialNet.__init__ with a reward branch
# (inside covariant_world_model.py, after the existing risk head)
#
# API change: add action_dim as constructor parameter.
# Update GeometricWorldModel.__init__ (line 621) to pass action_dim:
#   self.potential_net = CovariantPotentialNet(
#       latent_dim, num_charts, d_model, action_dim, ...)

# Reward head: action-conditioned attention → scalar r_hat
self.action_tok = ActionTokenizer(action_dim, d_model, latent_dim)   # reuse existing class
self.reward_attn = CovariantAttention(geo_cfg)                       # same GeodesicConfig
self.reward_out  = SpectralLinear(d_model, 1)
```

The forward pass mirrors `CovariantControlField.forward()` --- it concatenates
action tokens and chart tokens, then attends from the position embedding:

```python
# PROPOSED: CovariantPotentialNet.predict_reward()
def predict_reward(self, z, action, rw):
    """Predict instantaneous reward from (z, action, rw).

    Returns:
        r_hat: [B, 1] predicted reward.
    """
    act_x, act_z = self.action_tok(action, z)            # [B, A, d_model], [B, A, D]
    chart_x, chart_z = self.chart_tok(rw, z)              # [B, K, d_model], [B, K, D]

    ctx_x = torch.cat([act_x, chart_x], dim=1)           # [B, A+K, d_model]
    ctx_z = torch.cat([act_z, chart_z], dim=1)            # [B, A+K, D]
    x_q = self.z_embed(z)

    feat, _ = self.reward_attn(z, ctx_z, x_q, ctx_x, ctx_x)
    return self.reward_out(feat)                           # [B, 1]
```

This reuses the exact same `ChartTokenizer` and `ActionTokenizer` that the control
field already employs, and the same `GeodesicConfig` that governs all attention heads
in the world model. No new tokenizer classes or embedding layers are needed.

The reward prediction loss is simply the squared error against actual environment rewards:

$$
\mathcal{L}_{\text{reward}} = \frac{1}{BH} \sum_{b,t} \bigl(\hat{r}(z_{b,t}, a_{b,t}, \text{rw}_{b,t}) - r_{b,t}\bigr)^2
$$

:::{div} feynman-prose
Why not predict reward from the effective potential $\Phi_{\text{eff}}$ directly? Because
$\Phi_{\text{eff}}$ is a *value function* --- it integrates future rewards, not instantaneous
ones. The reward branch predicts the immediate step reward $r_t$, which is then used to
construct $\lambda$-return targets for training $V_{\text{critic}}$ (inside the same
`potential_net`) as a value function. The two branches are complementary: the reward
branch provides the building blocks, and the critic branch assembles them into
long-horizon predictions. Crucially, both share the same `ChartTokenizer` and positional
embeddings, so improvements in chart conditioning benefit both simultaneously.
:::


(sec-geometric-actor)=
## 4. The Geometric Actor --- Policy on the Poincare Ball

:::{div} feynman-prose
Now we need a policy. In standard RL, the actor maps observations to action distributions.
Here, the actor maps *latent states on the Poincare ball* to action distributions. The
input lives in curved space; the output lives in flat Euclidean action space (joint torques,
velocities). The actor must bridge these two geometries.

The design is straightforward: use CovariantAttention to read chart-conditioned features
from the latent state, then project to the mean and log-standard-deviation of a Gaussian.
The Gaussian is squashed through tanh to respect the action bounds of dm_control
environments (typically $[-1, 1]$).

This is the same squashed Gaussian trick that SAC uses, but the input representation is
richer: the actor sees not just a latent vector, but also the chart routing weights that
tell it which region of state space the agent occupies.
:::

The actor reuses the chart-conditioned attention pattern from `CovariantControlField`
(`covariant_world_model.py`, line 275) but with two differences: (1) it omits the
`ActionTokenizer` since a policy should not condition on its own output, and (2) it
replaces the force projection with dual heads for the mean and log-standard-deviation
of a Gaussian:

```python
# PROPOSED: GeometricActor — chart-conditioned policy
class GeometricActor(nn.Module):
    """Gaussian policy conditioned on Poincare ball state and chart routing.

    Uses the same ChartTokenizer, CovariantAttention, and z_embed pattern
    as CovariantControlField, but without ActionTokenizer (the policy
    conditions on state, not on its own actions).
    """

    def __init__(self, latent_dim, action_dim, num_charts, d_model=128):
        super().__init__()
        # ---- shared with CovariantControlField ----
        self.chart_tok = ChartTokenizer(num_charts, d_model, latent_dim)
        self.z_embed = SpectralLinear(latent_dim, d_model)
        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.attn = CovariantAttention(geo_cfg)
        # ---- actor-specific output heads ----
        self.mu_head = SpectralLinear(d_model, action_dim)
        self.log_std_head = SpectralLinear(d_model, action_dim)
        self.log_std_min, self.log_std_max = -5.0, 2.0

    def forward(self, z, rw):
        chart_x, chart_z = self.chart_tok(rw, z)
        x_q = self.z_embed(z)
        feat, _ = self.attn(z, chart_z, x_q, chart_x, chart_x)
        mu = self.mu_head(feat)                           # [B, A]
        log_std = self.log_std_head(feat).clamp(
            self.log_std_min, self.log_std_max,
        )                                                  # [B, A]
        return mu, log_std

    def sample(self, z, rw):
        mu, log_std = self.forward(z, rw)
        dist = torch.distributions.Normal(mu, log_std.exp())
        x = dist.rsample()                                # reparameterized
        action = torch.tanh(x)                             # squash to [-1, 1]
        # Log-probability with tanh correction (SAC)
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)      # [B, 1]
        return action, log_prob
```

The novel code is the two `SpectralLinear` output heads and the omission of
`ActionTokenizer`. The chart-conditioning path --- `ChartTokenizer`,
`CovariantAttention`, `GeodesicConfig`, `z_embed` --- follows the identical
pattern used by `CovariantControlField` and `CovariantPotentialNet`.

### Entropy Regularization

The MaxEnt objective ({prf:ref}`def-maxent-rl-objective-on-macrostates`) adds an entropy
bonus to the return:

$$
J_{T_c}(\pi) = \mathbb{E}_\pi\Bigl[\sum_{t \ge 0} \gamma^t
\bigl(\mathcal{R}(z_t, a_t) + T_c \mathcal{H}(\pi(\cdot \mid z_t))\bigr)\Bigr]
$$

The cognitive temperature $T_c$ ({prf:ref}`def-cognitive-temperature`) controls the
exploration-exploitation trade-off. In the actor loss, this appears as:

$$
\mathcal{L}_{\text{actor}} = -\mathbb{E}_{z \sim \text{imagine}}\bigl[
R_t^\lambda - T_c \log \pi(a_t \mid z_t)\bigr]
$$

where $R_t^\lambda$ is the $\lambda$-return target (see {ref}`sec-imagination-training`).

:::{admonition} Researcher Bridge: SAC and MaxEnt Control
:class: info
The connection between Geometric Dreamer and SAC is not superficial. The equivalence theorem
({prf:ref}`thm-equivalence-of-entropy-regularized-control-forms-discrete-macro`) proves that
MaxEnt control, exponentially tilted trajectory measures, and soft Bellman optimality are
identical. Our actor optimizes the same objective as SAC, but in a learned hyperbolic latent
space with chart-conditioned features rather than raw observations. The cognitive temperature
$T_c$ plays the same role as SAC's entropy coefficient $\alpha$.
:::


(sec-critic-as-potential)=
## 5. The Critic as Effective Potential

:::{div} feynman-prose
Here is what I think is the most elegant part of the design. The world model already
contains a value function. It has been there all along.

Look at `CovariantPotentialNet`. It computes:

$$
\Phi_{\text{eff}}(z) = \alpha \cdot U(z) + (1 - \alpha) \cdot V_{\text{critic}}(z, K)
+ \gamma_{\text{risk}} \cdot \Psi_{\text{risk}}(z, K)
$$

The second term, $V_{\text{critic}}$, is a learned scalar function of the latent state
and chart assignment. It is computed by CovariantAttention over chart tokens. The same
attention features are projected through two *independent* heads: `v_out` produces the
scalar $V$ and `v_force_out` produces the force vector $f_{\text{critic}}$. The force is
a direct neural prediction, not an autograd gradient of $V$ --- the docstring says
"without `torch.autograd.grad` (no second-order graph)". The screened Poisson PDE loss
encourages $f_{\text{critic}} \approx -\nabla_G V_{\text{critic}}$ but this is an
approximate consistency enforced by training, not an exact computational relationship.

In the supervised pipeline (Phases 1--3), $V_{\text{critic}}$ is trained via the screened
Poisson PDE loss with a reward-density proxy. But in Phase 4, we can train it directly
as a *value function* using TD targets from imagined trajectories. The architecture does
not change. The loss does.

The analytic drive $U(z) = -2\,\text{artanh}(\|z\|)$ provides a *static* exploration
incentive: states near the boundary of the Poincare ball have lower potential, so the
dynamics naturally push toward high-information regions. This is a fixed geometric proxy
for the full exploration gradient $\nabla_G S_c$
({prf:ref}`def-exploration-gradient-metric-form`), which depends on the policy's path
entropy and would require on-policy rollouts to evaluate. The static drive captures the
geometry; the policy-dependent entropy is handled by the MaxEnt actor loss.
The risk term $\Psi_{\text{risk}}$ penalizes states where the metric is ill-conditioned.
Together, $\Phi_{\text{eff}}$ is a value function with built-in exploration and safety
priors.
:::

### TD-$\lambda$ Targets from Imagination

Given an imagined trajectory $\{z_0, a_0, \hat{r}_0, z_1, a_1, \hat{r}_1, \ldots, z_H\}$,
the $\lambda$-return at step $t$ is:

$$
R_t^\lambda = \hat{r}_t + \gamma\bigl[(1 - \lambda)\,V(z_{t+1})
+ \lambda\,R_{t+1}^\lambda\bigr], \qquad R_H^\lambda = V(z_H)
$$

where $V(z) = \Phi_{\text{eff}}(z)$ is the effective potential evaluated at $z$. The
recursion is computed backwards from $t = H-1$ to $t = 0$.

The critic loss minimizes the squared error between the predicted value and the
stop-gradient $\lambda$-return:

$$
\mathcal{L}_{\text{critic}} = \frac{1}{BH} \sum_{b,t}
\bigl(V(z_{b,t}) - \operatorname{sg}[R_{b,t}^\lambda]\bigr)^2
$$

:::{div} feynman-prose
The beauty of this design is that the critic and the dynamics share the same
CovariantAttention features. The scalar $V_{\text{critic}}$ and the force
$f_{\text{critic}}$ are produced by separate linear projections of the same
feature vector `v_feat`. They are not the same object --- improving $V$ does
not automatically improve $f$ --- but they are trained in tandem on the same
data, and the screened Poisson PDE loss pushes them toward consistency.

This is exactly what the theory says should happen. The effective potential
({prf:ref}`def-effective-potential`) is simultaneously the Hamiltonian that governs
dynamics and the value function that evaluates states. The shared feature representation
means that better chart conditioning and better positional embeddings benefit both
the value estimate and the force prediction simultaneously, even though the final
projections are independent.
:::


(sec-imagination-training)=
## 6. Imagination Training --- Actor-Critic in Latent Space

### The Imagination Rollout

The `imagine()` method extends `GeometricWorldModel` to interleave actor-generated actions
with BAOAB integration:

```python
# PROPOSED: GeometricWorldModel.imagine()
def imagine(self, z_0, rw_0, actor, horizon):
    """Roll out imagined trajectory using actor for action selection.

    Uses self.potential_net.predict_reward() for reward prediction —
    no separate reward module needed.

    Args:
        z_0: [B, D] initial latent state (from encoder, detached).
        rw_0: [B, K] initial router weights (from encoder, detached).
        actor: GeometricActor that maps (z, rw) -> (action, log_prob).
        horizon: Number of imagination steps.

    Returns:
        dict with z_traj [B, H, D], actions [B, H, A], rewards [B, H],
        log_probs [B, H], phi_eff [B, H, 1].
    """
    z, rw = z_0, rw_0
    p = self.momentum_init(z_0)                           # [B, D] — initialize momentum
    z_list, a_list, r_list, lp_list, phi_list = [], [], [], [], []

    for t in range(horizon):
        # Actor generates action from current latent state
        action, log_prob = actor.sample(z, rw)           # [B, A], [B, 1]

        # Predict immediate reward (reward branch of potential_net)
        r_hat = self.potential_net.predict_reward(z, action, rw)  # [B, 1]

        # FIRST: Optional chart jump (Fisher-Rao component)
        if self.use_jump:
            z, p, chart_logits, rw, jumped = self._conditional_jump(z, p, action, rw)

        # THEN: n_refine_steps BAOAB sub-steps (Wasserstein component)
        for s in range(self.n_refine_steps):
            z, p, phi_eff, hodge_info = self._baoab_step(z, p, action, rw)

        z_list.append(z)
        a_list.append(action)
        r_list.append(r_hat.squeeze(-1))
        lp_list.append(log_prob.squeeze(-1))
        phi_list.append(phi_eff)

    return {
        "z_traj": torch.stack(z_list, dim=1),             # [B, H, D]
        "actions": torch.stack(a_list, dim=1),             # [B, H, A]
        "rewards": torch.stack(r_list, dim=1),             # [B, H]
        "log_probs": torch.stack(lp_list, dim=1),          # [B, H]
        "phi_eff": torch.stack(phi_list, dim=1),           # [B, H, 1]
        "z_final": z,                                       # [B, D]
    }
```

:::{div} feynman-prose
Notice what is happening here. The entire rollout is differentiable. Actions are sampled
via the reparameterization trick (rsample), rewards are predicted by a neural network,
and state transitions are computed by the BAOAB integrator (which uses differentiable
exponential maps and Christoffel corrections). Gradients flow from the actor loss all the
way back through the imagined trajectory to the actor parameters.

This is the Dreamer trick: instead of estimating policy gradients from noisy real-world
returns (which requires many samples), we compute exact gradients through a differentiable
world model (which requires an accurate model). The tradeoff is model bias vs. sample
efficiency. But our world model has CFL bounds
({ref}`sec-boris-baoab`) that prevent the imagined trajectories from diverging, so the
bias stays controlled even over long horizons.
:::

### The Combined Loss

Each imagination step produces three loss signals, applied to three separate parameter
groups:

```python
# PROPOSED: Phase 4 imagination training step (sketch)
# NOTE: encode_batch and encode_sequence are pseudocode helpers that loop
#       over frames, call encoder(obs_t), and collect (z_geo, rw, K).
def imagination_step(encoder, world_model, actor, batch, config):
    # 1. Encode real observations (no gradient to encoder in Phase 4)
    with torch.no_grad():
        z_0, rw_0 = encode_batch(encoder, batch["obs"][:, 0])

    # 2. Imagine trajectory (reward predicted inside via potential_net.predict_reward)
    imagination = world_model.imagine(
        z_0.detach(), rw_0.detach(), actor, config.imagination_horizon,
    )
    rewards = imagination["rewards"]        # [B, H]
    log_probs = imagination["log_probs"]    # [B, H]

    # 3. Compute lambda-returns from Phi_eff (V_critic branch)
    # NOTE: simplified sketch. The full loop (run_phase4 below) correctly
    # separates target computation (no_grad) from prediction (with grad).
    values = imagination["phi_eff"].squeeze(-1)  # [B, H] from BAOAB
    returns = compute_lambda_returns(
        rewards, values.detach(), config.gamma, config.lambda_gae,
    )                                       # [B, H]

    # 4. Critic loss (recompute values WITH gradients for backprop)
    L_critic = (values - returns.detach()).pow(2).mean()

    # 5. Actor loss (maximize returns + entropy)
    L_actor = -(returns.detach() - config.T_c * log_probs).mean()

    # 6. Reward loss (on real data, not imagination)
    r_pred = world_model.potential_net.predict_reward(
        z_0.detach(), batch["actions"][:, 0], rw_0.detach(),
    )
    L_reward = (r_pred.squeeze(-1) - batch["rewards"][:, 0]).pow(2).mean()

    return L_actor, L_critic, L_reward
```

$$
\mathcal{L}_{\text{Phase 4}} = \underbrace{\mathcal{L}_{\text{actor}}}_{\text{maximize returns}}
+ \underbrace{\mathcal{L}_{\text{critic}}}_{\text{fit value function}}
+ \underbrace{\mathcal{L}_{\text{reward}}}_{\text{predict rewards}}
+ \underbrace{\mathcal{L}_{\text{dynamics}}}_{\text{keep world model accurate}}
$$

The dynamics loss $\mathcal{L}_{\text{dynamics}}$ is the same Phase 2 loss
({ref}`sec-world-model-losses`) computed on real transition data, ensuring the world model
stays grounded as the policy changes the data distribution.

### Lambda-Return Computation

```python
# PROPOSED: compute_lambda_returns
def compute_lambda_returns(rewards, values, gamma, lambda_gae):
    """Compute GAE-style lambda-returns.

    Args:
        rewards: [B, H] predicted rewards.
        values: [B, H] critic values.
        gamma: Discount factor.
        lambda_gae: GAE lambda (0 = TD(0), 1 = Monte Carlo).

    Returns:
        returns: [B, H] lambda-return targets.
    """
    H = rewards.shape[1]
    returns = torch.zeros_like(rewards)
    last_return = values[:, -1]                           # bootstrap from final value

    for t in reversed(range(H)):
        last_return = rewards[:, t] + gamma * (
            (1 - lambda_gae) * values[:, t] + lambda_gae * last_return
        )
        returns[:, t] = last_return

    return returns
```


(sec-dmcontrol-integration)=
## 7. Integration with DM Control

:::{div} feynman-prose
Now let me explain how to connect this to real robots --- or rather, to simulated robots
in dm_control.

The dashboard supports eleven continuous control tasks:
cartpole, reacher, cheetah, walker, humanoid, hopper, finger, and acrobot. Each task
provides a flat observation vector (4--67 dimensions depending on the task) and a
continuous action space (1--21 dimensions). The observations include joint positions,
velocities, and sometimes contact forces --- exactly the kind of structured,
low-dimensional state information that our encoder can map directly to the Poincare ball.

The existing encoder expects features from a vision backbone (720 dimensions). For
state-based control, the existing `_SharedFeatureExtractor` (`atlas.py`, line 1728)
already handles this: it maps any `input_dim` to the encoder's hidden dimension via
a two-layer MLP. No new module is needed --- just set `input_dim` in the encoder config.
The rest of the pipeline --- chart routing, VQ, three-channel split --- works unchanged.
:::

### Feature Extraction for State-Based Control

No new module is needed. The encoder already contains a `_SharedFeatureExtractor`
(`atlas.py`, line 1728) that maps arbitrary `input_dim` → `hidden_dim`:

```python
# EXISTING: _SharedFeatureExtractor (atlas.py, lines 1728-1749)
class _SharedFeatureExtractor(nn.Module):
    """Shared feature extractor for hierarchical atlas stacks."""

    def __init__(self, input_dim, hidden_dim, latent_dim, bundle_size):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            SpectralLinear(input_dim, hidden_dim, bias=True),
            NormGatedGELU(...),
            SpectralLinear(hidden_dim, hidden_dim, bias=True),
            NormGatedGELU(...),
        )

    def forward(self, x):
        return self.feature_extractor(x)                  # [B, hidden_dim]
```

For dm_control, instantiate `PrimitiveAttentiveAtlasEncoder` with
`input_dim = obs_dim` (e.g., 24 for walker) instead of the default vision-backbone
dimension. The rest of the encoder --- chart routing, VQ, three-channel split --- works
unchanged.

### Task Configuration

:::{div} feynman-added
| Task | `obs_dim` | `action_dim` | Reward Structure | Difficulty |
|------|-----------|--------------|-----------------|------------|
| `cartpole-balance` | 5 | 1 | Dense (upright bonus) | Easy |
| `cartpole-swingup` | 5 | 1 | Dense (upright bonus) | Medium |
| `reacher-easy` | 6 | 2 | Dense (distance to target) | Easy |
| `reacher-hard` | 6 | 2 | Dense (distance to target) | Medium |
| `cheetah-run` | 17 | 6 | Dense (forward velocity) | Medium |
| `walker-walk` | 24 | 6 | Dense (forward + upright) | Medium |
| `walker-stand` | 24 | 6 | Dense (upright bonus) | Easy |
| `humanoid-walk` | 67 | 21 | Dense (forward velocity) | Hard |
| `hopper-stand` | 15 | 4 | Dense (upright + height) | Medium |
| `finger-spin` | 9 | 2 | Dense (angular velocity) | Medium |
| `acrobot-swingup` | 6 | 1 | Dense (tip height) | Hard |
:::

### Data Collection

Data collection uses the existing `RoboticFractalGas` infrastructure or a simple
random/learned policy rollout:

```python
# PROPOSED: collect_episode
# NOTE: The actual codebase uses plangym's batch API (env.step_batch).
#       This pseudocode shows the single-step logic for clarity.
def collect_episode(env, actor, encoder, max_steps=1000):
    """Collect one episode from dm_control environment via plangym.

    The encoder's built-in _SharedFeatureExtractor handles obs → features.

    Returns:
        dict with obs [T, D_obs], actions [T, A], rewards [T],
        dones [T] arrays.
    """
    obs_list, act_list, rew_list, done_list = [], [], [], []
    state, obs, info = env.reset(return_state=True)       # plangym 3-tuple

    for t in range(max_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            enc_out = encoder(obs_t)                        # feature extraction is internal
            z_geo, rw = enc_out[5], enc_out[4]             # [1, D], [1, K]
            action, _ = actor.sample(z_geo, rw)             # [1, A]

        action_np = action.squeeze(0).cpu().numpy()
        # plangym returns (state, obs, reward, done, info)
        state, next_obs, reward, done, info = env.step(state, action_np)

        obs_list.append(obs)
        act_list.append(action_np)
        rew_list.append(reward)
        done_list.append(done)

        obs = next_obs
        if done:
            break

    return {
        "obs": np.stack(obs_list),
        "actions": np.stack(act_list),
        "rewards": np.array(rew_list),
        "dones": np.array(done_list),
    }
```

### Replay Buffer

A simple sequence-based replay buffer stores complete episodes and samples fixed-length
subsequences for training:

```python
# PROPOSED: SequenceReplayBuffer
class SequenceReplayBuffer:
    """Stores episodes and samples fixed-length subsequences."""

    def __init__(self, capacity=100_000, seq_len=50):
        self.capacity = capacity
        self.seq_len = seq_len
        self.episodes = []
        self.total_steps = 0

    def add_episode(self, episode):
        self.episodes.append(episode)
        self.total_steps += len(episode["rewards"])
        while self.total_steps > self.capacity:
            removed = self.episodes.pop(0)
            self.total_steps -= len(removed["rewards"])

    def sample(self, batch_size):
        """Sample batch of subsequences [B, seq_len, ...]."""
        # Uniform random episode, then uniform random start index
        ...
```

:::{div} feynman-prose
The replay buffer is intentionally simple. Dreamer-v3 showed that a FIFO buffer with
uniform sampling works well for model-based RL --- the world model's imagination provides
the off-policy correction implicitly. Fancy prioritized replay is unnecessary when you
are training on *imagined* data, because the imagination distribution is controlled by
the actor, not by the replay buffer.

For the fractal gas integration, you can also fill the buffer with trajectories from
`RoboticFractalGas` runs. The fractal gas explores the state space more broadly than
a random policy (thanks to the distance-based fitness), providing richer initial data
for the encoder and world model to learn from. Think of the fractal gas as a smart
data collector and the Geometric Dreamer as the learner that extracts a policy from
that data.
:::


(sec-four-phase-pipeline)=
## 8. The Four-Phase Training Pipeline

### Overview

:::{div} feynman-added
| Phase | What Trains | What Is Frozen | Data Source | Key Losses |
|-------|------------|----------------|-------------|------------|
| 1 | Encoder | --- | Single frames | Recon + VQ + routing |
| 2 | World model | Encoder | Sequences | Geodesic + chart CE + energy |
| 3 | Encoder + WM (alternating) | --- | Sequences | Joint + enclosure + Zeno |
| 4 | Actor + critic + reward branch + WM | Encoder | Real + imagined | Actor + critic + reward + dynamics |
:::

### Phase 4: RL Training

:::{div} feynman-prose
Phase 4 is where the agent learns to *act*. Phases 1--3 gave it perception (encoder) and
a world model (dynamics). Phase 4 adds purpose (rewards) and agency (policy).

The training alternates between two loops. The **data collection loop** runs the current
policy in the dm_control environment and stores transitions in the replay buffer. The
**imagination loop** samples real states from the buffer, encodes them, imagines
$H$-step rollouts using the actor and world model, and updates all networks.

The world model continues training on real data during Phase 4. This is critical: as the
policy improves, it visits new regions of state space that the world model has never seen.
Without continued dynamics training, the imagination would diverge in these new regions,
and the actor would learn to exploit model errors rather than earn real rewards.
:::

```python
# PROPOSED: Phase 4 training loop (extends train.py phase structure)
def run_phase4(encoder, world_model, actor, env, config):
    """Phase 4: Model-based RL training.

    The reward head is world_model.potential_net.predict_reward().
    The critic is world_model.potential_net (V_critic branch).
    """
    # Encoder already contains _SharedFeatureExtractor — just set input_dim in config
    buffer = SequenceReplayBuffer(
        capacity=config.buffer_capacity,
        seq_len=config.seq_len,
    )

    reward_fn = world_model.potential_net.predict_reward   # reward branch
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=config.lr_actor)
    optimizer_wm = torch.optim.Adam(                       # WM + reward + critic
        world_model.parameters(), lr=config.lr_wm_phase4,  # (potential_net is a sub-module)
    )

    # Freeze encoder in Phase 4
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    for epoch in range(config.phase4_epochs):

        # --- Data collection ---
        if epoch % config.collect_every == 0:
            episode = collect_episode(env, actor, encoder,
                                      max_steps=config.max_episode_steps)
            buffer.add_episode(episode)

        # --- Model + reward training (on real data) ---
        batch = buffer.sample(config.batch_size)
        with torch.no_grad():
            z_all, rw_all, K_all = encode_sequence(encoder, batch["obs"])

        optimizer_wm.zero_grad()

        # World model dynamics loss (keep model accurate)
        wm_out = world_model(
            z_all[:, 0].detach(), batch["actions"][:, :-1], rw_all[:, 0].detach(),
        )
        L_dynamics = compute_phase2_loss(wm_out, z_all[:, 1:], K_all[:, 1:], config)[0]

        # Reward branch loss (on real transitions)
        r_pred = reward_fn(
            z_all[:, :-1].detach(), batch["actions"][:, :-1], rw_all[:, :-1].detach(),
        )
        L_reward = (r_pred.squeeze(-1) - batch["rewards"][:, :-1]).pow(2).mean()

        (L_dynamics + L_reward).backward()
        optimizer_wm.step()

        # --- Imagination training ---
        z_0 = z_all[:, 0].detach()
        rw_0 = rw_all[:, 0].detach()
        imagination = world_model.imagine(
            z_0, rw_0, actor, config.imagination_horizon,
        )

        # Lambda-return targets
        rw_expand = rw_0.unsqueeze(1).expand(
            -1, config.imagination_horizon, -1,
        ).reshape(-1, config.num_charts)
        z_flat = imagination["z_traj"].reshape(-1, config.latent_dim)

        with torch.no_grad():
            _, phi_tgt = world_model.potential_net.force_and_potential(z_flat, rw_expand)
            values_tgt = phi_tgt.reshape(-1, config.imagination_horizon)

        returns = compute_lambda_returns(
            imagination["rewards"], values_tgt,
            config.gamma, config.lambda_gae,
        )

        # Critic loss (recompute WITH gradients for backprop through potential_net)
        optimizer_wm.zero_grad()
        _, phi_pred = world_model.potential_net.force_and_potential(
            z_flat.detach(), rw_expand.detach(),
        )
        values_pred = phi_pred.reshape(-1, config.imagination_horizon)
        L_critic = (values_pred - returns.detach()).pow(2).mean()
        L_critic.backward()
        optimizer_wm.step()

        # Actor loss (maximize returns + entropy)
        optimizer_actor.zero_grad()
        L_actor = -(returns.detach() - config.T_c * imagination["log_probs"]).mean()
        L_actor.backward()
        if config.grad_clip > 0:
            nn.utils.clip_grad_norm_(actor.parameters(), config.grad_clip)
        optimizer_actor.step()
```

### Hyperparameter Defaults

:::{div} feynman-added
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_actor` | 3e-4 | Actor learning rate (separate optimizer) |
| `lr_wm_phase4` | 1e-4 | World model + critic + reward branch (single optimizer, slower in Phase 4) |
| `imagination_horizon` | 15 | Steps of imagination per update |
| `gamma` | 0.99 | Discount factor |
| `lambda_gae` | 0.95 | GAE lambda for $\lambda$-returns |
| `T_c` | 0.1 | Cognitive temperature (entropy weight) |
| `collect_every` | 5 | Collect new episode every N training epochs |
| `buffer_capacity` | 100,000 | Replay buffer size (transitions) |
| `batch_size` | 32 | Training batch size |
| `max_episode_steps` | 1000 | Max steps per collected episode |
| `phase4_epochs` | 10,000 | Total Phase 4 training epochs |
| `grad_clip` | 100.0 | Gradient clipping norm |
:::

:::{div} feynman-prose
The world model learning rate is deliberately lower in Phase 4 ($10^{-4}$ vs $10^{-3}$ in
Phase 2). The model is already well-trained from Phases 2--3; we just need to keep it
calibrated as the policy shifts the data distribution. Too high a learning rate would
cause catastrophic forgetting of dynamics in under-visited regions.

The imagination horizon of 15 steps is a balance. Too short, and the actor does not see
the long-term consequences of its actions. Too long, and compounding model errors corrupt
the imagined returns. The CFL bounds ({ref}`sec-boris-baoab`) help here: by preventing
individual integration steps from overshooting, they keep the cumulative error growth
linear rather than exponential. In our experiments, 15 steps is where the marginal benefit
of longer imagination starts to flatten.
:::


(sec-dreamer-diagnostics)=
## 9. Diagnostics --- Is Your Agent Learning?

:::{div} feynman-prose
Model-based RL has more failure modes than model-free RL. The policy can fail, the value
function can fail, the reward model can fail, and the world model can fail. Worse, these
failures interact: a bad world model produces bad imaginations, which produce bad policy
gradients, which produce a bad policy, which explores badly, which starves the world model
of good data.

Here is what to watch.
:::

### Reward Model Accuracy

The reward prediction error on held-out real transitions is the most direct measure of
whether the imagination is trustworthy. Track:

- **Mean absolute error** (MAE): $|\hat{r} - r|$. Should decrease over training.
- **Correlation**: $\text{corr}(\hat{r}, r)$. Should approach 1.0.
- **Error by chart**: Break down MAE per chart $K$. High error in one chart means the
  reward head is poorly calibrated in that region.

### Imagination vs. Reality Divergence

Compare world model predictions against actual transitions:

- **Geodesic prediction error**: $d_H(z_{\text{pred}}, z_{\text{actual}})$ averaged over
  $H$-step rollouts. Plot as a function of horizon step $h$ --- the error should grow
  sublinearly. If it grows exponentially, the CFL bounds are too loose.
- **Chart agreement**: Fraction of steps where predicted and actual chart assignments
  match. Below 70\% suggests chart boundary instability.

### Actor-Critic Convergence

- **Mean episode return**: The primary metric. Should increase over training.
- **Actor entropy**: $-\mathbb{E}[\log \pi]$. Should start high (exploration) and
  gradually decrease (exploitation), but never collapse to zero.
- **Value estimation error**: $|V(z_t) - R_t^{\text{actual}}|$ on real trajectories.
  Should decrease as the critic learns.
- **Actor gradient norm**: Sudden spikes indicate unstable imagination. Apply gradient
  clipping.

### Hodge Diagnostics in Phase 4

The Hodge ratios ({ref}`sec-world-model-diagnostics`) acquire new meaning in the RL
context:

- **Conservative ratio** $\|f_{\text{cons}}\| / \|f_{\text{total}}\|$: In Phase 4, this
  should increase as the critic $V_{\text{critic}}$ dominates the dynamics with
  reward-aligned forces.
- **Solenoidal ratio**: For tasks with cyclic structure (e.g., locomotion gaits), a
  moderate solenoidal component is healthy --- the value curl captures the
  non-conservative aspect of periodic reward landscapes.
- **Harmonic residual**: Should remain small. A growing harmonic component suggests the
  Hodge decomposition is not capturing the full force structure.

:::{admonition} Training Health Checklist
:class: feynman-added tip

1. **First 100 epochs**: Reward MAE should drop below 0.5. Episode returns may still be
   near random. Actor entropy should be high.

2. **Epochs 100--1000**: Episode returns should start climbing. Value estimation error
   should decrease. Imagination divergence at $h = 15$ should be under $0.5$ hyperbolic
   distance.

3. **Epochs 1000--5000**: Returns should plateau near task-specific baselines. Actor
   entropy should stabilize at a moderate level. Conservative ratio should be 0.6--0.9.

4. **Warning signs**:
   - Returns plateau at zero: reward model is inaccurate. Check $\mathcal{L}_{\text{reward}}$.
   - Returns oscillate wildly: world model is inaccurate in regions the policy visits.
     Increase `collect_every` to gather more real data.
   - Actor entropy collapses: $T_c$ is too low. Increase cognitive temperature.
   - Imagination divergence grows exponentially: CFL bounds are too loose. Increase
     `min_length` or decrease `dt`.
:::


(sec-dreamer-theory-code)=
## 10. Theory-to-Code Correspondence

:::{div} feynman-prose
Let me give you the full map from theory to implementation. Every row connects a
theoretical concept --- with its formal definition or theorem --- to the code that
realizes it and the loss that trains it. If you understand this table, you understand
the entire Geometric Dreamer algorithm.
:::

:::{div} feynman-added
| Theory | Reference | Code | Loss |
|--------|-----------|------|------|
| MaxEnt objective $J_{T_c}(\pi)$ | {prf:ref}`def-maxent-rl-objective-on-macrostates` | `GeometricActor.sample` + $\lambda$-returns | $\mathcal{L}_{\text{actor}}$ |
| Soft Bellman fixed point | {prf:ref}`prop-soft-bellman-form-discrete-actions` | `V_critic` in `CovariantPotentialNet` | $\mathcal{L}_{\text{critic}}$ |
| Effective potential $\Phi_{\text{eff}}$ | {prf:ref}`def-effective-potential` | `CovariantPotentialNet.force_and_potential` | TD-$\lambda$ targets |
| Cognitive temperature $T_c$ | {prf:ref}`def-cognitive-temperature` | `config.T_c` in actor loss | Entropy bonus |
| Equivalence theorem | {prf:ref}`thm-equivalence-of-entropy-regularized-control-forms-discrete-macro` | MaxEnt actor = soft Bellman critic | Unified objective |
| Belief filtering | {ref}`sec-filtering-template-on-the-discrete-macro-register` | Encoder chart routing + VQ | Phase 1 losses |
| Geodesic jump diffusion | {ref}`sec-the-coupled-jump-diffusion-sde` | `_baoab_step` + `_conditional_jump` | Phase 2 dynamics losses |
| BAOAB splitting | {prf:ref}`def-baoab-splitting` | `GeometricWorldModel._baoab_step` | Energy conservation |
| Value curl $\mathcal{F}_{ij}$ | {prf:ref}`thm-hodge-decomposition` | `CovariantValueCurl` + Boris rotation | Hodge consistency |
| Reward 1-form $r_t = \langle \mathcal{R}, v\rangle_G$ | {prf:ref}`def-reward-1-form` | Reward branch in `CovariantPotentialNet` (contraction with velocity implicit in action-conditioned attention) | $\mathcal{L}_{\text{reward}}$ |
| Hodge decomposition | {ref}`sec-hodge-decomposition-of-value` | `HodgeDecomposer` | $\mathcal{L}_{\text{Hodge}}$ |
| Causal enclosure | {prf:ref}`def-causal-enclosure` | `SharedDynAtlasEncoder` + probe | Enclosure loss |
| Coupling window | {prf:ref}`thm-information-stability-window-operational` | Chart routing stability | Chart entropy bounds |
| Thermodynamic governor | {ref}`sec-optimizer-thermodynamic-governor` | Trust-region + varentropy brake | Optimizer stability |
| Exploration incentive | {prf:ref}`def-exploration-gradient-metric-form` | Path-entropy gradient $\nabla_G S_c$ motivates the analytic drive $U(z) = -2\,\text{artanh}(\|z\|)$ as a static proxy | Built into $\Phi_{\text{eff}}$ |
| WFR decomposition | {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces` | BAOAB (W$_2$) + chart jumps (FR) | Geodesic + chart losses |
| Three-channel latent | {prf:ref}`def-three-channel-latent` | $(K, z_n, z_{\text{tex}})$ in encoder | Recon + VQ + structure filter |
:::

### Summary Diagram

```{mermaid}
flowchart TB
    subgraph Phase1["Phase 1: Encoder"]
        OBS["Observation"] --> ENC["SharedDynAtlasEncoder"]
        ENC --> ZGEO["z_geo, rw, K"]
    end

    subgraph Phase23["Phases 2-3: World Model"]
        ZGEO --> WM["GeometricWorldModel"]
        WM --> DYN["BAOAB dynamics"]
        WM --> CHART["Chart prediction"]
    end

    subgraph Phase4["Phase 4: RL"]
        ZGEO --> ACTOR["GeometricActor"]
        ACTOR --> |"a ~ pi(.|z)"| WM
        WM --> |"z_traj"| REWARD["potential_net.predict_reward"]
        REWARD --> |"r_hat"| RET["Lambda-returns"]
        WM --> |"Phi_eff"| CRITIC["V_critic"]
        CRITIC --> RET
        RET --> |"L_actor"| ACTOR
        RET --> |"L_critic"| CRITIC
    end

    ENV["dm_control"] --> |"obs, reward"| Phase1
    ACTOR --> |"action"| ENV
```

:::{div} feynman-prose
And there it is. Four phases, each building on the last. Phase 1 gives the agent eyes.
Phase 2 gives it a physics engine. Phase 3 aligns the two. Phase 4 gives it purpose.

The entire algorithm rests on one idea: if you can imagine the future accurately enough,
you can learn a policy without ever making a real mistake. The Poincare ball gives the
imagination stability (CFL bounds, norm-preserving Boris rotation). The chart structure
gives it compositionality (local dynamics are simple, global dynamics are rich). And the
MaxEnt objective gives it robustness (entropy regularization prevents premature
commitment).

This is model-based RL the way it should be: a geometric physics engine in the agent's
head, trained to dream, and a policy trained to make those dreams come true.
:::
