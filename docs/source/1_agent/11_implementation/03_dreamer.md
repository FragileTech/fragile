(sec-geometric-dreamer)=
# Geometric Dreamer --- Model-Based RL on the Poincare Ball

## TLDR

- `src/fragile/learning/rl/` implements a **two-manifold Dreamer** whose motor variable is the **canonical action latent** on the action manifold.
- Perception and action both use the same topoencoder family as `train_joint`: one latent space for observations, one for actions, both with explicit chart/code/nuisance structure and texture excluded from control.
- `GeometricActor` maps the observation-side symbolic state to an action-side symbolic state, and the canonical motor variable is the resulting action-manifold latent `action_z_geo`.
- The environment action is produced by decoding that canonical action latent with `action_model.decoder(...)`.
- The world model, reward head, replay buffer, and imagination rollout are all conditioned on the canonical action latent.
- The critic is shared with `world_model.potential_net`, but it is not the direct controller. It provides value anchoring, exact-field certification, conservative reward decomposition, and diagnostics.
- The residual reward sector is trained conservatively: it is budgeted, norm-penalized, orthogonalized against the exact field, and gated until the exact conservative field is non-flat and force-consistent.
- RL closure in `train_dreamer.py` is no longer a separate closure model. It is measured from the observation Markov model plus an `EnclosureProbe` adversary that tests texture leakage.
- Actor improvement is trust-gated and conservative-first, with replay-action supervision used only as a bootstrap term and an old-policy hyperbolic anchor providing the persistent trust region.
- Auxiliary boundary-control modules remain in the repo for explicit control-field experiments, while `train_dreamer.py` is organized around the canonical action latent.

## Roadmap

1. Why geometric model-based RL?
2. Architecture overview: perception, canonical action construction, action realization, and imagination.
3. Reward modeling on joint state-action latent geometry.
4. The canonical-action controller.
5. The critic as shared value field.
6. Imagination training with canonical-action latent rollouts.
7. DM Control and Fractal Gas integration.
8. The training pipeline.
9. Diagnostics that matter in practice.
10. Theory-to-code correspondence.


(sec-why-geometric-mbrl)=
## 1. Why Geometric Model-Based RL

:::{div} feynman-prose
Let me start with the main point, because otherwise the code can look more complicated than it really is.

What we have built is not "Dreamer with curved decorations." The geometry is carrying real load. The latent state lives on the Poincare ball, the world model evolves that state with a Boris-BAOAB integrator, chart structure keeps local geometry explicit, and both observations and actions are represented by typed latent atlases rather than by flat Euclidean feature vectors.

That already moves us away from standard RSSM-style Dreamer. But the decisive design choice is this: the controller is organized around a **canonical action latent**. The policy does not first choose a Euclidean action and then retrofit geometry around it. Instead, it predicts a structured point on the action manifold, and that latent motor state is what the rest of the RL stack treats as primary.

This gives the code a clean ontology. The observation manifold tells you where you are. The action manifold tells you what motor-side state you choose. The environment action is only a realization of that motor state. The result is a geometric Dreamer whose motor semantics stay explicit all the way through replay, world modeling, reward modeling, and imagination.
:::

(rb-current-dreamer-status)=
:::{admonition} Dreamer Design
:class: info
`src/fragile/learning/rl/train_dreamer.py` follows this design:

- observations and actions each have their own `SharedDynTopoEncoder` manifold;
- `GeometricActor` predicts the action-side structured state from the observation-side structured state;
- the motor variable is the canonical action latent `action_z_geo`;
- action generation is handled by `action_model.decoder(...)`;
- replay stores action-manifold traces directly;
- the critic is shared with `world_model.potential_net`;
- actor-style improvement happens through trust-gated imagined discounted reward over the canonical action latent, with conservative-first weighting of the residual sector;
- replay-action supervision is scheduled as a bootstrap term and hands off to RL updates once the return gate opens;
- old-policy trust is enforced with hyperbolic geodesic and discrete KL penalties, not by permanently cloning replay actions;
- RL closure/no-leak diagnostics are computed with the observation world model plus an `EnclosureProbe`, not with a separate closure dynamics model.

Auxiliary boundary-control modules (`GeometricActionEncoder`, `GeometricActionBoundaryDecoder`, critic-gradient control helpers) remain in the package for explicit control-field experiments. The path documented in this chapter is the canonical-action-latent path in `train_dreamer.py`.
:::


(sec-dreamer-architecture-overview)=
## 2. Architecture Overview

The perception-imagination-action loop has four stages:

```{mermaid}
flowchart LR
    subgraph Perception["Perception"]
        OBS["obs [B, D_obs]"] --> ENC["SharedDynTopoEncoder"]
        ENC --> Z["z_geo [B, D]"]
        ENC --> RW["rw [B, K]"]
        ENC --> K["K_chart, K_code"]
    end

    subgraph Action["Canonical Action Construction"]
        K --> ACTOR["GeometricActor"]
        ACTOR --> ASTRUCT["K_chart^act, K_code^act, z_n^act"]
        ASTRUCT --> ACAN["a* = action_z_geo"]
    end

    subgraph Boundary["Action Realization"]
        ACAN --> ADEC["action_model.decoder"]
        ADEC --> ACT["action_mean / action"]
    end

    subgraph Imagination["World Model + Reward Head"]
        Z --> WM["GeometricWorldModel"]
        RW --> WM
        ACAN --> WM
        Z --> RH["RewardHead"]
        RW --> RH
        ACAN --> RH
        ACT --> RH
        Z --> CRIT["critic = world_model.potential_net"]
        RW --> CRIT
        WM --> ZT["z_traj, rw_traj, phi_eff"]
        RH --> RT["reward decomposition"]
    end

    ACT --> ENV["dm_control or Fractal Gas collection"]
    ENV --> OBS
```

### Module Summary

:::{div} feynman-added
| Module | Location | Input | Output | Role in RL path |
|--------|----------|-------|--------|-----------------------|
| `SharedDynTopoEncoder` | `src/fragile/learning/vla/shared_dyn/encoder.py` | observations | `z_geo`, `rw`, `K_chart`, `K_code`, nuisance/texture internals | perception and typed latent construction |
| `SharedDynTopoEncoder` | `src/fragile/learning/vla/shared_dyn/encoder.py` | actions | `z_geo`, `rw`, `K_chart`, `K_code`, nuisance/texture internals | action-manifold representation and replay supervision |
| `GeometricActor` | `src/fragile/learning/rl/actor.py` | observation-side `(K, z_n)` | action-side `(K^{act}, z_n^{act}, z_geo^{act})` | policy over the action manifold |
| `GeometricWorldModel` | `src/fragile/learning/vla/covariant_world_model.py` | `z_0`, canonical action latents, `rw_0` | latent rollout, chart logits, momenta, `phi_eff`, Hodge diagnostics | controlled geometric imagination |
| `RewardHead` | `src/fragile/learning/rl/reward_head.py` | `z`, action-manifold latent, action routing/code state, `rw` | residual reward form, residual scalar reward, reward density, curl diagnostics | reward modeling and decomposition |
| `SequenceReplayBuffer` | `src/fragile/learning/rl/replay_buffer.py` | full episodes | contiguous training subsequences | replay with action-manifold traces |
| `GeometricActionEncoder` | `src/fragile/learning/rl/boundary.py` | `z`, replay action, `rw` | explicit-control boundary variables | available for control-field experiments |
| `GeometricActionBoundaryDecoder` | `src/fragile/learning/rl/boundary.py` | `z`, explicit latent control, `rw` | deterministic action mean, execution action, motor texture stats | available for control-field experiments |
:::

### Typed Latents in the RL Path

The encoder still constructs the three-channel latent organization developed earlier in the book:

$$
Z_t = (K_t, z_{n,t}, z_{\mathrm{tex},t}).
$$

In the RL path, these variables remain explicit on both manifolds:

- on the observation side, the actor consumes `(K_{\text{chart}}, K_{\text{code}}, z_n)`;
- on the action side, replay stores `(K_{\text{chart}}^{act}, K_{\text{code}}^{act}, z_n^{act}, z_{\mathrm{geo}}^{act})`;
- `z_{\mathrm{tex}}` is excluded from the decision interface and remains outside control.

The canonical motor variable is therefore

$$
a_t^\star := z_{\mathrm{geo},t}^{act},
$$

the geometric action latent reconstructed from the action-side structured state. This preserves the texture firewall while keeping the motor state explicitly on the action manifold.


(sec-reward-head)=
## 3. The Reward Head --- Predicting Rewards in Curved Space

:::{div} feynman-prose
The world model tells you how the latent state moves. That is not enough for control. A controller also needs to know what is good and bad along that motion.

Reward is not just a scalar label you stick onto a state. It is a field on the joint state-action geometry. In this chapter, the primary motor coordinate is the canonical action latent, so the reward head evaluates the pair `(state latent, canonical action latent)` and splits reward into a conservative part and a residual non-conservative part.

That is the key modeling choice. The action manifold carries the motor variable directly. The reward head therefore learns how reward changes along that action-manifold coordinate instead of introducing a second downstream control variable. It gives the controller a scalar training signal and, at the same time, exposes how far the environment departs from the conservative idealization.
:::

The reward model is implemented as a standalone `RewardHead`, not as an internal branch of `CovariantPotentialNet`. It reuses the same chart tokenizer and positional embedding as the shared value field, then conditions on:

- the current latent position `z`,
- the canonical action latent `a^\star`,
- the action-manifold routing/code state,
- the chart routing weights `rw`.

The decomposition used in code is

$$
\hat r_t
=
\hat r_{\mathrm{cons},t}
+
\hat r_{\mathrm{noncons},t}.
$$

Operationally:

- the shared value field provides the exact conservative scalar signal used for replay targets and imagined conservative return;
- `RewardHead` predicts `reward_density`, `reward_form_cov`, `reward_nonconservative`, and `reward_curl`;
- `reward_nonconservative` is the contraction of the residual one-form with the canonical action latent;
- the trainer constructs the conservative replay target as `rewards - gated_residual`, so the residual sector is treated as leftover structure rather than as equal-capacity free fit.

The losses are

$$
\mathcal{L}_{\mathrm{reward}}
=
\frac{1}{BT}\sum_{b,t}
\bigl(\hat r_{b,t} - r_{b,t}\bigr)^2,
\qquad
\mathcal{L}_{\mathrm{noncons}}
=
\frac{1}{BT}\sum_{b,t}
\bigl(\hat r_{\mathrm{noncons},b,t} - g_t (r_{b,t} - \hat r_{\mathrm{cons},b,t})\bigr)^2,
$$

with a scalar gate $g_t \in [0,1]$ that depends on exact-field stiffness and exact-vs-direct force consistency. In addition, the trainer uses

$$
\mathcal{L}_{\mathrm{cons\text{-}match}}
=
\frac{1}{BT}\sum_{b,t}
\bigl(\hat r_{\mathrm{cons},b,t} - (r_{b,t} - g_t \hat r_{\mathrm{noncons},b,t})\bigr)^2,
$$

plus residual norm, residual budget, and exact-orthogonality penalties. So the residual head is trained to explain only the part of reward that the exact sector fails to explain *and* only once the exact sector is certified as usable.

:::{admonition} Implementation Note: Joint State-Action Reward Geometry
:class: feynman-added note
The reward split used here is

- a conservative component anchored by the shared scalar value field,
- a residual one-form-like component on joint state-action latent geometry,
- a contraction of that residual component with the canonical action latent,
- diagnostic curl statistics that quantify circulation in the residual sector,
- an on-policy residual gate that suppresses the non-conservative sector until the exact field is stiff enough and the direct conservative force matches the exact scalar-gradient field.

This keeps the theory centered on one motor variable: the canonical action latent carried by the action manifold.
:::


(sec-geometric-actor)=
## 4. The Action Controller --- Observation Latent to Canonical Action Latent

:::{div} feynman-prose
This is the section where the controller becomes easiest to state cleanly.

`GeometricActor` is what drives `train_dreamer.py`, but it should not be read as a flat Euclidean action regressor. It is a structured map from the observation manifold to the action manifold.

Why organize it this way? Because it gives the theory one unambiguous motor object. The policy does not produce a raw environment action first and then recover some deeper latent intention afterward. It produces the action-side structured latent state, and from that state the code constructs a canonical action latent. That latent is what the rest of the stack treats as primary.

So the important boundary is "action manifold to actuator coordinates." The environment action is a realization of the canonical action latent, not the fundamental policy object itself.
:::

The controller has three pieces.

### 4.1 Structured Action Prediction

Given the observation-side symbolic state, `GeometricActor` predicts the action-side symbolic state:

$$
(K_t^{obs}, K_{code,t}^{obs}, z_{n,t}^{obs})
\mapsto
(K_t^{act}, K_{code,t}^{act}, z_{n,t}^{act}, z_{geo,t}^{act}).
$$

This keeps chart identity, code identity, and nuisance coordinates explicit in the policy rather than collapsing everything into a single flat head.

### 4.2 Canonical Action Latent

The motor variable is

$$
a_t^\star := z_{geo,t}^{act}.
$$

This is the **canonical action representation** used by the rest of the RL stack:

- replay stores it,
- the world model conditions on it,
- the reward head contracts against it,
- imagination rolls forward under it,
- actor-return improvement differentiates through it.

This is the core theoretical choice that makes the implementation coherent.

### 4.3 Action Realization

The environment-facing action is produced by decoding the canonical action latent with the action topoencoder decoder:

$$
a_t = D_{\mathrm{act}}(a_t^\star, rw_t^{act}).
$$

Texture remains outside the control interface. The decoder used in `train_dreamer.py` is deterministic, and action noise is treated as rollout-time execution detail rather than as the primary policy object.

:::{admonition} Status of `GeometricActor`
:class: feynman-added note
`src/fragile/learning/rl/actor.py` is the controller used by `train_dreamer.py`.

Auxiliary boundary-control modules in `boundary.py` remain useful for explicit control-field experiments and parallel formulations of the motor interface.
:::


(sec-critic-as-potential)=
## 5. The Critic as Effective Potential

:::{div} feynman-prose
Now we come to the central identification in this implementation: the critic is not a separate network sitting off to the side. It is the value branch of the world model's own potential network.

That is elegant, but it also has consequences. It means the same geometric object that shapes the latent dynamics also provides the value signal used for reward anchoring and diagnostics. In the best case, this is exactly what the theory wants: one coherent scalar landscape, not one network for physics and another network for preferences. In the messy reality of training, it means you have to keep several roles in balance at once. The critic must remain a good scalar anchor for replay returns. It must remain compatible with the screened Poisson structure. And it must stay smooth enough to support stable conservative reward decomposition over imagined trajectories.

So the critic here is not just "a value head." It is a shared geometric field with several simultaneous obligations. What it is not is the direct controller. The motor variable is the canonical action latent predicted by the actor.
:::

In the code:

$$
\texttt{critic} = \texttt{world\_model.potential\_net}.
$$

The value used operationally for conservative reward decomposition and diagnostics is the scalar `task_value(z, rw)`. The world model continues to use the same potential network to produce conservative forces and effective potentials.

The critic losses used in the training loop are:

$$
\mathcal{L}_{\mathrm{value}}
=
\operatorname{MSE}\!\bigl(V(z_t), R_t^{\mathrm{replay}}\bigr),
$$

with replay return-to-go targets, and

$$
\mathcal{L}_{\mathrm{Poisson}}
=
\text{screened Poisson consistency loss},
$$

using the conservative reward density estimate from the reward head, together with

$$
\mathcal{L}_{\mathrm{exact\text{-}increment}}
=
\operatorname{MSE}\!\bigl(
\hat r_{\mathrm{cons},t},
r^{\mathrm{cons,target}}_t
\bigr),
$$

$$
\mathcal{L}_{\mathrm{covector}}
=
\operatorname{MSE}\!\bigl(
\text{discounted local exact increment from } dV,
r^{\mathrm{cons,target}}_t
\bigr),
$$

and an adaptive stiffness penalty that keeps the exact covector from collapsing to the zero field on small-reward tasks.

The critic objective is therefore

$$
\mathcal{L}_{\mathrm{critic}}
=
w_{\mathrm{screened\_poisson}}(t)\,
\mathcal{L}_{\mathrm{Poisson}}
+
w_{\mathrm{critic}}\,
\mathcal{L}_{\mathrm{value}}
+
w_{\mathrm{critic\_exact\_increment}}\,
\mathcal{L}_{\mathrm{exact\text{-}increment}}
+
w_{\mathrm{critic\_covector\_align}}\,
\mathcal{L}_{\mathrm{covector}}
+
w_{\mathrm{critic\_stiffness}}\,
\mathcal{L}_{\mathrm{stiffness}}.
$$

Here `w_screened_poisson(t)` is linearly warmed up, so the PDE term shapes an already nontrivial value field instead of dominating an almost-flat field from the start.

:::{admonition} Implementation Note: Exact vs Direct Conservative Fields
:class: feynman-added note
The trainer now distinguishes two conservative objects:

- an **exact** conservative field obtained from the scalar heads inside `world_model.potential_net`;
- a **direct** conservative field used by the fast dynamics rollout.

Force-consistency losses distill the exact field into the direct field, and diagnostics log exact and direct Hodge ratios separately. The trust gate and conservative certification use the exact field, not the raw direct-force surrogate.
:::


(sec-imagination-training)=
## 6. Imagination Training --- Canonical-Action Latent Rollouts

:::{div} feynman-prose
Imagination is the place where the whole controller is composed into one differentiable latent rollout.

Start from a latent state `(z_0, rw_0)`. At each step, the agent symbolizes the observation-side state, predicts an action-side structured state, constructs the canonical action latent `a_t^\star`, decodes a deterministic environment action for logging, evaluates reward, and advances the world model under that same canonical action latent. So the imagined trajectory is a controlled path on the joint state-action latent geometry.

This is what makes the theory coherent. The same motor variable appears in policy prediction, action realization, reward evaluation, and latent dynamics.
:::

The imagination rollout records:

- policy states before action,
- chart routing before action,
- canonical action latents,
- action-manifold latents and router weights,
- deterministic decoded actions,
- reward decomposition,
- world-model state trajectory,
- effective potential values.

Schematically, for horizon $H$:

$$
(z_0, rw_0)
\mapsto
\{a_t^\star, a_t, \hat r_t, z_{t+1}, rw_{t+1}, \Phi_{\mathrm{eff},t}\}_{t=0}^{H-1}.
$$

### Actor-Return Improvement

The periodic actor-style improvement step differentiates discounted imagined reward through:

- `GeometricActor`,
- the action decoder,
- the reward head,
- the differentiable world model rollout.

The imagined-return step is conservative-first. The rollout logs separate discounted exact-sector and residual-sector returns, and the residual contribution is weighted by a trust-derived power law:

$$
\hat J_{\mathrm{actor}}
=
\hat J_{\mathrm{cons}}
+
\tau_{\mathrm{return}}^{\,p}\,
\hat J_{\mathrm{noncons}},
$$

and the return term in the loss is

$$
\mathcal{L}_{\mathrm{actor\text{-}return}}
=
- w_{\mathrm{actor\_return}}\,
g_{\mathrm{actor}}\,
\hat J_{\mathrm{actor}},
$$

where the gate $g_{\mathrm{actor}}$ combines:

- chart-prediction accuracy under the current policy rollout,
- exact-vs-direct force consistency,
- world-model / policy routing synchronization,
- exact Hodge conservative ratio,
- actor-scale trust and actor-stiffness trust,
- an exact-control calibration gate built from replay exact-increment error, on-policy covector-alignment error, and on-policy exact-field calibration ratio.

This update is applied only periodically, according to the configured warmup and update frequency, and only if the trust gate is nonzero.

### Actor Trust Region and Bootstrap Schedule

The actor loss is not just replay cloning plus return. The current implementation combines:

- a replay-action supervision term whose scale decays after warmup and is further suppressed by the return gate;
- a persistent old-policy hyperbolic anchor `d_H(\pi_\theta, \pi_{\theta_{\mathrm{old}}})`;
- old-policy KL penalties on chart and code logits;
- a gauge-covariant natural objective built from the exact covector and gated residual reward form;
- scale-barrier, world-model sync, and stiffness penalties.

So replay supervision is now a bootstrap scaffold, not the permanent trust region. The persistent trust region is the old-policy anchor in action-manifold geometry.

### Return Diagnostics

The optimization target is the discounted imagined reward above. Alongside it, the code logs three horizon summaries separately:

$$
\sum_{t=0}^{H-1}\gamma^t \hat r_{\mathrm{cons},t},
\qquad
\sum_{t=0}^{H-1}\gamma^t \hat r_{\mathrm{noncons},t},
\qquad
V(z_H).
$$

These diagnostics tell you how much of the imagined return comes from the conservative value-backed sector, how much comes from the residual sector, and what terminal value remains at the end of the rollout.


(sec-dmcontrol-integration)=
## 7. Integration with DM Control

:::{div} feynman-prose
Now let us connect all this machinery to actual data collection.

The environments are ordinary continuous-control tasks, but the rollout interface is richer than a standard actor-environment loop. The code does not store only observations, actions, rewards, and done flags. It also stores the inferred latent controls and the decoded motor-boundary variables associated with those actions. That is exactly what you should expect from a theory in which action is a boundary condition rather than a single flat number emitted by a policy head.
The environments are ordinary continuous-control tasks, but the rollout interface stores both the executed action and the action-manifold state that generated it. That is what you should expect from a theory in which the motor variable lives on an action manifold rather than in raw actuator coordinates.

This bookkeeping pays for itself during training. Replay batches supervise the action manifold, the actor's action-side symbolic predictions, the world model, the reward head, and the critic from the same collected trajectories.
:::

### Observation Path

For dm_control tasks, observations are flattened to a single vector and then passed through the encoder's built-in feature extractor and atlas pipeline. The current training script adjusts both `obs_dim` and `action_dim` at runtime to match the selected task if needed. `DMControlEnv` wraps `dm_control.suite.load(domain, task)` behind a `domain-task` string interface, flattens the observation dict, and exposes a gym-like action-space adapter.

The trainer also supports environment-specific presets through `task_preset`. At the moment, `task_preset=auto` recognizes both `cartpole-swingup` and `cartpole-balance`. Both presets use the smaller exact-heavy control stack, but they now diverge in two important ways:

- `cartpole-balance` keeps a shorter on-policy exact-supervision horizon and a lower motor temperature because the task already has dense stabilizing signal;
- `cartpole-swingup` uses stronger on-policy exact supervision, more seed data, and a hotter motor-noise schedule so the policy visits useful swing-up trajectories before the exact field is expected to certify control.

### Rollout Modes

The code supports three collection modes:

- seed-episode collection with random actions;
- online policy rollouts in dm_control, with deterministic decoded `action_means` plus execution noise at collection time;
- Fractal Gas collection with policy-guided walkers.

That execution noise is no longer a single fixed `sigma_motor` knob. The trainer supports a scheduleable motor temperature with epoch annealing and exact-field-aware cooling, so swing-up tasks can keep exploration high until the exact control gate starts to certify the conservative field.

### Replay Contents

The replay buffer stores full episodes and includes the following per-step arrays:

:::{div} feynman-added
| Key | Meaning |
|-----|---------|
| `obs` | flattened environment observations |
| `actions` | executed actions |
| `action_means` | deterministic decoded action means |
| `action_latents` | canonical action latents on the action manifold |
| `action_router_weights` | action-manifold routing weights |
| `action_charts` | action chart indices |
| `action_codes` | action code indices |
| `action_code_latents` | action-manifold code latents |
| `rewards` | environment rewards |
| `dones` | termination flags |
:::

This makes the replay buffer a training source not only for dynamics and value anchoring, but also for action-manifold supervision.


(sec-four-phase-pipeline)=
## 8. The Training Pipeline

### Overview

One update is best read as a joint four-stage loop:

:::{div} feynman-added
| Substep | What Updates | Main Signals |
|---------|--------------|--------------|
| Encoder + closure | observation topoencoder, action topoencoder, `EnclosureProbe` | reconstruction, routing, quantization, symbolic closure, Zeno, no-leak |
| Atlas sync | shared atlas state across modules | chart centers, codebooks, actor/action bindings |
| World model + reward + critic | world model, reward head, shared value field | geodesic dynamics, chart prediction, conservative-first reward fit, exact/direct force consistency, energy, Hodge, replay return anchoring, screened Poisson consistency |
| Actor | `GeometricActor`, `actor_old` snapshot | bootstrap replay supervision, old-policy trust region, trust-gated imagined return, gauge-covariant natural improvement |
:::

### 8.1 Encoder Update

The encoder remains active unless `freeze_encoder=True`. Its losses still include:

- Phase-1 reconstruction and atlas regularization,
- dynamics-code closure through the shared codebook probe,
- Zeno-style transition diagnostics,
- adversarial no-leak closure pressure through `EnclosureProbe`.

This keeps the representation trainable during RL unless the user explicitly freezes it.

### 8.2 Atlas Synchronization

After the encoder step, the code synchronizes atlas-dependent parameters across the observation model, action model, world model, critic, actor, `actor_old`, and reward head.

This keeps chart centers, codebooks, and action-manifold bindings consistent before the dynamics, reward, and actor updates run.

### 8.3 World-Model, Reward, and Critic Update

The world model is trained on detached encoder outputs. The action input is the canonical action latent encoded from replay action means by the action topoencoder. The losses are:

$$
\mathcal{L}_{\mathrm{wm}}
=
w_{\mathrm{dynamics}}\mathcal{L}_{\mathrm{geo}}
+ w_{\mathrm{dynamics}}\mathcal{L}_{\mathrm{chart}}
+ w_{\mathrm{reward}}\mathcal{L}_{\mathrm{reward}}
+ w_{\mathrm{reward\_conservative\_match}}\mathcal{L}_{\mathrm{cons\text{-}match}}
+ w_{\mathrm{reward\_nonconservative}}\mathcal{L}_{\mathrm{noncons}}
+ w_{\mathrm{reward\_exact\_orth}}\mathcal{L}_{\mathrm{exact\text{-}orth}}
+ w_{\mathrm{exact\_force\_consistency}}\mathcal{L}_{\mathrm{force\text{-}exact}}
+ w_{\mathrm{exact\_task\_force\_consistency}}\mathcal{L}_{\mathrm{task\text{-}force\text{-}exact}}
+ w_{\mathrm{exact\_risk\_force\_consistency}}\mathcal{L}_{\mathrm{risk\text{-}force\text{-}exact}}
+ w_{\mathrm{reward\_nonconservative\_norm}}\mathcal{L}_{\mathrm{residual\text{-}norm}}
+ w_{\mathrm{reward\_nonconservative\_budget}}\mathcal{L}_{\mathrm{residual\text{-}budget}}
+ w_{\mathrm{momentum}}\mathcal{L}_{\mathrm{momentum}}
+ w_{\mathrm{energy}}\mathcal{L}_{\mathrm{energy}}
+ w_{\mathrm{Hodge}}\mathcal{L}_{\mathrm{Hodge}}
+ w_{\mathrm{hodge\_conservative\_margin}}\mathcal{L}_{\mathrm{Hodge\text{-}cons}}
+ w_{\mathrm{hodge\_solenoidal}}\mathcal{L}_{\mathrm{Hodge\text{-}sol}}.
$$

with exact/direct conservative-field consistency losses and conservative-preference penalties added on top of the original dynamics/reward terms. So the dynamics model evolves under action-manifold coordinates rather than raw actuator coordinates, and the reward split is explicitly biased toward the exact field before the residual sector is allowed to explain replay reward.

### 8.4 Actor Update

The critic is anchored on replay return-to-go, exact conservative increments, multi-step covector alignment, adaptive stiffness, and screened Poisson consistency inside the world-model stage. The same exact-field calibration is also applied on policy-visited imagined rollouts. The actor stage combines:

- supervised prediction of replay action charts, action codes, and action-side nuisance coordinates, but with a scheduled bootstrap scale;
- a persistent old-policy hyperbolic anchor plus chart/code KL trust terms;
- periodic trust-gated imagined discounted reward through canonical-action latent rollouts;
- a gauge-covariant natural objective and world-model synchronization penalties.

Operationally, the actor-return gate now factors into:

- a general imagination trust certificate,
- actor scale and stiffness trust,
- an exact-control gate built from exact-increment calibration and on-policy exact-field calibration.

So a run can no longer receive large return-driven actor updates merely because the world model is self-consistent. The exact conservative field must also be calibrated enough to justify control.

### 8.5 Benchmarking Loop

The repository now includes a small control benchmark harness in `src/experiments/benchmark_dreamer_control.py`. It launches short Dreamer runs over a small task/seed set, then summarizes:

- best and final eval reward,
- final `rew_20`,
- exact and on-policy exact covector norms,
- exact-increment error,
- return trust, return gate, and exact-control gate,
- exact conservative Hodge ratio and policy force error.

This is intentionally not a giant sweep system. It is a compact evidence loop for validating whether theory-facing changes improve control across tasks rather than in a single cherry-picked log.

So the training loop is:

$$
\text{representation and closure update}
\rightarrow
\text{atlas sync}
\rightarrow
\text{world model, reward, and critic update}
\rightarrow
\text{actor update}.
$$


(sec-dreamer-diagnostics)=
## 9. Diagnostics --- Is Your Agent Learning?

:::{div} feynman-prose
In a system like this, "is training working?" is not a single question. You have to ask at least four different questions.

Is the representation still geometrically healthy? Is the action manifold staying organized? Is the world model still grounded on replay transitions? And are imagined rollouts under the canonical action latent producing useful returns?

That is why the diagnostic surface in the code is so broad. The important thing is not to stare at every metric all the time. The important thing is to group them by failure mode and to know what each group is trying to catch.
:::

### Representation and Atlas Health

- `chart/usage_entropy`, `chart/active_charts`, `chart/active_symbols`
- per-chart code entropy and active code counts
- `chart/router_confidence`, `chart/top2_gap`

These tell you whether the macro register is alive and whether the atlas is collapsing.

### Action-Manifold and Actor Health

- `closure/L_total`
- `closure/L_obs_state`
- `closure/obs_state_acc`
- `closure/enclosure_acc_full`, `closure/enclosure_acc_base`
- `closure/enclosure_defect_acc`, `closure/enclosure_defect_ce`
- `actor/L_chart`, `actor/L_code`, `actor/L_zn`
- `actor/L_supervise_raw`, `actor/L_supervise`, `actor/supervise_scale`
- `actor/L_old_policy_geodesic`, `actor/L_old_policy_chart_kl`, `actor/L_old_policy_code_kl`
- `actor/chart_acc`, `actor/code_acc`
- `actor/return_trust_used`, `actor/return_gate`, `actor/return_applied`
- `actor/regime_bc_weight`, `actor/regime_rl_weight`
- `actor/state_alpha`, `actor/state_beta_pi`, `actor/stiffness_trust`
- `policy/action_canonical_norm_mean`
- `policy/action_router_entropy`

These indicate whether the action manifold remains organized, whether symbolic obs-action closure is stable, whether texture leakage is being removed, and whether the actor has actually handed off from replay bootstrap to trust-certified RL updates.

### World-Model Health

- `wm/L_geodesic`
- `wm/L_chart`
- `wm/chart_acc`
- `wm/L_reward`
- `wm/L_reward_nonconservative`
- `wm/L_reward_conservative_match`
- `wm/L_reward_exact_orth`
- `wm/L_force_exact`
- `wm/reward_nonconservative_gate`
- `wm/force_exact_rel_mean`
- `wm/L_momentum`
- `wm/L_energy`
- `wm/L_hodge`
- `geometric/hodge_conservative_direct`
- `geometric/hodge_conservative_exact`

These are the primary indicators that imagination remains grounded on real transition data and that the direct rollout field is staying close to the exact scalar-gradient conservative field.

### Critic and Imagination Health

- `critic/L_value`
- `critic/L_poisson`
- `critic/L_exact_increment`
- `critic/L_covector_align`
- `critic/L_stiffness`
- `critic/value_abs_err`
- `critic/exact_covector_norm_mean`
- `critic/stiffness_target`
- `critic/replay_bellman_abs`
- `imagination/reward_mean`
- `imagination/discounted_reward_mean`
- `imagination/nonconservative_return_mean`
- `imagination/exact_boundary_mean`
- `imagination/terminal_value_mean`
- `imagination/full_return_mean`
- `imagination/router_drift`

These tell you whether the value field is calibrated and whether imagined rollouts keep a healthy balance between residual reward, conservative reward, and terminal value.

### Interpreting the Non-Conservative Sector

- `wm/reward_nonconservative_mean`
- `wm/reward_nonconservative_frac`
- `wm/reward_curl_norm_mean`
- `imagination/reward_nonconservative_mean`
- `imagination/nonconservative_return_mean`
- `imagination/reward_curl_norm_mean`

These should be read as audits of how far the reward geometry departs from its conservative component along the joint state-action latent trajectory.


(sec-dreamer-theory-code)=
## 10. Theory-to-Code Correspondence

:::{div} feynman-prose
This table states the theory directly in the same terms used by the code.
:::

:::{div} feynman-added
| Theory | Reference | Current code | Status |
|--------|-----------|--------------|--------|
| Three-channel latent | {ref}`sec-the-shutter-as-a-vq-vae` | topoencoder builds `K_chart`, `K_code`, `z_n`, `z_tex`, `z_geo` | Implemented |
| Texture firewall | {ref}`sec-the-shutter-as-a-vq-vae` | RL heads consume `z_geo` and `rw`, not `z_tex` | Implemented |
| Covariant geometric world model | {ref}`sec-why-physics-for-world-model` | `GeometricWorldModel` with BAOAB, chart jumps, Hodge diagnostics | Implemented |
| Two-manifold RL state | {ref}`sec-the-shutter-as-a-vq-vae` | observations and actions each use a `SharedDynTopoEncoder` atlas | Implemented |
| Canonical action latent | {ref}`sec-the-boundary-interface-symplectic-structure` | `a_t^\star = z_{geo,t}^{act}` is the motor variable shared across policy, replay, reward, and dynamics | Implemented |
| Action realization | {ref}`sec-the-boundary-interface-symplectic-structure` | `action_model.decoder(a_t^\star, rw_t^{act})` produces deterministic environment actions | Implemented |
| Replay-grounded action manifold | {ref}`sec-the-boundary-interface-symplectic-structure` | replay stores action latents, routing weights, chart/code indices, and code latents | Implemented |
| Reward decomposition | {prf:ref}`def-reward-1-form` | `RewardHead` predicts conservative and residual pieces on joint state-action latent geometry, with conservative-first replay targets and residual gating | Implemented |
| Controlled latent imagination | {ref}`sec-wfr-boundary-conditions-waking-vs-dreaming` | `_imagine` and `_imagine_actor_return` roll out the world model under canonical action latents | Implemented |
| Replay-grounded value anchoring | {ref}`sec-the-reward-field-value-forms-and-hodge-geometry` | replay RTG + screened Poisson + exact increment + covector alignment + adaptive stiffness | Implemented |
| Action-manifold policy | {prf:ref}`def-maxent-rl-objective-on-macrostates` | `GeometricActor` maps observation-side symbolic state to action-side symbolic state | Implemented |
| Exact-field certification | {ref}`sec-the-reward-field-value-forms-and-hodge-geometry` | exact scalar-gradient conservative field is certified separately from the fast direct rollout field, with force-consistency losses between them | Implemented |
| Causal enclosure / no-leak | {prf:ref}`def-causal-enclosure` | RL path uses `EnclosureProbe` to test and suppress texture leakage under the observation Markov model | Implemented |
| Policy trust region | {ref}`Link <rb-barriers-trust-regions>` | old-policy hyperbolic geodesic anchor, old-policy chart/code KL, and parameter-space Mach limit | Implemented |
| Auxiliary explicit-control boundary modules | {ref}`sec-the-boundary-interface-symplectic-structure` | `GeometricActionEncoder`, `GeometricActionBoundaryDecoder`, and control-field helpers remain available for alternative experiments | Available |
| Variance-curvature coupling | {prf:ref}`lem-variance-curvature-correspondence` | motor texture scales with inverse metric | Implemented in execution noise |
| WFR transport + reaction split | {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces` | BAOAB substeps plus optional chart jump | Implemented |
:::

### Practical Summary

This Dreamer implementation should be understood as:

1. A **two-manifold** geometric model-based RL system.
2. A **canonical-action-latent** controller.
3. A **joint state-action reward geometry** with conservative and residual sectors.
4. A **jointly trained** latent-world-model, value, and actor loop.

:::{div} feynman-prose
So what should you carry away from all this?

Carry away the architecture, not the brand name. Dreamer here means learning from imagined rollouts in a curved latent space. The observation manifold tells you where you are. The action manifold tells you what motor-side state you choose. The canonical action latent is the single motor variable shared by policy, replay, world model, reward head, and imagination. The critic anchors the conservative part of reward and the long-horizon value structure. The residual sector stays visible through the reward head and its diagnostics.

That is the theory realized by the code in `src/fragile/learning/rl/`.
:::
