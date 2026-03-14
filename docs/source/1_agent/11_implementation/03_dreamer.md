(sec-geometric-dreamer)=
# Geometric Dreamer --- Model-Based RL on the Poincare Ball

## TLDR

- The current implementation in `src/fragile/learning/rl/` is a **critic-induced control-field** variant of Dreamer, not the separate-actor MaxEnt variant described in older drafts.
- Perception still comes from the shared dynamics topoencoder: observations are encoded into atlas routing weights and a geometric latent on the Poincare ball.
- Control is produced in two stages:
  1. the critic defines a latent control covector by differentiating the value field;
  2. a boundary decoder maps that latent control into motor macrostate, nuisance, compliance, deterministic action mean, and execution-only motor texture.
- Imagination uses the covariant world model plus the learned reward head to roll out **controlled counterfactual trajectories** in latent space.
- The reward head is separate from the world model core. It decomposes reward into conservative and non-conservative pieces, but the deployed controller currently uses the **conservative-first** control law based on the critic gradient.
- Training is not organized as a clean standalone "Phase 4 actor-critic block." Instead, each update interleaves encoder losses, boundary losses, world-model losses, critic anchoring, and a periodic actor-return improvement step inside one joint loop.

## Roadmap

1. Why geometric model-based RL?
2. Architecture overview: perception, control field, boundary decoding, and imagination.
3. Reward modeling in the current code.
4. The deployed boundary controller.
5. The critic as shared value field.
6. Imagination training in the current implementation.
7. DM Control and Fractal Gas integration.
8. The current training pipeline.
9. Diagnostics that matter in practice.
10. Theory-to-code correspondence for the deployed path.


(sec-why-geometric-mbrl)=
## 1. Why Geometric Model-Based RL

:::{div} feynman-prose
Let me start with the main point, because otherwise the code can look more complicated than it really is.

What we have built is not "Dreamer with curved decorations." The geometry is carrying real load. The latent state lives on the Poincare ball, the world model evolves that state with a Boris-BAOAB integrator, chart structure keeps local geometry explicit, and the controller acts through boundary variables rather than by pretending actions are just another Euclidean feature vector.

That already moves us away from standard RSSM-style Dreamer. But the current implementation makes one more move. Instead of training a standalone stochastic actor and then asking the critic to chase it, the deployed controller is derived from the critic itself. The critic defines a scalar value field on latent space. Its gradient gives a control covector. A separate boundary decoder translates that internal control variable into actual motor commands.

This is closer to the control-field story developed in the rest of Volume 1: the policy is not an arbitrary black box; it is a structured boundary realization of an internal value-driven flow. The result is a system that is more geometric than standard Dreamer, but also more conservative than the fully general theory. In particular, the current code emphasizes the exact part of the reward/cost structure and treats the non-conservative sector mainly as a learned diagnostic decomposition rather than as a first-class driver of control.
:::

(rb-current-dreamer-status)=
:::{admonition} Researcher Bridge: Current Status vs Older Dreamer Drafts
:class: info
Older drafts of this chapter described a separate `GeometricActor`, MaxEnt entropy bonus, and `lambda`-return actor loss. The shipped code in `src/fragile/learning/rl/train_dreamer.py` now follows a different design:

- control is induced by the critic gradient;
- action generation is handled by `GeometricActionBoundaryDecoder`;
- replay actions are inverted by `GeometricActionEncoder`;
- the critic is shared with `world_model.potential_net`;
- actor-style improvement happens through imagined discounted reward, not through a separate stochastic actor objective.

This chapter documents the **current implementation** and treats the older actor-centric formulation as future or optional work rather than the deployed path.
:::


(sec-dreamer-architecture-overview)=
## 2. Architecture Overview

The deployed perception-imagination-action loop has four stages:

```{mermaid}
flowchart LR
    subgraph Perception["Perception"]
        OBS["obs [B, D_obs]"] --> ENC["SharedDynTopoEncoder"]
        ENC --> Z["z_geo [B, D]"]
        ENC --> RW["rw [B, K]"]
        ENC --> K["K_chart, K_code"]
    end

    subgraph Control["Internal Control Field"]
        Z --> CRIT["critic = world_model.potential_net"]
        RW --> CRIT
        CRIT --> GRAD["dV / dz"]
        GRAD --> CT["control_tan, control_cov"]
    end

    subgraph Boundary["Boundary Realization"]
        CT --> DEC["GeometricActionBoundaryDecoder"]
        Z --> DEC
        RW --> DEC
        DEC --> ACT["action_mean / action"]
        DEC --> MBOUND["motor macro / nuisance / compliance / texture"]
    end

    subgraph Imagination["World Model + Reward Head"]
        Z --> WM["GeometricWorldModel"]
        RW --> WM
        CT --> WM
        Z --> RH["RewardHead"]
        RW --> RH
        CT --> RH
        ACT --> RH
        WM --> ZT["z_traj, rw_traj, phi_eff"]
        RH --> RT["reward decomposition"]
    end

    ACT --> ENV["dm_control or Fractal Gas collection"]
    ENV --> OBS
```

### Module Summary

:::{div} feynman-added
| Module | Location | Input | Output | Role in deployed path |
|--------|----------|-------|--------|-----------------------|
| `SharedDynTopoEncoder` | `src/fragile/learning/vla/shared_dyn/encoder.py` | observations | `z_geo`, `rw`, `K_chart`, `K_code`, nuisance/texture internals | perception and typed latent construction |
| `GeometricWorldModel` | `src/fragile/learning/vla/covariant_world_model.py` | `z_0`, latent control covectors, `rw_0` | latent rollout, chart logits, momenta, `phi_eff`, Hodge diagnostics | controlled geometric imagination |
| `RewardHead` | `src/fragile/learning/rl/reward_head.py` | `z`, decoded boundary action, `rw`, latent control | conservative reward, non-conservative reward, total reward | reward modeling and decomposition |
| `GeometricActionEncoder` | `src/fragile/learning/rl/boundary.py` | `z`, replay action, `rw` | latent control, motor macro, nuisance, compliance | inverse boundary map on replay data |
| `GeometricActionBoundaryDecoder` | `src/fragile/learning/rl/boundary.py` | `z`, latent control, `rw` | deterministic action mean, execution action, motor texture stats | forward motor boundary map |
| `SequenceReplayBuffer` | `src/fragile/learning/rl/replay_buffer.py` | full episodes | contiguous training subsequences | replay with boundary traces |
| `GeometricActor` | `src/fragile/learning/rl/actor.py` | `z`, `rw` | Gaussian action distribution | present in module, not wired into `train_dreamer.py` |
:::

### Typed Latents in the RL Path

The encoder still constructs the three-channel latent organization developed earlier in the book:

$$
Z_t = (K_t, z_{n,t}, z_{\mathrm{tex},t}).
$$

In the deployed RL path, however, control heads do **not** consume these variables as a fully explicit tuple. Instead:

- `K_{\text{chart}}` appears operationally through routing weights `rw`;
- `K_{\text{code}}` is retained for encoder-side diagnostics and dynamics-code closure probes;
- `z_n` is folded into the geometric latent `z_{\mathrm{geo}}`;
- `z_{\mathrm{tex}}` is excluded from the control heads and remains outside the decision interface.

This means the current implementation preserves the **texture firewall** from the theory, while using a collapsed deployed control state `(z_{\mathrm{geo}}, rw)` rather than a fully factorized `(K, z_n)` controller.


(sec-reward-head)=
## 3. The Reward Head --- Predicting Rewards in Curved Space

:::{div} feynman-prose
The world model tells you how the latent state moves. That is not enough for control. A controller also needs to know what is good and bad along that motion.

Now, in the abstract theory, reward is not just a scalar label you stick onto a state. It is a field on the boundary and, in the most general case, it can have both exact and circulating pieces. The code mirrors that distinction, but in a very practical way. The reward head predicts a conservative scalar part and a non-conservative form-like part, then combines them into the scalar reward used by training.

This is an important detail. The current implementation has not yet promoted the full non-conservative sector into the control law itself. But it does not throw that structure away either. It learns it, measures it, and logs it. So you should think of the reward head as half model, half audit instrument: it gives the controller a scalar training signal and, at the same time, tells you how far the environment may be from the conservative idealization.
:::

The reward model is implemented as a standalone `RewardHead`, not as an internal branch of `CovariantPotentialNet`. It reuses the same chart tokenizer and positional embedding as the shared value field, then conditions on:

- the current latent position `z`,
- the deterministic boundary action mean,
- the chart routing weights `rw`,
- the tangent latent control field.

The decomposition used in code is:

$$
\hat r(z, a, rw, u)
=
\hat r_{\mathrm{cons}}(z, a, rw, u)

+ \hat r_{\mathrm{noncons}}(z, a, rw, u).
$$

Operationally:

- `reward_conservative` is a scalar output head;
- `reward_form_cov` is a learned covector-like field;
- `reward_nonconservative` is the contraction of that field with the current control;
- `reward_curl` is an antisymmetric diagnostic tensor used to monitor circulation.

This is the currently deployed reward loss:

$$
\mathcal{L}_{\mathrm{reward}}
=
\frac{1}{BT}\sum_{b,t}
\bigl(\hat r_{b,t} - r_{b,t}\bigr)^2,
\qquad
\mathcal{L}_{\mathrm{noncons}}
=
\frac{1}{BT}\sum_{b,t}\hat r_{\mathrm{noncons},b,t}^2.
$$

The second term makes the deployed controller **conservative-first**: the code learns the non-conservative residual but softly discourages it from dominating unless the data forces it to persist.

:::{admonition} Implementation Note: Conservative-First Reward Geometry
:class: feynman-added note
The broader theory supports a full gauge-covariant control signal based on $d_A V = dV - A$. The current Dreamer implementation stops one step earlier:

- it learns a decomposition consistent with that theory;
- it uses the conservative critic gradient as the deployed control driver;
- it logs non-conservative fractions and curl magnitudes as diagnostics.

This is a deliberate approximation, not a claim that the full gauge sector has already been closed in the controller.
:::


(sec-geometric-actor)=
## 4. The Boundary Controller --- Critic Gradient to Motor Flux

:::{div} feynman-prose
This is the section where the current implementation most clearly departs from the older actor-centric draft.

There is a `GeometricActor` module in the codebase. It is a sensible chart-conditioned Gaussian policy. But it is not the thing that actually drives `train_dreamer.py`. The deployed controller is more structural. First, the critic defines a scalar value field on latent space. Then the code differentiates that field with respect to latent position to obtain a control covector. Then a boundary decoder translates that internal control variable into an action that the environment can execute.

Why do it this way? Because it keeps the connection to the rest of the theory explicit. The action is not primary. The internal control field is primary. The action is a boundary realization of that field. The decoder is therefore not just a policy head; it is an implementation of the motor boundary condition.

This also explains the extra variables you see in the boundary modules. The decoder does not output only an action. It also outputs motor macro probabilities, motor nuisance coordinates, compliance matrices, and geometry-scaled texture noise. That is the code's way of keeping the motor boundary typed rather than collapsing everything into one flat action vector.
:::

The deployed controller has three pieces.

### 4.1 Critic-Induced Control Field

Given a latent state and chart routing, the controller computes:

$$
u_{\mathrm{cov}} = dV,
\qquad
u_{\mathrm{tan}} = G^{-1} u_{\mathrm{cov}}.
$$

In the current code this is obtained by differentiating the critic scalar with respect to latent position and then raising the index using the conformal metric.

This is a faithful implementation of the **conservative** control law. Relative to the full theory:

$$
\nabla_A V = G^{-1}(dV - A),
$$

the current implementation corresponds to the approximation $A = 0$ in the deployed controller.

### 4.2 Boundary Decoding

The `GeometricActionBoundaryDecoder` maps the tangent control field to:

- motor macro probabilities,
- motor nuisance coordinates,
- motor compliance matrix,
- deterministic boundary action mean,
- geometry-scaled execution texture.

The deterministic action used for planning and for replay supervision is:

$$
a_{\mathrm{mean}}
=
\tanh\!\Big(
a_{\mathrm{base}}
M_{\mathrm{comp}}\,u_{\mathrm{nuis}}
\Big).
$$

Here the nuisance and compliance variables are learned internal boundary variables, not direct environment observations.

### 4.3 Execution-Only Motor Texture

At execution time, the controller may inject motor texture:

$$
a_{\mathrm{exec}}
=
\tanh\!\bigl(a_{\mathrm{raw}} + \xi_{\mathrm{motor}}\bigr),
\qquad
\xi_{\mathrm{motor}}\sim \mathcal{N}(0,\sigma_{\mathrm{motor}}^2 G^{-1}).
$$

This preserves the theory's variance-curvature coupling in a restricted form: stochasticity scales with the inverse metric. In the current code this texture is **not** part of a MaxEnt actor loss; it is an execution-time perturbation used for deployed rollouts and data collection.

:::{admonition} Status of `GeometricActor`
:class: feynman-added note
`src/fragile/learning/rl/actor.py` contains a chart-conditioned Gaussian actor. It is currently best understood as an experimental or future-facing module:

- it matches the older MaxEnt Dreamer formulation;
- it is exported from the RL package;
- it is not the controller used by `train_dreamer.py`.

The authoritative deployed path is the critic-gradient controller plus boundary decoder described in this section.
:::


(sec-critic-as-potential)=
## 5. The Critic as Effective Potential

:::{div} feynman-prose
Now we come to the central identification in the current implementation: the critic is not a separate network sitting off to the side. It is the value branch of the world model's own potential network.

That is elegant, but it also has consequences. It means the same geometric object that shapes the latent dynamics also provides the value signal used for control. In the best case, this is exactly what the theory wants: one coherent scalar landscape, not one network for physics and another network for preferences. In the messy reality of training, it means you have to keep several roles in balance at once. The critic must remain a good scalar anchor for replay returns. It must remain compatible with the screened Poisson structure. And because the control field is derived from it, it also has to be smooth enough for action decoding and imagination.

So the critic here is not just "a value head." It is a shared geometric field with several simultaneous obligations. That is why the training loop keeps both replay-based anchoring and PDE-style regularization alive even during RL updates.
:::

In the deployed code:

$$
\texttt{critic} = \texttt{world\_model.potential\_net}.
$$

The value used operationally for control is the scalar `task_value(z, rw)`. The world model continues to use the same potential network to produce conservative forces and effective potentials.

The critic losses used in the current training loop are:

$$
\mathcal{L}_{\mathrm{value}}
=
\operatorname{MSE}\!\bigl(V(z_t), R_t^{\mathrm{replay}}\bigr),
$$

with replay return-to-go targets, and

$$
\mathcal{L}_{\mathrm{Poisson}}
=
\text{screened Poisson consistency loss}
$$

using the conservative reward density estimate from the reward head.

The deployed critic objective is therefore

$$
\mathcal{L}_{\mathrm{critic}}
=
w_{\mathrm{screened\_poisson}}\,
\mathcal{L}_{\mathrm{Poisson}}
+
w_{\mathrm{critic}}\,
\mathcal{L}_{\mathrm{value}}.
$$

This is **not** the same as the older chapter draft in which the critic was fitted only to imagined `lambda`-returns. The current implementation keeps the critic replay-anchored and PDE-regularized throughout RL training.


(sec-imagination-training)=
## 6. Imagination Training --- Controlled Counterfactual Rollouts

:::{div} feynman-prose
The word "imagination" can be misleading here, because the rest of the book gives it a very precise thermodynamic meaning. In the boundary chapter, dreaming is the reflective-boundary regime: no sensory inflow, no motor outflow, a sealed system recirculating internally.

The function called `_imagine` in the current code is not that idealized reflective mode. It is something more operational. It starts from an encoded latent state, recomputes the critic-induced control field at every step, decodes that field into a motor action, predicts reward, and advances the world model under that control. So what it imagines is not a passive dream. It imagines a **counterfactual controlled trajectory**.

That distinction matters because it keeps the terminology honest. The current implementation is doing model-based planning and policy improvement inside the latent model. It is not yet implementing the exact reflective-boundary dreaming mode from the field-theoretic chapters. Those are related ideas, but they are not identical.
:::

The deployed imagination rollout records:

- policy states before action,
- chart routing before action,
- tangent and covariant latent controls,
- decoded motor boundary variables,
- deterministic boundary actions,
- reward decomposition,
- world-model state trajectory,
- effective potential values.

Schematically, for horizon $H$:

$$
(z_0, rw_0)
\mapsto
\{u_t, a_t, \hat r_t, z_{t+1}, rw_{t+1}, \Phi_{\mathrm{eff},t}\}_{t=0}^{H-1}.
$$

### Actor-Return Improvement

The periodic actor-style improvement step does **not** use a separate stochastic actor and does **not** use `lambda`-returns. Instead, it differentiates the discounted imagined reward through:

- the critic-induced control field,
- the boundary decoder,
- the differentiable world model rollout.

The current objective is:

$$
\mathcal{L}_{\mathrm{actor\text{-}return}}
=
- w_{\mathrm{actor\_return}}
\cdot
\mathbb{E}\Big[\sum_{t=0}^{H-1}\gamma^t \hat r_t\Big].
$$

This update is applied only periodically, according to the configured warmup and update frequency.

### Boundary Value Diagnostic

Although the actor-return update itself uses discounted imagined reward only, the code also computes the terminal bootstrap contribution:

$$
V_{\mathrm{boundary}}
=
\gamma^H V(z_H),
$$

and tracks the combined control objective

$$
J_{\mathrm{imag}}
=
\sum_{t=0}^{H-1}\gamma^t \hat r_t

+ \gamma^H V(z_H)
$$

as a diagnostic. This is useful operationally because it reveals when long-horizon control is being dominated by terminal value rather than by reward accumulated along the imagined path.

:::{admonition} Terminology Note: Imagination vs Reflective Dreaming
:class: feynman-added warning
When this chapter says "imagination," it refers to the **implemented** `_imagine` and `_imagine_actor_return` routines in `train_dreamer.py`.

- They are controlled latent rollouts.
- They include internally generated motor actions.
- They should be read as planning/counterfactual evaluation routines.

They are therefore distinct from the stricter reflective-boundary dreaming mode introduced in {ref}`sec-wfr-boundary-conditions-waking-vs-dreaming`.
:::


(sec-dmcontrol-integration)=
## 7. Integration with DM Control

:::{div} feynman-prose
Now let us connect all this machinery to actual data collection.

The environments are ordinary continuous-control tasks, but the rollout interface is richer than a standard actor-environment loop. The code does not store only observations, actions, rewards, and done flags. It also stores the inferred latent controls and the decoded motor-boundary variables associated with those actions. That is exactly what you should expect from a theory in which action is a boundary condition rather than a single flat number emitted by a policy head.

This extra bookkeeping pays for itself during training. Replay batches can supervise the inverse boundary encoder, the forward boundary decoder, and the coupling between replay actions and critic-induced intentions, all from the same collected trajectories.
:::

### Observation Path

For dm_control tasks, observations are flattened to a single vector and then passed through the encoder's built-in feature extractor and atlas pipeline. The current training script adjusts `obs_dim` at runtime to match the selected task if needed.

### Rollout Modes

The code supports three collection modes:

- seed-episode collection with random actions;
- online policy rollouts in dm_control;
- Fractal Gas collection with policy-guided walkers.

### Replay Contents

The replay buffer stores full episodes and may include the following per-step arrays:

:::{div} feynman-added
| Key | Meaning |
|-----|---------|
| `obs` | flattened environment observations |
| `actions` | executed actions |
| `action_means` | deterministic boundary action means |
| `controls` / `controls_cov` | latent control covectors |
| `controls_tan` | latent tangent controls |
| `control_valid` | whether the control trace is valid supervision |
| `motor_macro_probs` | decoded motor macro distribution |
| `motor_nuisance` | decoded motor nuisance variables |
| `motor_compliance` | decoded compliance matrices |
| `rewards` | environment rewards |
| `dones` | termination flags |
:::

This makes the replay buffer a training source not only for dynamics and value anchoring, but also for the boundary map itself.


(sec-four-phase-pipeline)=
## 8. The Current Training Pipeline

### Overview

The present implementation should be read as a **joint Phase-4 loop** built on top of the earlier encoder and world-model machinery. It is most accurate to describe one update as four substeps:

:::{div} feynman-added
| Substep | What Updates | Main Signals |
|---------|--------------|--------------|
| Encoder | topoencoder + dynamics-code probe | reconstruction, VQ, routing, closure, Zeno |
| Boundary | action encoder + action decoder | action reconstruction, control supervision, cycle consistency, value-intent alignment |
| World model + reward | world model + reward head | geodesic dynamics, chart prediction, reward fit, energy, Hodge |
| Critic / actor-return | shared value field, periodic boundary improvement | replay return anchoring, screened Poisson consistency, imagined discounted reward |
:::

### 8.1 Encoder Update

The encoder remains active unless `freeze_encoder=True`. Its losses still include:

- Phase-1 reconstruction and atlas regularization,
- dynamics-code closure through the shared codebook probe,
- Zeno-style transition diagnostics.

This keeps the representation trainable during RL unless the user explicitly freezes it.

### 8.2 Boundary Update

The boundary stage trains two complementary maps:

- `GeometricActionEncoder`: replay action $\to$ latent control plus motor boundary variables,
- `GeometricActionBoundaryDecoder`: latent control $\to$ deterministic boundary action plus typed motor boundary state.

The losses combine:

- action reconstruction,
- decoder supervision from replay traces,
- control supervision where replay controls are valid,
- cycle consistency,
- motor macro supervision,
- nuisance/compliance supervision,
- alignment with the critic-induced value intent.

This is the most theory-specific part of the implementation because it is where latent control and motor execution are explicitly tied together.

### 8.3 World-Model and Reward Update

The world model is trained on detached encoder outputs. The current losses are:

$$
\mathcal{L}_{\mathrm{wm}}
=
w_{\mathrm{dynamics}}\mathcal{L}_{\mathrm{geo}}
+ w_{\mathrm{dynamics}}\mathcal{L}_{\mathrm{chart}}
+ w_{\mathrm{reward}}\mathcal{L}_{\mathrm{reward}}
+ w_{\mathrm{reward\_nonconservative}}\mathcal{L}_{\mathrm{noncons}}
+ w_{\mathrm{momentum}}\mathcal{L}_{\mathrm{momentum}}
+ w_{\mathrm{energy}}\mathcal{L}_{\mathrm{energy}}
+ w_{\mathrm{Hodge}}\mathcal{L}_{\mathrm{Hodge}}.
$$

The control input to the world model is the **latent control covector**, not the raw environment action. This is exactly the right choice for the current theory-facing implementation because it makes the world model evolve under internal control variables rather than under already-decoded actuator coordinates.

### 8.4 Critic Anchoring and Periodic Actor-Return Improvement

The critic is always anchored on replay return-to-go plus screened Poisson consistency. On top of that, the code periodically runs an actor-return step that backpropagates imagined discounted reward through the control field and boundary decoder.

So the current Phase-4 loop is best summarized as:

$$
\text{representation update}
\rightarrow
\text{boundary update}
\rightarrow
\text{dynamics/reward update}
\rightarrow
\text{value anchoring}
\rightarrow
\text{periodic control improvement}.
$$

This is not as cleanly factorized as the older separate-actor chapter draft, but it reflects the code that is actually running.


(sec-dreamer-diagnostics)=
## 9. Diagnostics --- Is Your Agent Learning?

:::{div} feynman-prose
In a system like this, "is training working?" is not a single question. You have to ask at least four different questions.

Is the representation still geometrically healthy? Is the boundary map still faithful? Is the world model still grounded on replay transitions? And is the value-driven control field actually producing useful imagined trajectories?

That is why the diagnostic surface in the current code is so broad. The important thing is not to stare at every metric all the time. The important thing is to group them by failure mode and to know what each group is trying to catch.
:::

### Representation and Atlas Health

- `chart/usage_entropy`, `chart/active_charts`, `chart/active_symbols`
- per-chart code entropy and active code counts
- `chart/router_confidence`, `chart/top2_gap`

These tell you whether the macro register is alive and whether the atlas is collapsing.

### Boundary Health

- `boundary/L_action_recon`
- `boundary/L_control_supervise`
- `boundary/L_macro_supervise`
- `boundary/L_motor_nuisance_supervise`
- `boundary/L_motor_compliance_supervise`
- `boundary/value_intent_cos`
- `boundary/control_raise_err`, `boundary/control_lower_err`
- `boundary/texture_std_mean`

These indicate whether the motor boundary map remains consistent with replay data and with the critic-induced control field.

### World-Model Health

- `wm/L_geodesic`
- `wm/L_chart`
- `wm/chart_acc`
- `wm/L_reward`
- `wm/L_reward_nonconservative`
- `wm/L_momentum`
- `wm/L_energy`
- `wm/L_hodge`

These are the primary indicators that imagination remains grounded on real transition data.

### Critic and Imagination Health

- `critic/L_value`
- `critic/L_poisson`
- `critic/value_abs_err`
- `critic/replay_bellman_abs`
- `imagination/reward_mean`
- `imagination/discounted_reward_mean`
- `imagination/boundary_value_mean`
- `imagination/bootstrap_share`
- `imagination/router_drift`

These tell you whether the value field is calibrated and whether imagined rollouts are reward-dominated or merely bootstrap-dominated.

### Interpreting the Non-Conservative Sector

- `wm/reward_nonconservative_mean`
- `wm/reward_nonconservative_frac`
- `wm/reward_curl_norm_mean`
- `imagination/reward_nonconservative_frac`
- `imagination/reward_curl_norm_mean`

In the current code these should be read mainly as **audits** of how far the environment appears to depart from the conservative approximation used by the deployed control law.


(sec-dreamer-theory-code)=
## 10. Theory-to-Code Correspondence

:::{div} feynman-prose
This final table is deliberately conservative. It does not list the theory as we might want the implementation to become. It lists the theory as it is actually realized by the current code.

That means some rows are exact correspondences, some are faithful approximations, and some are explicitly marked as deferred. That is the right way to document a living codebase. Otherwise the text turns into wishful thinking, which is worse than being incomplete.
:::

:::{div} feynman-added
| Theory | Reference | Current code | Status |
|--------|-----------|--------------|--------|
| Three-channel latent | {ref}`sec-the-shutter-as-a-vq-vae` | topoencoder builds `K_chart`, `K_code`, `z_n`, `z_tex`, `z_geo` | Implemented |
| Texture firewall | {ref}`sec-the-shutter-as-a-vq-vae` | RL heads consume `z_geo` and `rw`, not `z_tex` | Implemented |
| Covariant geometric world model | {ref}`sec-why-physics-for-world-model` | `GeometricWorldModel` with BAOAB, chart jumps, Hodge diagnostics | Implemented |
| Motor boundary realization | {ref}`sec-the-boundary-interface-symplectic-structure` | `GeometricActionBoundaryDecoder` | Implemented |
| Inverse boundary map | {ref}`sec-the-boundary-interface-symplectic-structure` | `GeometricActionEncoder` | Implemented |
| Conservative control field | {prf:ref}`def-entropy-regularized-objective-functional` | critic gradient raised by metric | Implemented as conservative approximation |
| Full gauge-covariant control $\nabla_A V$ | {ref}`sec-the-control-identity-cost-rate-along-flow` | reward head learns non-conservative residual, but controller still uses $dV$ | Deferred |
| Reward decomposition | {prf:ref}`def-reward-1-form` | `RewardHead` predicts conservative and non-conservative pieces | Partially implemented |
| Reflective dreaming mode | {ref}`sec-wfr-boundary-conditions-waking-vs-dreaming` | current `_imagine` performs controlled rollout with action decoding | Not yet identical |
| Replay-grounded value anchoring | {ref}`sec-the-control-identity-cost-rate-along-flow` | replay RTG + screened Poisson critic losses | Implemented |
| MaxEnt actor with entropy bonus | {prf:ref}`def-maxent-rl-objective-on-macrostates` | `GeometricActor` exists but is not wired into `train_dreamer.py` | Present but not deployed |
| Variance-curvature coupling | {prf:ref}`lem-variance-curvature-correspondence` | motor texture scales with inverse metric | Implemented in execution noise |
| WFR transport + reaction split | {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces` | BAOAB substeps plus optional chart jump | Implemented |
:::

### Practical Summary

The current Dreamer implementation should be understood as:

1. A **geometric, control-field** model-based RL system.
2. A **boundary-aware** controller with typed motor variables.
3. A **conservative-first** realization of the broader theory.
4. A **jointly trained** phase-4 loop rather than a clean standalone actor-critic block.

:::{div} feynman-prose
So what should you carry away from all this?

Carry away the architecture, not the brand name. The current code is called Dreamer because it learns from imagined rollouts in a world model. But the important thing is how it does that. It dreams in a curved latent space. It controls through a boundary decoder. It anchors its value field on replay and PDE structure. And it keeps the non-conservative sector visible, even though the deployed controller has not yet fully internalized it.

That is the honest state of the implementation. It is already much closer to the theory than ordinary model-based RL, and it still leaves a clear path for future work: promote the full $d_A V$ control law, separate reflective dreaming from controlled planning, and deploy the explicit MaxEnt actor only when that is actually the algorithm we want to run.
:::
