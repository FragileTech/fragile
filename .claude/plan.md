# Dreamer High-Value Next Steps Plan

## Goal

Implement the four highest-value, theory-consistent changes identified from the current `walker`, `cartpole-swingup`, and `cartpole-balance` runs:

1. make the exact conservative field matter more in control,
2. strengthen on-policy exact supervision,
3. improve exploration for swing-up tasks in a theory-consistent way,
4. add a small multi-seed benchmark harness so future iterations are judged by reproducible evidence instead of ad hoc logs.

The target is not to force reward up with hacks. The target is to make the control loop follow the exact-field story in the docs strongly enough that policy improvement transfers to the environment.

## Current Diagnosis

### What is already fixed enough

- residual reward and curl are no longer the main uncontrolled failure mode,
- old-policy trust-region behavior is in place,
- multi-step replay exact supervision exists,
- on-policy critic calibration exists,
- cartpole presets now prevent immediate representation collapse,
- `cartpole-balance` shows that the updated trainer can improve a control task.

### What is still blocking performance

1. The exact conservative critic field is still too weak numerically.
   - Logs still show `c_cov=0.0000` and `a_al=0.0000` on the tasks that matter.
   - That means the actor trust plumbing is no longer the main blocker; the exact field itself is not yet driving control strongly enough.

2. `cartpole-swingup` still lacks sufficient useful coverage.
   - The actor receives some RL updates, but the policy does not visit enough useful swing-up trajectories for the exact field to calibrate at the right scale.

3. Control still depends too much on approximate model rollouts.
   - The exact field is certified, but the actor objective still mostly uses a return path whose strongest signal is imagined rollouts rather than exact conservative structure.

4. We still do not have a compact benchmark loop for evaluating changes across tasks and seeds.
   - Without that, every architectural change risks being overfit to one run.

## Guiding Theory Principles

1. The exact/conservative sector should be the primary driver of value learning and control.
2. The residual/nonconservative sector should explain only what the exact sector cannot.
3. Policy trust should be certified on policy-visited states, not only replay states.
4. Exploration should act through the control/noise channel, not through external reward shaping.
5. Benchmarking should evaluate real environment reward, exact-field calibration, and trust behavior together.

## Scope

Files expected to change:

- `/home/guillem/fragiletech/fragile/src/fragile/learning/rl/config.py`
- `/home/guillem/fragiletech/fragile/src/fragile/learning/rl/train_dreamer.py`
- `/home/guillem/fragiletech/fragile/tests/fractalai/learning/test_dreamer.py`
- `/home/guillem/fragiletech/fragile/docs/source/1_agent/11_implementation/03_dreamer.md`
- one new small experiment script under `/home/guillem/fragiletech/fragile/src/experiments/` or `/home/guillem/fragiletech/fragile/scripts/`

## Workstream 1 — Make the exact field matter more in control

### Problem

The actor currently has:

- an exact conservative signal,
- a gauge-covariant natural term,
- trust gating,
- old-policy trust-region penalties,

but the exact conservative field still remains too flat to dominate the control update. The actor can therefore keep moving under a weakly calibrated field.

### Change A — Add multi-step exact field control supervision beyond one-step local matching

Extend exact-field training from one-step and replay-local structure to horizon-aware structure that is still exact-field consistent.

Implement two new critic-side targets built from the same exact scalar object:

1. multi-step exact increment target:
   - already partially present,
   - strengthen its role in presets and metrics,
   - expose its horizon-weighted calibration error clearly.

2. multi-step exact transport / control-alignment target:
   - align the exact covector with the hyperbolic displacement over `k` steps,
   - keep the target tied to the same discounted exact increment,
   - use the hyperbolic log map instead of a plain Euclidean state delta.

The important constraint is that both scalar and differential targets must continue to describe the same exact object.

### Change B — Make actor control more exact-sector dominant

Keep the current conservative-first actor design, but tighten it further:

- introduce a conservative-control weight in the actor objective derived from exact-field calibration,
- downweight actor-return updates when the exact conservative field is flat even if general trust is moderate,
- leave residual reward as a secondary correction controlled by exact calibration.

Concretely, extend the actor gate so it uses not only trust and stiffness, but also a direct exact-field calibration factor from:

- exact-increment calibration error,
- on-policy exact covector alignment error,
- on-policy exact covector norm relative to adaptive target.

This should not replace the existing trust gate. It should refine it so the actor cannot behave as if the exact field is ready when it is still numerically flat.

### Change C — Optionally expose a pure exact-control objective for small-task presets

For small control tasks such as cartpole, add a config option that makes the actor natural/control objective explicitly conservative-only for the first training phase.

That means:

- phase 1: optimize only exact conservative return/natural term,
- phase 2: reintroduce gated residual contribution once exact calibration is stable.

This is theory-consistent because the exact sector should lead and the residual should only enter after calibration.

## Workstream 2 — Strengthen on-policy exact supervision

### Problem

On-policy calibration exists, but it is still too light for tasks like `swingup`. The exact field is therefore certified on-policy in principle, but not strongly enough to alter learning dynamics.

### Change A — Increase on-policy critic supervision from auxiliary to structural

The current on-policy losses should remain separate in accounting, but become stronger in training behavior:

- raise their preset weight on tasks where exact control is failing,
- expose their calibration errors next to replay errors,
- make them part of the actor gate more directly.

This means the policy should not only consult on-policy trust metrics; it should be trained against them.

### Change B — Expand on-policy horizon and batch construction

Strengthen `_collect_policy_state_rollout(...)` usage so on-policy supervision is not overly myopic:

- allow longer on-policy horizon than the default cartpole setting,
- allow separate config for replay multi-step horizon and on-policy multi-step horizon,
- ensure on-policy batch sampling covers more than a tiny subset when the task is small enough to afford it.

### Change C — Add on-policy exact calibration metrics that are easy to compare against replay metrics

Add metrics such as:

- `critic/on_policy/exact_increment_abs_err`
- `critic/on_policy/exact_covector_norm_mean`
- `critic/on_policy/calibration_ratio`
- `actor/exact_control_gate`

These should show whether the exact field is weak everywhere or only weak on policy-visited states.

## Workstream 3 — Improve exploration for swing-up tasks without reward hacks

### Problem

`cartpole-swingup` needs policy coverage of useful swing-up trajectories before the exact field can learn a meaningful control direction. Current exploration uses thermal motor noise, but it is not yet sufficient or structured enough.

### Change A — Add scheduleable thermal motor exploration

Keep exploration in the control channel, but make it scheduled and task-aware:

- add `sigma_motor_init`, `sigma_motor_final`, and `sigma_motor_anneal_epochs`,
- default cartpole swing-up to a larger initial motor temperature than balance,
- anneal only after exact-field calibration improves.

This is theory-consistent because it changes the stochastic control process, not the task reward.

### Change B — Add exact-field-aware exploration cooling

Instead of annealing motor noise only by epoch, optionally gate the decay by exact-field readiness:

- if on-policy exact calibration is poor, keep exploration high,
- once exact conservative control metrics pass threshold, anneal toward the nominal task setting.

This should be implemented as a bounded schedule, not a discontinuous switch.

### Change C — Separate swing-up and balance presets

The current cartpole presets are still too close structurally. Split them more clearly:

- `balance`: lower exploration, shorter calibration horizon, preserve already-good behavior,
- `swingup`: higher initial exploration, longer on-policy horizon, stronger on-policy exact supervision, possibly longer actor-return horizon only after calibration.

## Workstream 4 — Add a compact multi-seed benchmark harness

### Problem

We currently rely on manual runs and log inspection. That makes it hard to know whether a change improved the algorithm or only one lucky run.

### Change

Add a small benchmark harness dedicated to Dreamer control debugging.

Requirements:

- run a small set of tasks and seeds, not a giant sweep,
- save a compact CSV or JSON summary,
- report both reward and theory metrics.

Minimum benchmark set:

- `cartpole-balance`
- `cartpole-swingup`
- optional `walker-walk` once the cartpole loop is stable

Per-run summary should include:

- best eval reward,
- final eval reward,
- final `rew_20`,
- final `critic/exact_covector_norm_mean`,
- final `critic/on_policy/exact_covector_norm_mean`,
- final `critic/exact_increment_abs_err`,
- final `actor/return_trust_used`,
- final `actor/return_gate`,
- final `actor/policy_hodge_conservative_exact_mean`,
- final `actor/policy_force_rel_err_mean`

The harness should reuse the trainer CLI rather than duplicating training logic.

## Implementation Sequence

### Phase 1 — Exact-field control strengthening

1. Add config knobs for:
   - exact-control gate scaling,
   - separate replay/on-policy multistep horizons if needed,
   - scheduleable motor noise.
2. Extend the critic metrics and exact-field gating logic in `train_dreamer.py`.
3. Tighten the actor gate so exact-field calibration matters explicitly.
4. Add unit tests for new exact-field gate helpers and preset behavior.

### Phase 2 — On-policy strengthening

1. Expand on-policy rollout usage and metrics.
2. Strengthen cartpole `swingup` preset.
3. Re-run short cartpole experiments to verify:
   - `balance` stays healthy,
   - `swingup` improves over current `0.0`-to-`1.7` regime.

### Phase 3 — Exploration schedule

1. Add scheduleable motor noise.
2. Split `cartpole_balance` and `cartpole_swingup` exploration defaults.
3. Verify the swing-up policy explores more broadly without destabilizing `balance`.

### Phase 4 — Benchmark harness

1. Add the small benchmark script.
2. Add one regression-style test that the harness can parse trainer outputs and emit a summary.
3. Use the harness to compare current baseline vs new exact-field changes over a few seeds.

## Tests and Validation

### Required static checks

- `uv run pytest -q tests/fractalai/learning/test_dreamer.py tests/fractalai/learning/test_reward_form_split.py`
- `uv run ruff check ...` on all touched files
- `python -m py_compile ...`

### Required behavioral checks

1. `cartpole-balance`
   - should retain or improve the current ~`143` eval regime,
   - should not regress into zero-reward behavior.

2. `cartpole-swingup`
   - should improve beyond the current failing run,
   - first target is to beat the previous `1.7` eval peak reliably,
   - second target is to show exact-field metrics moving with reward.

3. benchmark harness
   - should show whether improvements hold across at least 2-3 seeds.

## Success Criteria

A successful implementation of these next steps should produce all of the following:

- `balance` remains strong,
- `swingup` beats the previous low-reward plateau,
- exact-field metrics are no longer permanently near zero on the improved run,
- actor return updates correlate better with real reward than with purely imagined return,
- the benchmark summary makes future comparisons cheap and reproducible.

## Failure Criteria

If after this implementation:

- `c_cov` and on-policy exact covector metrics remain effectively zero,
- `swingup` still stays near zero despite stronger exploration and on-policy calibration,
- and `balance` regresses,

then the next bottleneck is likely deeper than trainer weighting. At that point the next investigation should shift to the exact critic parameterization itself or to using the exact conservative field more directly inside imagination on small tasks.

## Recommendation

Proceed with all four workstreams in the order above. The first two are the highest leverage and should be implemented before more long runs on `walker`.