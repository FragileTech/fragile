"""Generate C++ test fixtures from the reference Python implementation.

Runs the actual fragile.fractalai code (asymmetric_rescale, l2_norm,
FractalCloningOperator, FractalGas) while recording every RNG draw
(companion indices, uniform samples, actions, dt), and emits
``fixtures_generated.hpp`` with the inputs, recorded draws, and expected
outputs so the C++ tests can replay the exact same computation.

Run from the repo root with the project venv:

    uv run python fractal-gas-web/tests/fixtures/generate_fixtures.py

The MockEnv here implements the IDENTICAL dynamics as
fractal-gas-web/tests/mock_env.hpp — keep them in sync.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fragile.fractalai.fractalai import asymmetric_rescale, l2_norm
from fragile.fractalai import fractal_gas as fractal_gas_mod
from fragile.fractalai.fractal_gas import FractalGas
from fragile.fractalai.videogames import cloning as cloning_mod
from fragile.fractalai.videogames.cloning import FractalCloningOperator

OUT_PATH = Path(__file__).parent / "fixtures_generated.hpp"

DONE_THRESHOLD = 20.0  # keep in sync with mock_env.hpp


def _float_literal(v: float) -> str:
    text = f"{float(v):.9g}"
    if not any(c in text for c in ".einf"):
        text += ".0"
    return text + "f"


def fmt_floats(values) -> str:
    return "{" + ", ".join(_float_literal(v) for v in np.asarray(values).ravel()) + "}"


def fmt_ints(values) -> str:
    return "{" + ", ".join(str(int(v)) for v in np.asarray(values).ravel()) + "}"


class MockEnv:
    """Python twin of tests/mock_env.hpp (float32 arithmetic throughout)."""

    def reset(self, return_state: bool = True):
        state = np.zeros(3, dtype=np.float32)
        return state, state.copy(), {}

    def sample_action(self):
        raise RuntimeError("actions must come from the recorded sampler")

    def step_batch(self, states, actions, dt, return_state: bool = True):
        new_states, observs, rewards, dones = [], [], [], []
        for state, action, d in zip(states, actions, dt):
            s = state.astype(np.float32).copy()
            a = np.float32(action)
            df = np.float32(d)
            s[0] = s[0] + (a + np.float32(1.0)) * df
            s[1] = s[1] + np.float32(0.5) * df
            s[2] = s[2] + np.float32(1.0)
            new_states.append(s)
            observs.append(s.copy())
            rewards.append(
                np.float32(0.1) * (a - np.float32(1.0)) * df + np.float32(0.01) * s[0]
            )
            dones.append(bool(s[0] > DONE_THRESHOLD))
        truncated = [False] * len(dones)
        return (
            new_states,
            np.array(observs, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(truncated, dtype=bool),
            [{} for _ in dones],
        )


def gen_asymmetric_rescale_cases(out: list[str]) -> None:
    torch.manual_seed(31337)
    cases = [
        ("basic", torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])),
        ("negatives", torch.tensor([-3.5, 0.0, 2.25, -1.0, 7.5, -0.125])),
        ("all_equal", torch.tensor([2.5, 2.5, 2.5, 2.5])),  # std == 0 -> ones
        ("single", torch.tensor([4.2])),  # std is NaN -> ones
        ("random", torch.randn(32)),
        ("large_scale", torch.tensor([1e6, 2e6, -5e5, 3.3e6, 0.0])),
    ]
    names = []
    for name, x in cases:
        y = asymmetric_rescale(x)
        out.append(f"const std::vector<float> kAsym_{name}_in = {fmt_floats(x.numpy())};")
        out.append(f"const std::vector<float> kAsym_{name}_out = {fmt_floats(y.numpy())};")
        names.append(name)
    pairs = ", ".join(f'{{"{n}", &kAsym_{n}_in, &kAsym_{n}_out}}' for n in names)
    out.append(
        "struct AsymCase { const char* name; const std::vector<float>* in;"
        " const std::vector<float>* out; };"
    )
    out.append(f"const std::vector<AsymCase> kAsymCases = {{{pairs}}};")


def gen_l2_case(out: list[str]) -> None:
    torch.manual_seed(7)
    n, dim = 6, 5
    obs = torch.randn(n, dim)
    companions = torch.tensor([3, 0, 5, 1, 2, 4])
    dist = l2_norm(obs.reshape(n, -1), obs[companions].reshape(n, -1))
    out.append(f"const int kL2N = {n};")
    out.append(f"const int kL2Dim = {dim};")
    out.append(f"const std::vector<float> kL2Obs = {fmt_floats(obs.numpy())};")
    out.append(f"const std::vector<int32_t> kL2Companions = {fmt_ints(companions.numpy())};")
    out.append(f"const std::vector<float> kL2Expected = {fmt_floats(dist.numpy())};")


def gen_cloning_case(out: list[str]) -> None:
    """Record one calculate_fitness + decide_cloning with some dead walkers."""
    torch.manual_seed(1234)
    n, dim = 10, 6
    obs = torch.randn(n, dim)
    cum_rewards = torch.randn(n).abs() * 3
    step_rewards = torch.randn(n)
    alive = torch.ones(n, dtype=torch.bool)
    alive[2] = False
    alive[7] = False

    recorded_companions: list[torch.Tensor] = []
    recorded_uniforms: list[torch.Tensor] = []

    original_compas = cloning_mod.random_alive_compas
    original_rand_like = cloning_mod.torch.rand_like

    def recording_compas(oobs, ref):
        compas = original_compas(oobs, ref)
        recorded_companions.append(compas.clone())
        return compas

    def recording_rand_like(x):
        u = original_rand_like(x)
        recorded_uniforms.append(u.clone())
        return u

    cloning_mod.random_alive_compas = recording_compas
    cloning_mod.torch.rand_like = recording_rand_like
    try:
        op = FractalCloningOperator(dist_coef=1.5, reward_coef=0.75, use_cumulative_reward=True)
        vr, fit_companions = op.calculate_fitness(obs, cum_rewards, step_rewards, alive)
        clone_companions, will_clone = op.decide_cloning(vr, alive)
    finally:
        cloning_mod.random_alive_compas = original_compas
        cloning_mod.torch.rand_like = original_rand_like

    probs = (vr[clone_companions] - vr) / torch.where(
        vr > op.eps, vr, torch.tensor(op.eps, dtype=vr.dtype)
    )

    out.append(f"const int kCloneN = {n};")
    out.append(f"const int kCloneDim = {dim};")
    out.append("const float kCloneDistCoef = 1.5f;")
    out.append("const float kCloneRewardCoef = 0.75f;")
    out.append("const bool kCloneUseCumulative = true;")
    out.append(f"const std::vector<float> kCloneObs = {fmt_floats(obs.numpy())};")
    out.append(f"const std::vector<float> kCloneCumRewards = {fmt_floats(cum_rewards.numpy())};")
    out.append(f"const std::vector<float> kCloneStepRewards = {fmt_floats(step_rewards.numpy())};")
    out.append(f"const std::vector<uint8_t> kCloneAlive = {fmt_ints(alive.numpy().astype(int))};")
    out.append(
        f"const std::vector<int32_t> kCloneFitCompanions = {fmt_ints(recorded_companions[0].numpy())};"
    )
    out.append(
        f"const std::vector<int32_t> kCloneCloneCompanions = {fmt_ints(recorded_companions[1].numpy())};"
    )
    out.append(f"const std::vector<float> kCloneUniforms = {fmt_floats(recorded_uniforms[0].numpy())};")
    out.append(f"const std::vector<float> kCloneExpectedVr = {fmt_floats(vr.numpy())};")
    out.append(f"const std::vector<float> kCloneExpectedProbs = {fmt_floats(probs.numpy())};")
    out.append(
        f"const std::vector<uint8_t> kCloneExpectedWillClone = {fmt_ints(will_clone.numpy().astype(int))};"
    )


def gen_full_run_case(out: list[str]) -> None:
    """Record a 5-iteration FractalGas run on the mock env with n_elite=2."""
    n, iters = 8, 5
    torch.manual_seed(99)
    np.random.seed(99)

    recorded_companions: list[np.ndarray] = []
    recorded_uniforms: list[np.ndarray] = []
    recorded_actions: list[np.ndarray] = []
    recorded_dts: list[np.ndarray] = []

    original_compas = cloning_mod.random_alive_compas
    original_rand_like = cloning_mod.torch.rand_like

    def recording_compas(oobs, ref):
        compas = original_compas(oobs, ref)
        recorded_companions.append(compas.numpy().copy())
        return compas

    def recording_rand_like(x):
        u = original_rand_like(x)
        recorded_uniforms.append(u.numpy().copy())
        return u

    def action_sampler(count: int) -> np.ndarray:
        actions = np.random.randint(0, 4, size=count)
        recorded_actions.append(actions.copy())
        return actions

    class MockFractalGas(FractalGas):
        def _init_actions(self):
            return np.zeros(self.N, dtype=int)

        def _render_walker_frame(self, state):
            return np.zeros((1, 1, 3), dtype=np.uint8)

    env = MockEnv()
    gas = MockFractalGas(
        env,
        N=n,
        dist_coef=1.0,
        reward_coef=1.0,
        use_cumulative_reward=False,
        dt_range=(1, 4),
        action_sampler=action_sampler,
        n_elite=2,
    )

    original_sample_dt = gas.kinetic_op.sample_dt

    def recording_sample_dt(count: int) -> np.ndarray:
        dt = original_sample_dt(count)
        recorded_dts.append(dt.copy())
        return dt

    gas.kinetic_op.sample_dt = recording_sample_dt

    cloning_mod.random_alive_compas = recording_compas
    cloning_mod.torch.rand_like = recording_rand_like
    try:
        state = gas.reset()
        iter_rewards, iter_vr, iter_will_clone, iter_max_reward = [], [], [], []
        for _ in range(iters):
            state, info = gas.step(state)
            iter_rewards.append(state.rewards.numpy().copy())
            iter_vr.append(state.virtual_rewards.numpy().copy())
            iter_will_clone.append(info["will_clone"].numpy().astype(int).copy())
            iter_max_reward.append(float(info["max_reward"]))
    finally:
        cloning_mod.random_alive_compas = original_compas
        cloning_mod.torch.rand_like = original_rand_like

    assert len(recorded_companions) == 2 * iters
    assert len(recorded_uniforms) == iters
    assert len(recorded_actions) == iters
    assert len(recorded_dts) == iters

    fit_companions = [recorded_companions[2 * i] for i in range(iters)]
    clone_companions = [recorded_companions[2 * i + 1] for i in range(iters)]

    def emit_2d(name: str, rows, fmt) -> None:
        body = ", ".join(fmt(r) for r in rows)
        elem = "float" if fmt is fmt_floats else "int32_t"
        out.append(f"const std::vector<std::vector<{elem}>> {name} = {{{body}}};")

    out.append(f"const int kRunN = {n};")
    out.append(f"const int kRunIters = {iters};")
    out.append("const int kRunNElite = 2;")
    emit_2d("kRunFitCompanions", fit_companions, fmt_ints)
    emit_2d("kRunCloneCompanions", clone_companions, fmt_ints)
    emit_2d("kRunUniforms", recorded_uniforms, fmt_floats)
    emit_2d("kRunActions", recorded_actions, fmt_ints)
    emit_2d("kRunDts", recorded_dts, fmt_ints)
    emit_2d("kRunExpectedRewards", iter_rewards, fmt_floats)
    emit_2d("kRunExpectedVr", iter_vr, fmt_floats)
    will_clone_rows = ", ".join(fmt_ints(r) for r in iter_will_clone)
    out.append(
        f"const std::vector<std::vector<int32_t>> kRunExpectedWillClone = {{{will_clone_rows}}};"
    )
    out.append(f"const std::vector<float> kRunExpectedMaxReward = {fmt_floats(iter_max_reward)};")
    out.append(
        "const std::vector<float> kRunFinalObservations = "
        + fmt_floats(state.observations.numpy())
        + ";"
    )


def main() -> None:
    out: list[str] = [
        "// AUTO-GENERATED by tests/fixtures/generate_fixtures.py - do not edit.",
        "// Recorded from the reference Python implementation in",
        "// src/fragile/fractalai (asymmetric_rescale, l2_norm,",
        "// FractalCloningOperator, FractalGas on the mock env).",
        "#ifndef FRACTAL_GAS_FIXTURES_GENERATED_HPP",
        "#define FRACTAL_GAS_FIXTURES_GENERATED_HPP",
        "",
        "#include <cstdint>",
        "#include <vector>",
        "",
        "namespace fixtures {",
        "",
    ]
    gen_asymmetric_rescale_cases(out)
    out.append("")
    gen_l2_case(out)
    out.append("")
    gen_cloning_case(out)
    out.append("")
    gen_full_run_case(out)
    out.append("")
    out.append("}  // namespace fixtures")
    out.append("")
    out.append("#endif  // FRACTAL_GAS_FIXTURES_GENERATED_HPP")
    OUT_PATH.write_text("\n".join(out))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
