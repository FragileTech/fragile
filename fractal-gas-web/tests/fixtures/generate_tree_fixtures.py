"""Generate C++ test fixtures for the tree ("Graph") algorithm from the
HISTORICAL reference implementation (git commit cb9f3296, vendored verbatim
under reference_cb9f3296/).

Runs the real FractalTree / MontezumaTree on mock environments while
recording every RNG draw (companion indices of the fitness and clone phases,
the clone-decision uniforms, actions and dt of the stepped walkers) and the
per-iteration state (population size, parents, leaf/oobs/will_clone masks,
cumulative and virtual rewards, best index, visit-count reward), and emits
``fixtures_tree_generated.hpp`` so tests/test_fractal_tree.cpp can replay the
exact same computation through src/fractal_tree.cpp.

Run from the repo root with the project venv:

    uv run python fractal-gas-web/tests/fixtures/generate_tree_fixtures.py

The MockEnv here is the Python twin of tests/mock_env.hpp (identical
dynamics, float32 throughout); VisitMockEnv is the twin of
tests/visit_mock_env.hpp. Keep them in sync.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# Resolve `fragile` to the vendored historical package, not the installed one.
REFERENCE_DIR = Path(__file__).parent / "reference_cb9f3296"
sys.path.insert(0, str(REFERENCE_DIR))
for name in [m for m in sys.modules if m == "fragile" or m.startswith("fragile.")]:
    del sys.modules[name]

import fragile.core as core  # noqa: E402
import fragile.fractalai as fai  # noqa: E402
import fragile.videogames as videogames  # noqa: E402
from fragile.actions import UniformDtSampler  # noqa: E402

assert Path(core.__file__).resolve().is_relative_to(REFERENCE_DIR.resolve()), core.__file__

OUT_PATH = Path(__file__).parent / "fixtures_tree_generated.hpp"

DONE_THRESHOLD = 20.0  # keep in sync with mock_env.hpp
VISIT_W, VISIT_H, VISIT_ROOMS = 160, 160, 24  # keep in sync with visit_mock_env.hpp
VISIT_DONE_X = 150


def _float_literal(v: float) -> str:
    text = f"{float(v):.9g}"
    if not any(c in text for c in ".einf"):
        text += ".0"
    return text + "f"


def fmt_floats(values) -> str:
    return "{" + ", ".join(_float_literal(v) for v in np.asarray(values).ravel()) + "}"


def fmt_ints(values) -> str:
    return "{" + ", ".join(str(int(v)) for v in np.asarray(values).ravel()) + "}"


def emit_2d(out: list[str], name: str, rows, fmt, elem: str) -> None:
    body = ", ".join(fmt(r) for r in rows)
    out.append(f"const std::vector<std::vector<{elem}>> {name} = {{{body}}};")


# --- mock environments ---------------------------------------------------------


class MockEnv:
    """Python twin of tests/mock_env.hpp (float32 arithmetic throughout)."""

    observation_space = SimpleNamespace(shape=(3,), dtype=np.float32)
    action_space = SimpleNamespace(shape=(), dtype=np.int64)
    img_shape = (1, 1, 3)

    def reset(self):
        state = np.zeros(3, dtype=np.float32)
        return state, state.copy(), {}

    def sample_action(self):
        raise RuntimeError("actions must come from the recorded sampler")

    def get_image(self):
        return np.zeros(self.img_shape, dtype=np.uint8)

    def set_state(self, state):
        pass

    def step_batch(self, states, actions, dt):
        new_states, observs, rewards, dones = [], [], [], []
        for state, action, d in zip(states, actions, dt):
            s = np.asarray(state, dtype=np.float32).copy()
            a = np.float32(action)
            df = np.float32(d)
            s[0] = s[0] + (a + np.float32(1.0)) * df
            s[1] = s[1] + np.float32(0.5) * df
            s[2] = s[2] + np.float32(1.0)
            new_states.append(s)
            observs.append(s.copy())
            rewards.append(np.float32(0.1) * (a - np.float32(1.0)) * df + np.float32(0.01) * s[0])
            dones.append(bool(s[0] > DONE_THRESHOLD))
        return (
            np.array(new_states, dtype=np.float32),
            np.array(observs, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array([False] * len(dones), dtype=bool),
            [{"rgb": np.zeros(self.img_shape, dtype=np.uint8)} for _ in dones],
        )


class VisitMockEnv(MockEnv):
    """Python twin of tests/visit_mock_env.hpp: integer (x, y, room) walk on
    the Montezuma-sized visit grid, so MontezumaTree can count visits."""

    gym_env = SimpleNamespace(_x_repeat=1)

    def reset(self):
        state = np.array([80.0, 80.0, 1.0], dtype=np.float32)
        return state, state.copy(), {}

    def step_batch(self, states, actions, dt):
        new_states, observs, rewards, dones = [], [], [], []
        for state, action, d in zip(states, actions, dt):
            x, y, room = (int(v) for v in np.asarray(state))
            a = int(action)
            d = int(d)
            nx = (x + (a + 1) * d) % VISIT_W
            ny = (y + d) % VISIT_H
            nroom = (room + (1 if a == 3 else 0)) % VISIT_ROOMS
            s = np.array([nx, ny, nroom], dtype=np.float32)
            new_states.append(s)
            observs.append(s.copy())
            rewards.append(
                np.float32(0.1) * (np.float32(a) - np.float32(1.0)) * np.float32(d)
                + np.float32(0.01) * np.float32(nx)
            )
            dones.append(bool(nx > VISIT_DONE_X))
        return (
            np.array(new_states, dtype=np.float32),
            np.array(observs, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array([False] * len(dones), dtype=bool),
            [{"rgb": np.zeros(self.img_shape, dtype=np.uint8)} for _ in dones],
        )


# --- recording ---------------------------------------------------------------


class Recorder:
    def __init__(self):
        self.companions: list[np.ndarray] = []
        self.uniforms: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.dts: list[np.ndarray] = []
        self.stepped: list[int] = []
        self.other: list[np.ndarray] = []


def make_recording_class(base, rec: Recorder):
    class RecordingTree(base):
        def sample_actions(self, max_walkers=None):
            n = max_walkers if max_walkers is not None else self.n_walkers
            acts = np.random.randint(0, 4, size=n)
            rec.actions.append(acts.copy())
            return torch.tensor(acts, device=self.device, dtype=self.action_type)

        def sample_dt(self, n_walkers=None):
            dt = super().sample_dt(n_walkers)
            rec.dts.append(np.asarray(dt).copy())
            return dt

        def step_walkers(self):
            rec.stepped.append(int(self.will_clone.sum().item()))
            return super().step_walkers()

        def calculate_other_reward(self):
            other = super().calculate_other_reward()
            if isinstance(other, torch.Tensor):
                rec.other.append(other.detach().cpu().numpy().copy())
            return other

    return RecordingTree


def run_case(out, prefix, base_cls, env, iters, seed, with_visits, extra_kwargs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rec = Recorder()
    cls = make_recording_class(base_cls, rec)

    original_core_compas = core.random_alive_compas
    original_fai_compas = fai.random_alive_compas
    original_rand = torch.rand

    def recording_compas(oobs, ref):
        compas = original_core_compas(oobs, ref)
        rec.companions.append(compas.cpu().numpy().copy())
        return compas

    def recording_rand(*args, **kwargs):
        u = original_rand(*args, **kwargs)
        rec.uniforms.append(u.cpu().numpy().copy())
        return u

    core.random_alive_compas = recording_compas
    fai.random_alive_compas = recording_compas
    fai.torch.rand = recording_rand
    try:
        start, min_leafs, max_walkers = 6, 6, 64
        tree = cls(
            max_walkers=max_walkers,
            env=env,
            dt_sampler=UniformDtSampler(min_dt=1, max_dt=5),
            device="cpu",
            start_walkers=start,
            min_leafs=min_leafs,
            state_shape=(3,),
            state_dtype=np.float32,
            img_shape=(1, 1, 3),
            **extra_kwargs,
        )
        tree.reset()
        assert len(rec.actions) == 1 and len(rec.dts) == 1
        reset_actions, reset_dts = rec.actions.pop(), rec.dts.pop()
        assert not rec.companions and not rec.uniforms

        rows = {k: [] for k in ["n_before", "n_after", "leaves", "stepped", "best",
                                "parent", "leaf", "oobs", "will", "clone_ix",
                                "cum", "vr", "other"]}
        for it in range(iters):
            n_before = tree.n_walkers
            steps_before = len(rec.stepped)
            other_before = len(rec.other)
            tree.step_tree()
            if len(rec.stepped) == steps_before:
                rec.stepped.append(0)  # early return: no clone_data / step_walkers
            n_after = tree.n_walkers
            rows["n_before"].append(n_before)
            rows["n_after"].append(n_after)
            rows["leaves"].append(int(tree.is_leaf[:n_before].sum().item()))
            rows["stepped"].append(rec.stepped[-1])
            rows["best"].append(tree.best_ix)
            rows["parent"].append(tree.parent[:n_after].cpu().numpy().copy())
            rows["leaf"].append(tree.is_leaf[:n_after].cpu().numpy().astype(np.int64))
            rows["oobs"].append(tree.oobs[:n_after].cpu().numpy().astype(np.int64))
            rows["will"].append(tree.will_clone[:n_before].cpu().numpy().astype(np.int64))
            rows["clone_ix"].append(tree.clone_ix[:n_before].cpu().numpy().copy())
            rows["cum"].append(tree.cum_reward[:n_after].cpu().numpy().copy())
            rows["vr"].append(tree.virtual_reward[:n_after].cpu().numpy().copy())
            if with_visits:
                assert len(rec.other) == other_before + 1
                rows["other"].append(rec.other[-1])
        total_steps = tree.total_steps
        final_obs = tree.observ.cpu().numpy().copy()
        visits = tree.visits.copy() if with_visits else None
    finally:
        core.random_alive_compas = original_core_compas
        fai.random_alive_compas = original_fai_compas
        fai.torch.rand = original_rand

    assert len(rec.companions) == 2 * iters, len(rec.companions)
    assert len(rec.uniforms) == iters
    stepped_iters = sum(1 for k in rows["stepped"] if k > 0)
    assert len(rec.actions) == stepped_iters and len(rec.dts) == stepped_iters
    fit = [rec.companions[2 * i] for i in range(iters)]
    clone = [rec.companions[2 * i + 1] for i in range(iters)]
    # Actions / dt rows only exist for iterations that stepped; emit empty
    # rows otherwise so indices line up with iterations.
    actions_rows, dt_rows = [], []
    j = 0
    for k in rows["stepped"]:
        if k > 0:
            actions_rows.append(rec.actions[j])
            dt_rows.append(rec.dts[j])
            j += 1
        else:
            actions_rows.append(np.zeros(0, dtype=np.int64))
            dt_rows.append(np.zeros(0, dtype=np.int64))

    P = prefix
    out.append(f"const int {P}Start = {start};")
    out.append(f"const int {P}MinLeafs = {min_leafs};")
    out.append(f"const int {P}MaxWalkers = {max_walkers};")
    out.append(f"const int {P}Iters = {iters};")
    out.append(f"const std::vector<int32_t> {P}ResetActions = {fmt_ints(reset_actions)};")
    out.append(f"const std::vector<int32_t> {P}ResetDts = {fmt_ints(reset_dts)};")
    emit_2d(out, f"{P}FitCompanions", fit, fmt_ints, "int32_t")
    emit_2d(out, f"{P}CloneCompanions", clone, fmt_ints, "int32_t")
    emit_2d(out, f"{P}Uniforms", rec.uniforms, fmt_floats, "float")
    emit_2d(out, f"{P}Actions", actions_rows, fmt_ints, "int32_t")
    emit_2d(out, f"{P}Dts", dt_rows, fmt_ints, "int32_t")
    out.append(f"const std::vector<int32_t> {P}ExpectedNBefore = {fmt_ints(rows['n_before'])};")
    out.append(f"const std::vector<int32_t> {P}ExpectedNAfter = {fmt_ints(rows['n_after'])};")
    out.append(f"const std::vector<int32_t> {P}ExpectedLeaves = {fmt_ints(rows['leaves'])};")
    out.append(f"const std::vector<int32_t> {P}ExpectedStepped = {fmt_ints(rows['stepped'])};")
    out.append(f"const std::vector<int32_t> {P}ExpectedBest = {fmt_ints(rows['best'])};")
    emit_2d(out, f"{P}ExpectedParent", rows["parent"], fmt_ints, "int32_t")
    emit_2d(out, f"{P}ExpectedLeaf", rows["leaf"], fmt_ints, "int32_t")
    emit_2d(out, f"{P}ExpectedOobs", rows["oobs"], fmt_ints, "int32_t")
    emit_2d(out, f"{P}ExpectedWillClone", rows["will"], fmt_ints, "int32_t")
    emit_2d(out, f"{P}ExpectedCloneIx", rows["clone_ix"], fmt_ints, "int32_t")
    emit_2d(out, f"{P}ExpectedCum", rows["cum"], fmt_floats, "float")
    emit_2d(out, f"{P}ExpectedVr", rows["vr"], fmt_floats, "float")
    out.append(f"const std::vector<float> {P}FinalObservations = {fmt_floats(final_obs)};")
    out.append(f"const int64_t {P}PyTotalSteps = {int(total_steps)};")
    if with_visits:
        emit_2d(out, f"{P}ExpectedOther", rows["other"], fmt_floats, "float")
        cells = np.argwhere(visits != 0)
        rooms = fmt_ints(cells[:, 0])
        ys = fmt_ints(cells[:, 1])
        xs = fmt_ints(cells[:, 2])
        vals = fmt_floats(visits[visits != 0])
        out.append(f"const std::vector<int32_t> {P}FinalCellRoom = {rooms};")
        out.append(f"const std::vector<int32_t> {P}FinalCellY = {ys};")
        out.append(f"const std::vector<int32_t> {P}FinalCellX = {xs};")
        out.append(f"const std::vector<float> {P}FinalCellValue = {vals};")
    print(f"{prefix}: n {rows['n_before'][0]} -> {rows['n_after'][-1]}, stepped {rows['stepped']}, "
          f"total_steps(py) {total_steps}, sum(stepped) {sum(rows['stepped'])}")


def main() -> None:
    out: list[str] = [
        "// AUTO-GENERATED by tests/fixtures/generate_tree_fixtures.py - do not edit.",
        "// Recorded from the HISTORICAL reference implementation of the tree",
        "// algorithm (git cb9f3296, vendored verbatim in reference_cb9f3296/):",
        "// FractalTree on the mock env, MontezumaTree on the visit mock env.",
        "#ifndef FRACTAL_GAS_FIXTURES_TREE_GENERATED_HPP",
        "#define FRACTAL_GAS_FIXTURES_TREE_GENERATED_HPP",
        "",
        "#include <cstdint>",
        "#include <vector>",
        "",
        "namespace fixtures {",
        "",
    ]
    run_case(out, "kTree", core.FractalTree, MockEnv(), iters=14, seed=2024,
             with_visits=False, extra_kwargs={})
    out.append("")
    run_case(out, "kTV", videogames.MontezumaTree, VisitMockEnv(), iters=14, seed=4048,
             with_visits=True,
             extra_kwargs={"count_visits": True, "erase_coef": 0.05, "agg_block_size": 5})
    out.append("")
    out.append("}  // namespace fixtures")
    out.append("")
    out.append("#endif  // FRACTAL_GAS_FIXTURES_TREE_GENERATED_HPP")
    OUT_PATH.write_text("\n".join(out))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
