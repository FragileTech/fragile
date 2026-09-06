"""Public batch ownership, restoration and planning contracts of the C++ engine."""

import gc
import itertools
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from fragile.fractalai.control import ControlEngine, ControlEnv, ExplorationHistory
from fragile.fractalai.control.build import repository_root


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    if os.environ.get("FRAGILE_CONTROL_LIBRARY"):
        return Path(os.environ["FRAGILE_CONTROL_LIBRARY"])
    build = tmp_path_factory.mktemp("control-native")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(repository_root() / "fractal-gas-web"),
            "-B",
            str(build),
            "-DFG_CONTROL_ONLY=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--target", "fg_control", "-j4"], check=True)
    return next((build / "control").glob("*fg_control.*so"))


@pytest.fixture
def scene():
    return {
        "size": [100, 100],
        "bodies": [{"position": [50, 50], "controlled": True, "velocity": [1, 0], "drag": 0}],
    }


def test_lossless_snapshots_and_simultaneous_gather(scene, library):
    with ControlEngine(scene, worlds=4, library=library) as engine:
        state = engine.get_states()
        state.kinematics[:, 0, 0] = [10, 20, 30, 40]
        engine.set_states(state)
        engine.gather_states([1, 0, 0, 2])
        np.testing.assert_array_equal(engine.get_states().kinematics[:, 0, 0], [20, 10, 10, 30])
        snapshot = engine.serialize_states()
        assert len(snapshot) == 32 + 4 * engine.words * 4
        actions = np.array([[1, 0], [0, 1], [0.2, -1], [0, 0]], np.float32)
        engine.step_batch(actions, [1, 2, 3, 4])
        expected = engine.serialize_states()
        engine.deserialize_states(snapshot)
        engine.step_batch(actions, [1, 2, 3, 4])
        assert engine.serialize_states() == expected
        assert engine.transition_results().shape == (4, 4)
        np.testing.assert_array_equal(engine.transition_results()[:, 1], [1, 2, 3, 4])
        with pytest.raises(ValueError, match="checksum"):
            engine.deserialize_states(snapshot[:-1] + bytes([snapshot[-1] ^ 1]))
        assert engine.serialize_states() == expected


def test_borrowed_buffer_outlives_engine_and_writes_detach(scene, library):
    engine = ControlEngine(scene, library=library)
    state = engine.get_states(copy=False)
    original = state.data.copy()
    assert not state.data.flags.writeable
    for _ in range(4):
        engine.step_batch([1, 0], 6)
    engine.reset(55)
    engine.close()
    del engine
    gc.collect()
    np.testing.assert_array_equal(state.data, original)
    np.testing.assert_array_equal(state.kinematics[0, 0, :2], [50, 50])


@pytest.mark.parametrize("threads", [4, 64])
def test_thread_count_preserves_wave_and_environment(scene, library, threads):
    with (
        ControlEngine(scene, threads=1, library=library) as a,
        ControlEngine(scene, threads=threads, library=library) as b,
    ):
        for engine in (a, b):
            action, _ = engine.plan(walkers=32, horizon=8, elites=2, recording=1, seed=9)
            engine.step_batch(action, 6)
        assert a.serialize_states() == b.serialize_states()
        np.testing.assert_array_equal(a.exploration_tree().metadata, b.exploration_tree().metadata)
        np.testing.assert_array_equal(a.exploration_tree().values, b.exploration_tree().values)


def test_recorded_tree_replays_through_public_api(scene, library, tmp_path):
    with ControlEngine(scene, library=library) as engine:
        engine.plan(walkers=24, horizon=5, elites=2, recording=1)
        history = engine.exploration_tree()
        leaf = int(history.metadata[-1, 0])
        engine.replay_node(leaf)
        expected = engine.serialize_states()
        graph = history.to_networkx()
        path = [leaf]
        while list(graph.predecessors(path[-1])):
            path.append(next(graph.predecessors(path[-1])))
        path.reverse()
        engine.deserialize_states(history.root_snapshot)
        for parent, child in itertools.pairwise(path):
            edge = graph.edges[parent, child]
            engine.step_batch(edge["actions"], edge["dt"])
        assert engine.serialize_states() == expected
        file = tmp_path / "tree.npz"
        history.save(file)
        restored = ExplorationHistory.load(file)
        assert restored.root_snapshot == history.root_snapshot
        np.testing.assert_array_equal(restored.values, history.values)


def test_outputs_and_invalid_inputs(scene, library):
    with ControlEngine(scene, library=library) as engine:
        out = np.empty((1, engine.stride * 4), dtype=np.uint8)
        assert engine.get_states(out=out).data is out
        obs = np.empty((1, engine.observation_dim), np.float32)
        assert engine.observations(obs) is obs
        before = engine.serialize_states()
        for actions, duration in [([np.nan, 0], 1), ([0, 0], -1), ([0, 0], 1.5)]:
            with pytest.raises(ValueError):
                engine.step_batch(actions, duration)
            assert engine.serialize_states() == before
        with pytest.raises(ValueError):
            engine.gather_states([-1])
        other_scene = json.loads(json.dumps(scene))
        other_scene["bodies"][0]["mass"] = 2
        with ControlEngine(other_scene, library=library) as other:
            with pytest.raises(ValueError, match="different"):
                engine.set_states(other.get_states())
        engine.plan(walkers=8, horizon=1, recording=0)
        assert engine.exploration_tree().metadata.shape == (0, 5)
        engine.step_batch([1, 0], 6)
        engine.reset()
        assert engine.metrics()["reward"] == 0
        assert engine.metrics()["frames"] == 0
        assert engine.metrics()["tick"] == 0


@pytest.mark.parametrize(
    "name",
    [
        entry["id"]
        for entry in json.loads(
            (repository_root() / "fractal-gas-web/web/lab/scenario-catalog.json").read_text()
        )
    ],
)
def test_every_preset_compiles_and_plans(name, library):
    path = repository_root() / "fractal-gas-web/web/lab/scenarios" / f"{name}.json"
    with ControlEngine(path, threads=2, library=library) as engine:
        action, metrics = engine.plan(walkers=16, horizon=3, recording=1)
        assert metrics["iterations"] == 3
        assert np.isfinite(action).all()
        engine.step_batch(action, 6)
        assert engine.observations().shape == (1, engine.observation_dim)


def test_existing_python_gas_accepts_control_environment(scene, library):
    from fragile.fractalai.robots.robotic_gas import RoboticFractalGas

    with ControlEnv(scene, threads=2, library=library) as env:
        gas = RoboticFractalGas(env=env, N=12, seed=7, record_frames=False, n_elite=2)
        state = gas.reset()
        for _ in range(3):
            state, _ = gas.step(state)
        assert state.observations.shape[0] == 12
        assert state.actions.shape == (12, 2)
        assert np.isfinite(state.observations.cpu().numpy()).all()


def test_native_wasm_state_format_and_short_physics_agree(scene, library):
    root = repository_root()
    if not (root / "fractal-gas-web/web/lab/engine/control.wasm").exists():
        pytest.skip("Build the WebAssembly module to run cross-backend validation")
    program = """
import {loadNative, NativeEngine} from './fractal-gas-web/web/lab/native.js';
const engine = new NativeEngine(await loadNative(), SCENE);
const initial = Buffer.from(engine.snapshot()).toString('base64');
engine.step(new Float32Array([0.5, 0.2]), 12);
console.log(JSON.stringify({initial, state: Array.from(engine.states().slice(8,14))}));
engine.dispose();
""".replace("SCENE", json.dumps(scene))
    output = subprocess.run(
        ["node", "--input-type=module", "-e", program],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    wasm = json.loads(output.stdout)
    import base64

    with ControlEngine(scene, library=library) as engine:
        assert engine.serialize_states() == base64.b64decode(wasm["initial"])
        engine.step_batch([0.5, 0.2], 12)
        np.testing.assert_allclose(engine.get_states().kinematics[0, 0], wasm["state"], atol=1e-5)


def test_variable_actions_checkpoint_restores_search_and_rng(scene, library):
    scene["bodies"][0]["actuator"] = {"kind": "kart"}
    with ControlEngine(scene, library=library) as engine:
        assert engine.action_dim == 3
        assert [c["name"] for c in engine.channels] == ["throttle", "steering", "brake"]
        np.testing.assert_array_equal(engine.action_low, [-1, -1, 0])
        engine.begin_plan(walkers=16, horizon=6, elites=2, recording=2, frames=3, seed=137)
        assert not engine.advance_plan()
        saved = engine.checkpoint()
        while not engine.advance_plan():
            pass
        expected = engine.selected_action()
        tree = engine.exploration_tree()
        assert tree.pose_dim == 2
        assert tree.values.shape[1] == 8
        engine.restore_checkpoint(saved)
        while not engine.advance_plan():
            pass
        np.testing.assert_array_equal(engine.selected_action(), expected)
        restored = engine.exploration_tree()
        np.testing.assert_array_equal(restored.metadata, tree.metadata)
        np.testing.assert_array_equal(restored.values, tree.values)
        before = engine.checkpoint()
        with pytest.raises(ValueError, match="checksum"):
            engine.restore_checkpoint(saved[:-1] + bytes([saved[-1] ^ 1]))
        assert engine.checkpoint() == before


def test_broadcast_profile_and_nonmutating_inspection(scene, library):
    with (
        ControlEngine(scene, library=library) as root,
        ControlEngine(scene, worlds=8, library=library) as batch,
    ):
        root.step_batch([1, 0.2], 3)
        batch.broadcast_snapshot(root.serialize_states())
        for row in batch.get_states().data:
            np.testing.assert_array_equal(row, root.get_states().data[0])
        state = batch.serialize_states()
        vectors = batch.inspect([1, 0])
        assert vectors.shape[1] == 8
        assert vectors[0, 7] > 0
        assert batch.serialize_states() == state
        batch.step_batch(np.zeros((8, 2), np.float32), 2)
        out = batch.get_states()
        batch.set_states(out)
        batch.gather_states(np.arange(8)[::-1])
        profile = batch.profile(reset=True)
        assert profile["world_frames"] == 16
        assert profile["get_bytes"] >= out.data.nbytes
        assert profile["set_bytes"] == out.data.nbytes
        assert profile["gather_bytes"] == out.data.nbytes
        assert batch.profile()["world_frames"] == 0
