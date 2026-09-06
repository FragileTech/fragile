"""Batched control physics and native Wave/FMC planning.

All physics runs in the same C++ library as the browser. ctypes releases the
GIL during native calls. A native engine must not be called concurrently;
use separate engines for independent Python callers.
"""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import numpy as np

from fragile.fractalai.control._paths import repository_root


_P = ct.c_void_p
_U = ct.c_uint32
_I = ct.c_int
_Z = ct.c_size_t


def _library(path: str | Path | None = None) -> ct.CDLL:
    if path is None:
        path = os.environ.get("FRAGILE_CONTROL_LIBRARY")
    if path is None:
        filename = (
            "fg_control.dll"
            if sys.platform == "win32"
            else ("libfg_control.dylib" if sys.platform == "darwin" else "libfg_control.so")
        )
        path = repository_root() / "fractal-gas-web/build-control-native/control" / filename
    if not Path(path).is_file():
        msg = "Build the engine with: uv run python -m fragile.fractalai.control.build"
        raise ImportError(msg)
    lib = ct.CDLL(str(path))
    signatures = {
        "error": (ct.c_char_p, []),
        "create": (_P, [ct.c_char_p, _I, _I]),
        "destroy": (None, [_P]),
        "reset": (_I, [_P, _U, _U]),
        "info": (_I, [_P, _I]),
        "action_bound": (ct.c_float, [_P, _I, _I]),
        "action_body": (_I, [_P, _I]),
        "action_name": (ct.c_char_p, [_P, _I]),
        "broadcast": (_I, [_P, _P, _Z]),
        "profile": (_P, [_P]),
        "profile_reset": (_I, [_P]),
        "inspect": (_I, [_P]),
        "inspection": (_P, [_P]),
        "checkpoint_size": (_Z, [_P]),
        "checkpoint_write": (_I, [_P, _P, _Z]),
        "checkpoint_restore": (_I, [_P, _P, _Z]),
        "hash_lo": (_U, [_P]),
        "hash_hi": (_U, [_P]),
        "actions": (_P, [_P]),
        "frames": (_P, [_P]),
        "states": (_P, [_P]),
        "metrics": (_P, [_P]),
        "results": (_P, [_P]),
        "step": (_I, [_P]),
        "get_states": (_I, [_P, _P, _Z]),
        "set_states": (_I, [_P, _P, _Z]),
        "gather": (_I, [_P, _P, _Z]),
        "snapshot_size": (_Z, [_P]),
        "serialize": (_I, [_P, _P, _Z]),
        "deserialize": (_I, [_P, _P, _Z]),
        "borrow": (_P, [_P]),
        "batch_data": (_P, [_P]),
        "release": (None, [_P]),
        "observe": (_I, [_P, _P, _Z]),
        "plan_begin": (_I, [_P, ct.c_char_p, _U]),
        "plan_advance": (_I, [_P]),
        "plan_action": (_P, [_P]),
        "wave_step": (_I, [_P]),
        "wave_states": (_P, [_P]),
        "tree_export": (_I, [_P]),
        "tree_meta": (_P, [_P]),
        "tree_values": (_P, [_P]),
        "tree_root": (_P, [_P]),
        "tree_root_size": (_Z, [_P]),
        "replay_node": (_I, [_P, _U]),
        "raycast": (ct.c_float, [_P] + [ct.c_float] * 5),
    }
    for name, (result, args) in signatures.items():
        fn = getattr(lib, f"fgc_{name}")
        fn.restype, fn.argtypes = result, args
    return lib


class _Lease:
    def __init__(self, lib: ct.CDLL, handle: int):
        self.lib, self.handle = lib, handle

    def __del__(self):
        if self.handle:
            self.lib.fgc_release(self.handle)
            self.handle = None


class _LeasedArray(np.ndarray):
    _lease: _Lease | None = None

    def __array_finalize__(self, source):
        self._lease = getattr(source, "_lease", None)


@dataclass
class BatchState:
    """Complete native state rows, with ownership and scene identity.

    ``data`` contains aligned uint8 rows. Borrowed rows are read-only and stay
    alive independently of the Engine. Subsequent engine writes detach leased
    storage; release long-lived views when no longer needed.
    """

    data: np.ndarray
    fingerprint: int
    bodies: int

    def copy(self) -> BatchState:
        return BatchState(np.array(self.data, copy=True), self.fingerprint, self.bodies)

    @property
    def kinematics(self) -> np.ndarray:
        """View with shape [worlds, bodies, 6]: x, y, vx, vy, angle, omega."""
        values = self.data.view(np.float32)[:, 8 : 8 + 6 * self.bodies]
        return values.reshape(len(self.data), 6, self.bodies).transpose(0, 2, 1)


@dataclass
class ExplorationHistory:
    """Compact transition tree; full world states are reconstructed from its root."""

    metadata: np.ndarray
    values: np.ndarray
    root_snapshot: bytes
    action_dim: int
    pose_dim: int | None = None

    def to_networkx(self):
        """Export compatible node/edge data for existing NetworkX analysis tools."""
        import networkx as nx

        graph = nx.DiGraph()
        for meta, value in zip(self.metadata, self.values):
            node, parent, depth, frames, flags = map(int, meta)
            graph.add_node(
                node,
                depth=depth,
                reward=float(value[0]),
                step_reward=float(value[1]),
                virtual_reward=float(value[2]),
                dead=bool(flags & 1),
                tethered=bool(flags & 2),
                positions=value[3 : 3 + (self.pose_dim or self.action_dim)].reshape(-1, 2).copy(),
            )
            if parent:
                graph.add_edge(parent, node, actions=value[-self.action_dim :].copy(), dt=frames)
        return graph

    def save(self, path: str | Path) -> None:
        """Write a lossless compressed NumPy archive."""
        np.savez_compressed(
            path,
            metadata=self.metadata,
            values=self.values,
            root_snapshot=np.frombuffer(self.root_snapshot, dtype=np.uint8),
            action_dim=self.action_dim,
            pose_dim=self.pose_dim or self.action_dim,
        )

    @classmethod
    def load(cls, path: str | Path) -> ExplorationHistory:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["metadata"],
                data["values"],
                data["root_snapshot"].tobytes(),
                int(data["action_dim"]),
                int(data["pose_dim"]) if "pose_dim" in data else int(data["action_dim"]),
            )


class ControlEngine:
    """Custom C++ physics with efficient batches, snapshots, Wave and FMC.

    The primary interface accepts contiguous float32 actions [worlds, action_dim].
    Scene geometry is compiled once. All get/set/step APIs operate on complete
    mutable world state, including environmental RNG and task progress.
    """

    metric_names = (
        "reward",
        "frames",
        "collisions",
        "dead_worlds",
        "tick",
        "deliveries",
        "pickups",
        "gates",
        "iterations",
        "dead_ratio",
        "clone_ratio",
        "mean_reward",
        "max_reward",
        "tree_nodes",
        "pruned",
        "planning_ms",
    )

    def __init__(
        self,
        scene: dict | str | Path,
        worlds: int = 1,
        threads: int = 1,
        seed: int = 7,
        library: str | Path | None = None,
    ):
        if isinstance(scene, Path) or (
            isinstance(scene, str) and not scene.lstrip().startswith("{")
        ):
            scene = json.loads(Path(scene).read_text(encoding="utf-8"))
        elif isinstance(scene, str):
            scene = json.loads(scene)
        self.scene = json.loads(json.dumps(scene))
        self._lib = _library(library)
        self._handle = self._lib.fgc_create(json.dumps(self.scene).encode(), worlds, threads)
        if not self._handle:
            raise ValueError(self._lib.fgc_error().decode())
        self.worlds = worlds
        info = [self._lib.fgc_info(self._handle, i) for i in range(15)]
        self.bodies, self.controlled, self.stride, self.words = info[1:5]
        self.observation_dim = info[11]
        self.action_dim = info[12]
        self.channels = [
            {
                "body": self._lib.fgc_action_body(self._handle, i),
                "name": self._lib.fgc_action_name(self._handle, i).decode(),
                "low": self._lib.fgc_action_bound(self._handle, i, 0),
                "high": self._lib.fgc_action_bound(self._handle, i, 1),
            }
            for i in range(self.action_dim)
        ]
        self.action_low = np.array([c["low"] for c in self.channels], dtype=np.float32)
        self.action_high = np.array([c["high"] for c in self.channels], dtype=np.float32)
        self.fingerprint = self._lib.fgc_hash_lo(self._handle) | (
            self._lib.fgc_hash_hi(self._handle) << 32
        )
        self.reset(seed)

    def _check(self, code: int) -> int:
        if code < 0:
            raise ValueError(self._lib.fgc_error().decode())
        return code

    def _live(self):
        if not self._handle:
            msg = "Control engine is closed"
            raise RuntimeError(msg)
        return self._handle

    @staticmethod
    def _array(pointer: int, dtype, shape) -> np.ndarray:
        element = np.ctypeslib.as_ctypes_type(np.dtype(dtype))
        return np.ctypeslib.as_array(ct.cast(pointer, ct.POINTER(element)), shape=shape)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.fgc_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()

    def neutral_action(self) -> np.ndarray:
        """Return a valid default batch: zero projected into each channel interval."""
        return np.broadcast_to(
            np.clip(np.zeros(self.action_dim, np.float32), self.action_low, self.action_high),
            (self.worlds, self.action_dim),
        ).copy()

    def reset(self, seed: int = 7) -> BatchState:
        self._check(
            self._lib.fgc_reset(self._live(), seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF)
        )
        return self.get_states()

    def get_states(self, *, copy: bool = True, out: np.ndarray | None = None) -> BatchState:
        shape = (self.worlds, self.stride * 4)
        if out is not None or copy:
            data = np.empty(shape, dtype=np.uint8) if out is None else out
            if (
                data.shape != shape
                or data.dtype != np.uint8
                or not data.flags.c_contiguous
                or not data.flags.writeable
            ):
                msg = f"State output must be contiguous uint8 with shape {shape}"
                raise ValueError(msg)
            self._check(self._lib.fgc_get_states(self._live(), data.ctypes.data, data.nbytes))
        else:
            lease = _Lease(self._lib, self._lib.fgc_borrow(self._live()))
            data = self._array(self._lib.fgc_batch_data(lease.handle), np.uint8, shape).view(
                _LeasedArray
            )
            data._lease = lease
            data.flags.writeable = False
        return BatchState(data, self.fingerprint, self.bodies)

    def set_states(self, state: BatchState) -> None:
        if state.fingerprint != self.fingerprint:
            msg = "State belongs to a different compiled scene"
            raise ValueError(msg)
        data = np.ascontiguousarray(state.data)
        if data.dtype != np.uint8 or data.shape != (self.worlds, self.stride * 4):
            msg = "State batch shape or dtype mismatch"
            raise ValueError(msg)
        self._check(self._lib.fgc_set_states(self._live(), data.ctypes.data, data.nbytes))

    def gather_states(self, indices) -> None:
        indices = np.asarray(indices)
        if indices.shape != (self.worlds,) or indices.dtype.kind not in "iu":
            msg = "Provide one integer source index per destination world"
            raise ValueError(msg)
        if np.any(indices < 0) or np.any(indices >= self.worlds):
            msg = "State index outside batch"
            raise ValueError(msg)
        indices = np.ascontiguousarray(indices, dtype=np.int32)
        self._check(self._lib.fgc_gather(self._live(), indices.ctypes.data, len(indices)))

    def serialize_states(self) -> bytes:
        size = self._lib.fgc_snapshot_size(self._live())
        data = np.empty(size, dtype=np.uint8)
        self._check(self._lib.fgc_serialize(self._live(), data.ctypes.data, size))
        return data.tobytes()

    def deserialize_states(self, data: bytes) -> None:
        buffer = np.frombuffer(data, dtype=np.uint8)
        self._check(self._lib.fgc_deserialize(self._live(), buffer.ctypes.data, len(buffer)))

    def broadcast_snapshot(self, data: bytes) -> None:
        """Restore one scene-compatible world into every row of this batch."""
        source = np.frombuffer(data, dtype=np.uint8)
        self._check(self._lib.fgc_broadcast(self._live(), source.ctypes.data, source.nbytes))

    def profile(self, *, reset: bool = False) -> dict[str, float]:
        """Read accumulated counters, optionally resetting afterward.

        ``world_frames`` counts completed batch and FMC/Wave simulation frames.
        ``step_ms`` times direct batch steps; ``plan_ms`` times planner advances,
        including their sampling and cloning work.
        """
        names = (
            "step_ms",
            "world_frames",
            "get_ms",
            "get_bytes",
            "set_ms",
            "set_bytes",
            "gather_ms",
            "gather_bytes",
            "plan_ms",
            "plan_advances",
            "tracked_buffer_bytes",
            "snapshot_bytes",
        )
        values = self._array(self._lib.fgc_profile(self._live()), np.float64, (12,)).copy()
        if reset:
            self._check(self._lib.fgc_profile_reset(self._live()))
        return dict(zip(names, map(float, values)))

    def inspect(self, actions=None) -> np.ndarray:
        """Read first-world force/contact/tether vectors without advancing state.

        Columns: kind (0 force, 1 contact, 2 tether), body A/B, x/y,
        vector x/y, magnitude. Contacts describe current proximity, not past impulses.
        """
        if actions is not None:
            actions = np.asarray(actions, dtype=np.float32)
            if actions.shape != (self.action_dim,) or not np.all(np.isfinite(actions)):
                msg = "Inspection requires one finite action vector"
                raise ValueError(msg)
            self._array(
                self._lib.fgc_actions(self._live()), np.float32, (self.worlds, self.action_dim)
            )[0] = actions
        count = self._check(self._lib.fgc_inspect(self._live()))
        if not count:
            return np.empty((0, 8), np.float32)
        return self._array(self._lib.fgc_inspection(self._live()), np.float32, (count, 8)).copy()

    def step_batch(self, actions, dt=1) -> dict[str, float]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape == (self.action_dim,) and self.worlds == 1:
            actions = actions[None]
        if actions.shape != (self.worlds, self.action_dim):
            msg = f"Actions must have shape {(self.worlds, self.action_dim)}"
            raise ValueError(msg)
        frame_values = np.broadcast_to(np.asarray(dt), (self.worlds,))
        if not np.all(np.isfinite(frame_values)) or np.any(frame_values != np.floor(frame_values)):
            msg = "Action durations must be integer frame counts"
            raise ValueError(msg)
        if np.any(frame_values < 0) or np.any(frame_values > 4096):
            msg = "Action durations must be between 0 and 4096"
            raise ValueError(msg)
        self._array(self._lib.fgc_actions(self._live()), np.float32, actions.shape)[:] = actions
        self._array(self._lib.fgc_frames(self._live()), np.int32, (self.worlds,))[:] = frame_values
        self._check(self._lib.fgc_step(self._live()))
        return self.metrics()

    def observations(self, out: np.ndarray | None = None) -> np.ndarray:
        shape = (self.worlds, self.observation_dim)
        out = np.empty(shape, dtype=np.float32) if out is None else out
        if (
            out.shape != shape
            or out.dtype != np.float32
            or not out.flags.c_contiguous
            or not out.flags.writeable
        ):
            msg = "Observation output must be contiguous float32 with the documented shape"
            raise ValueError(msg)
        self._check(self._lib.fgc_observe(self._live(), out.ctypes.data, out.size))
        return out

    def metrics(self) -> dict[str, float]:
        values = self._array(self._lib.fgc_metrics(self._live()), np.float32, (16,))
        return dict(zip(self.metric_names, map(float, values)))

    def transition_results(self, out: np.ndarray | None = None) -> np.ndarray:
        """Return [worlds, 4]: reward, frames advanced, terminal flag, collisions."""
        shape = (self.worlds, 4)
        out = np.empty(shape, dtype=np.float32) if out is None else out
        if out.shape != shape or out.dtype != np.float32 or not out.flags.writeable:
            msg = "Transition output must be writable float32 [worlds, 4]"
            raise ValueError(msg)
        out[:] = self._array(self._lib.fgc_results(self._live()), np.float32, shape)
        return out

    def begin_plan(self, *, seed: int = 7, **settings) -> None:
        self._planner_settings = settings.copy()
        self._check(self._lib.fgc_plan_begin(self._live(), json.dumps(settings).encode(), seed))

    def advance_plan(self) -> bool:
        return bool(self._check(self._lib.fgc_plan_advance(self._live())))

    def selected_action(self) -> np.ndarray:
        pointer = self._lib.fgc_plan_action(self._live())
        if not pointer:
            raise ValueError(self._lib.fgc_error().decode())
        return self._array(pointer, np.float32, (self.action_dim,)).copy()

    def plan(self, **settings) -> tuple[np.ndarray, dict[str, float]]:
        self.begin_plan(**settings)
        while not self.advance_plan():
            pass
        return self.selected_action(), self.metrics()

    def wave_step(self) -> dict[str, float]:
        self._check(self._lib.fgc_wave_step(self._live()))
        return self.metrics()

    def checkpoint(self) -> bytes:
        """Save the world, active search population, elite bank, tree and planner RNG."""
        size = self._lib.fgc_checkpoint_size(self._live())
        if not size:
            self._check(-1)
        out = ct.create_string_buffer(size)
        self._check(self._lib.fgc_checkpoint_write(self._live(), out, size))
        return out.raw

    def restore_checkpoint(self, data: bytes) -> None:
        """Atomically restore a checkpoint produced by the same engine backend."""
        source = ct.create_string_buffer(data)
        self._check(self._lib.fgc_checkpoint_restore(self._live(), source, len(data)))

    def descriptor(self) -> dict:
        """Versioned algorithm–engine contract; state rows are opaque to controllers."""
        return {
            "version": 1,
            "state": {"words": self.words, "stride": self.stride},
            "channels": self.channels,
            "observation_dimension": self.observation_dim,
            "capabilities": ["batch-step", "gather", "snapshot", "planner-checkpoint"],
        }

    def exploration_tree(self) -> ExplorationHistory:
        count = self._check(self._lib.fgc_tree_export(self._live()))
        meta = (
            self._array(self._lib.fgc_tree_meta(self._live()), np.uint32, (count, 5)).copy()
            if count
            else np.empty((0, 5), np.uint32)
        )
        width = 3 + self.action_dim + 2 * self.controlled
        values = (
            self._array(self._lib.fgc_tree_values(self._live()), np.float32, (count, width)).copy()
            if count
            else np.empty((0, width), np.float32)
        )
        root_size = self._lib.fgc_tree_root_size(self._live())
        root = ct.string_at(self._lib.fgc_tree_root(self._live()), root_size) if root_size else b""
        return ExplorationHistory(meta, values, root, self.action_dim, 2 * self.controlled)

    def replay_node(self, node_id: int) -> BatchState:
        self._check(self._lib.fgc_replay_node(self._live(), node_id))
        return self.get_states()

    def raycast(self, origin, direction, distance: float = 1000) -> float:
        result = self._lib.fgc_raycast(self._live(), *origin, *direction, distance)
        if result < 0:
            raise ValueError(self._lib.fgc_error().decode())
        return float(result)
