"""Compatibility adapter for the existing Python FractalGas/PlanningFractalGas.

The packed ControlEngine API avoids this adapter's per-walker Python objects.
Use this adapter when composing existing Torch gas operators or tree analysis.
"""

from dataclasses import dataclass

import numpy as np

from fragile.fractalai.control.engine import BatchState, ControlEngine


@dataclass
class ControlState:
    """One complete world row and its observation for Python gas compatibility."""

    data: np.ndarray
    observation: np.ndarray
    fingerprint: int
    bodies: int

    def copy(self):
        return ControlState(
            self.data.copy(), self.observation.copy(), self.fingerprint, self.bodies
        )


class _ActionSpace:
    def __init__(self, low, high, seed):
        dimensions = len(low)
        self.shape = (dimensions,)
        self.dtype = np.float32
        self.low = self.minimum = np.asarray(low, np.float32)
        self.high = self.maximum = np.asarray(high, np.float32)
        self.rng = np.random.default_rng(seed)

    def sample(self):
        return self.rng.uniform(self.low, self.high).astype(np.float32)


class ControlEnv:
    """Expose custom control physics to existing Python planning algorithms.

    Physics still steps as one parallel native batch. RGB rendering is supplied
    by the browser lab; use ``record_frames=False`` in Python gas instances.
    """

    def __init__(self, scene, *, threads=1, seed=7, library=None):
        self.engine = ControlEngine(scene, threads=threads, seed=seed, library=library)
        self.scene = self.engine.scene
        self._options = {"threads": threads, "seed": seed, "library": library}
        self._batch = None
        self.action_space = _ActionSpace(self.engine.action_low, self.engine.action_high, seed)
        self.name = self.scene.get("name", "fractal-control")

    @staticmethod
    def _states(engine):
        batch, obs = engine.get_states(), engine.observations()
        states = np.empty(engine.worlds, dtype=object)
        for i in range(engine.worlds):
            states[i] = ControlState(batch.data[i], obs[i], batch.fingerprint, batch.bodies)
        return states, obs

    def reset(self, seed=None, return_state=True):
        self.engine.reset(self._options["seed"] if seed is None else seed)
        states, observations = self._states(self.engine)
        return (states[0], observations[0], {}) if return_state else (observations[0], {})

    def get_state(self):
        return self._states(self.engine)[0][0]

    def set_state(self, state):
        self.engine.set_states(BatchState(state.data[None], state.fingerprint, state.bodies))

    clone_state = get_state
    restore_state = set_state

    def step(self, action, state=None, dt=1, return_state=True):
        if state is not None:
            self.set_state(state)
        self.engine.step_batch(action, dt)
        states, obs = self._states(self.engine)
        result = self.engine.transition_results()[0]
        return (
            states[0] if return_state else None,
            obs[0],
            float(result[0]),
            bool(result[2]),
            False,
            {"frames": int(result[1]), "collisions": int(result[3])},
        )

    def step_batch(self, states, actions, dt=None, **kwargs):
        n = len(states)
        if n == 0:
            return (
                np.empty(0, object),
                np.empty((0, self.engine.observation_dim), np.float32),
                np.empty(0, np.float32),
                np.empty(0, bool),
                np.empty(0, bool),
                [],
            )
        if self._batch is None or self._batch.worlds != n:
            if self._batch is not None:
                self._batch.close()
            self._batch = ControlEngine(self.scene, worlds=n, **self._options)
        if any(state.fingerprint != self.engine.fingerprint for state in states):
            msg = "State belongs to another scene"
            raise ValueError(msg)
        self._batch.set_states(
            BatchState(
                np.stack([s.data for s in states]), self.engine.fingerprint, self.engine.bodies
            )
        )
        self._batch.step_batch(actions, 1 if dt is None else dt)
        new_states, observations = self._states(self._batch)
        results = self._batch.transition_results()
        infos = [{"frames": int(r[1]), "collisions": int(r[3])} for r in results]
        return (
            new_states,
            observations,
            results[:, 0],
            results[:, 2].astype(bool),
            np.zeros(n, bool),
            infos,
        )

    def close(self):
        self.engine.close()
        if self._batch is not None:
            self._batch.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
