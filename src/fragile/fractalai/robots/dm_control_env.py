"""DM Control environment wrapper with state cloning support.

Provides a lightweight wrapper around dm_control.suite that exposes
the same interface as AtariEnv, enabling state cloning / restoration
for tree-based planning algorithms such as Fractal Monte Carlo.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


@dataclass
class DMControlState:
    """Container for DM Control environment state."""

    physics_state: np.ndarray  # from env.physics.get_state()
    observation: np.ndarray  # flattened observation vector
    rgb_frame: np.ndarray | None = None
    body_zpos: np.ndarray | None = None  # z-positions of all bodies from xpos[:, 2]

    def copy(self) -> DMControlState:
        """Return a deep copy of this state."""
        return DMControlState(
            physics_state=self.physics_state.copy(),
            observation=self.observation.copy(),
            rgb_frame=self.rgb_frame.copy() if self.rgb_frame is not None else None,
            body_zpos=self.body_zpos.copy() if self.body_zpos is not None else None,
        )


class _DMControlActionSpace:
    """Adapter to provide gym-like action_space interface for DM Control."""

    def __init__(self, action_spec):
        self.shape = action_spec.shape
        self.minimum = action_spec.minimum
        self.maximum = action_spec.maximum
        self.dtype = np.float64

    def sample(self) -> np.ndarray:
        """Sample a random action uniformly within the action bounds."""
        return np.random.uniform(low=self.minimum, high=self.maximum, size=self.shape).astype(
            np.float64
        )


class DMControlEnv:
    """Lightweight wrapper around dm_control.suite with state cloning support.

    Mirrors the AtariEnv interface so that planning algorithms can treat
    both Atari and continuous-control environments interchangeably.

    Args:
        name: Environment name as "domain-task", e.g. "cartpole-balance".
        task_kwargs: Extra keyword arguments forwarded to ``suite.load``.
        render_width: Width in pixels for RGB rendering.
        render_height: Height in pixels for RGB rendering.
        include_rgb: Whether to include an RGB frame in cloned states.
    """

    def __init__(
        self,
        name: str,
        task_kwargs: dict | None = None,
        render_width: int = 480,
        render_height: int = 480,
        include_rgb: bool = True,
    ):
        parts = name.split("-", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Expected name in 'domain-task' format (e.g. 'cartpole-balance'), got '{name}'"
            )
        domain, task = parts[0], parts[1]

        from dm_control import suite  # lazy import

        self.env = suite.load(domain, task, task_kwargs=task_kwargs or {})

        action_spec = self.env.action_spec()
        self.action_spec_data = action_spec
        self.action_space = _DMControlActionSpace(action_spec)

        self.name = name
        self.render_width = render_width
        self.render_height = render_height
        self.include_rgb = include_rgb

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flatten_obs(self, time_step_or_obs) -> np.ndarray:
        """Flatten an observation dict into a single vector."""
        obs = time_step_or_obs
        if hasattr(obs, "observation"):
            obs = obs.observation
        parts = []
        for key in obs.keys():
            val = np.asarray(obs[key], dtype=np.float64).flatten()
            parts.append(val)
        return np.concatenate(parts)

    def _clone_state(self, observation: np.ndarray | None = None) -> DMControlState:
        """Snapshot the current environment state."""
        physics_state = self.env.physics.get_state().copy()
        if observation is None:
            observation = self._flatten_obs(self.env.task.get_observation(self.env.physics))
        rgb_frame = (
            self.env.physics.render(
                height=self.render_height, width=self.render_width, camera_id=0
            )
            if self.include_rgb
            else None
        )
        body_zpos = self.env.physics.data.xpos[:, 2].copy()
        return DMControlState(
            physics_state=physics_state,
            observation=observation,
            rgb_frame=rgb_frame,
            body_zpos=body_zpos,
        )

    def _restore_state(self, state: DMControlState):
        """Restore the environment to a previously cloned state."""
        with self.env.physics.reset_context():
            self.env.physics.set_state(state.physics_state)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> DMControlState:
        """Reset the environment and return the initial state."""
        time_step = self.env.reset()
        obs = self._flatten_obs(time_step)
        return self._clone_state(observation=obs)

    def step(
        self,
        action,
        state: DMControlState | None = None,
        dt: int = 1,
        return_state: bool = True,
    ):
        """Advance the environment by *dt* physics steps.

        Args:
            action: Action array compatible with the environment's action spec.
            state: If provided, restore this state before stepping.
            dt: Number of sub-steps (action repeats).
            return_state: Whether to clone and return the resulting state.

        Returns:
            Tuple of (new_state, observation, total_reward, done, truncated, info).
        """
        if state is not None:
            self._restore_state(state)

        import dm_env  # lazy import

        total_reward = 0.0
        done = False
        truncated = False
        info: dict = {}
        time_step = None

        for _ in range(dt):
            time_step = self.env.step(action)
            total_reward += time_step.reward or 0.0
            if time_step.step_type == dm_env.StepType.LAST:
                done = True
            if done:
                break

        obs = self._flatten_obs(time_step)
        new_state = self._clone_state(observation=obs) if return_state else None
        return new_state, obs, total_reward, done, truncated, info

    def step_batch(self, states, actions, dt=None, **kwargs):
        """Step a batch of (state, action) pairs sequentially.

        Args:
            states: Array-like of :class:`DMControlState` objects.
            actions: Array-like of action arrays.
            dt: Per-element action-repeat counts (defaults to 1).

        Returns:
            Tuple of (new_states, observations, rewards, dones, truncateds, infos).
        """
        N = len(states)
        if dt is None:
            dt = np.ones(N, dtype=int)

        new_states = np.empty(N, dtype=object)
        observations = []
        rewards = np.zeros(N, dtype=np.float32)
        dones = np.zeros(N, dtype=bool)
        truncateds = np.zeros(N, dtype=bool)
        infos = []

        for i in range(N):
            new_state, obs, reward, done, trunc, info = self.step(
                action=actions[i],
                state=states[i],
                dt=int(dt[i]),
                return_state=True,
            )
            new_states[i] = new_state
            observations.append(obs)
            rewards[i] = reward
            dones[i] = done
            truncateds[i] = trunc
            infos.append(info)

        observations = np.array(observations)
        return new_states, observations, rewards, dones, truncateds, infos

    def render(self) -> np.ndarray:
        """Render the current physics state as an RGB array."""
        return self.env.physics.render(
            height=self.render_height, width=self.render_width, camera_id=0
        )

    def close(self):
        """Close the underlying environment if supported."""
        if hasattr(self.env, "close"):
            self.env.close()

    # ------------------------------------------------------------------
    # Convenience methods matching AtariEnv interface
    # ------------------------------------------------------------------

    def clone_state(self) -> DMControlState:
        """Clone the current environment state."""
        return self._clone_state()

    def restore_state(self, state: DMControlState):
        """Restore the environment to a previously cloned state."""
        self._restore_state(state)

    def set_state(self, state: DMControlState):
        """Alias for :meth:`restore_state`."""
        self._restore_state(state)

    def get_state(self) -> DMControlState:
        """Return the current environment state."""
        return self._clone_state()


class VectorizedDMControlEnv:
    """Thread-parallel wrapper that distributes ``step_batch`` across workers.

    Each worker owns an independent :class:`DMControlEnv` instance.  MuJoCo's C
    code releases the GIL during physics stepping, so ``ThreadPoolExecutor``
    provides effective parallelism without serialization overhead.

    Single-walker methods (``reset``, ``step``, ``render``, …) are delegated to
    the primary (index-0) environment instance.

    Args:
        name: Environment name as "domain-task".
        n_workers: Number of parallel worker threads / env copies.
        **kwargs: Forwarded to each :class:`DMControlEnv` constructor.
    """

    def __init__(self, name: str, n_workers: int = 2, **kwargs):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self._envs = [DMControlEnv(name=name, **kwargs) for _ in range(n_workers)]
        self._executor = ThreadPoolExecutor(max_workers=n_workers)
        self._n_workers = n_workers

    # -- Proxied attributes from the primary env -------------------------

    @property
    def action_space(self):
        return self._envs[0].action_space

    @property
    def name(self):
        return self._envs[0].name

    @property
    def render_width(self):
        return self._envs[0].render_width

    @property
    def render_height(self):
        return self._envs[0].render_height

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def _normalize_env_indices(self, env_indices=None) -> np.ndarray:
        if env_indices is None:
            return np.arange(self._n_workers, dtype=int)
        indices = np.asarray(env_indices, dtype=int)
        if indices.ndim != 1:
            raise ValueError(f"env_indices must be 1D, got shape {indices.shape}")
        if len(indices) == 0:
            return indices
        if indices.min() < 0 or indices.max() >= self._n_workers:
            raise ValueError(
                f"env_indices must be within [0, {self._n_workers}), got {indices.tolist()}",
            )
        return indices

    # -- Parallel reset / direct stepping --------------------------------

    def reset_batch(self, env_indices=None, seeds=None) -> np.ndarray:
        """Reset selected workers in parallel and return flattened observations."""
        indices = self._normalize_env_indices(env_indices)
        if len(indices) == 0:
            return np.empty((0, 0), dtype=np.float32)
        if seeds is None:
            seeds = [None] * len(indices)
        elif len(seeds) != len(indices):
            raise ValueError(
                f"len(seeds)={len(seeds)} must match len(env_indices)={len(indices)}",
            )

        futures = [
            self._executor.submit(self._envs[idx].reset, seed=seed)
            for idx, seed in zip(indices, seeds)
        ]
        states = [future.result() for future in futures]
        obs_dim = states[0].observation.shape[0]
        observations = np.empty((len(indices), obs_dim), dtype=np.float32)
        for row, state in enumerate(states):
            observations[row] = state.observation.astype(np.float32, copy=False)
        return observations

    def step_actions_batch(self, actions, dt=None, env_indices=None):
        """Step selected workers from their internal states without cloning."""
        indices = self._normalize_env_indices(env_indices)
        N = len(indices)
        if len(actions) != N:
            raise ValueError(f"len(actions)={len(actions)} must match len(env_indices)={N}")
        if dt is None:
            dt = np.ones(N, dtype=int)
        elif len(dt) != N:
            raise ValueError(f"len(dt)={len(dt)} must match len(env_indices)={N}")
        if N == 0:
            empty = np.empty((0, 0), dtype=np.float32)
            return empty, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool), np.zeros(
                0,
                dtype=bool,
            ), []

        futures = [
            self._executor.submit(
                self._envs[idx].step,
                actions[row],
                None,
                int(dt[row]),
                False,
            )
            for row, idx in enumerate(indices)
        ]

        observations = []
        rewards = np.zeros(N, dtype=np.float32)
        dones = np.zeros(N, dtype=bool)
        truncateds = np.zeros(N, dtype=bool)
        infos: list = []
        for row, future in enumerate(futures):
            _state, obs, reward, done, trunc, info = future.result()
            observations.append(np.asarray(obs, dtype=np.float32))
            rewards[row] = reward
            dones[row] = done
            truncateds[row] = trunc
            infos.append(info)

        return np.stack(observations), rewards, dones, truncateds, infos

    # -- Parallel step_batch ---------------------------------------------

    def step_batch(self, states, actions, dt=None, **kwargs):
        """Step a batch of (state, action) pairs in parallel across workers.

        Splits the batch into ``n_workers`` chunks, each processed by its own
        ``DMControlEnv``.  Results are concatenated in the original order.
        """
        N = len(states)
        if dt is None:
            dt = np.ones(N, dtype=int)

        indices = np.arange(N)
        chunks = np.array_split(indices, self._n_workers)

        futures = []
        for env, chunk in zip(self._envs, chunks):
            if len(chunk) == 0:
                continue
            futures.append(
                self._executor.submit(
                    env.step_batch,
                    states[chunk],
                    actions[chunk],
                    dt[chunk],
                    **kwargs,
                )
            )

        # Collect results and reassemble in original order
        new_states = np.empty(N, dtype=object)
        rewards = np.zeros(N, dtype=np.float32)
        dones = np.zeros(N, dtype=bool)
        truncateds = np.zeros(N, dtype=bool)
        infos: list = [{}] * N
        obs_parts: list[tuple[np.ndarray, np.ndarray]] = []

        non_empty_chunks = [c for c in chunks if len(c) > 0]
        for future, chunk in zip(futures, non_empty_chunks):
            ns, obs, rew, dn, tr, inf = future.result()
            obs_parts.append((chunk, obs))
            for j, idx in enumerate(chunk):
                new_states[idx] = ns[j]
                rewards[idx] = rew[j]
                dones[idx] = dn[j]
                truncateds[idx] = tr[j]
                infos[idx] = inf[j]

        # Build observation array after we know the obs dimension
        obs_dim = obs_parts[0][1].shape[1]
        observations = np.empty((N, obs_dim), dtype=np.float64)
        for chunk, obs in obs_parts:
            observations[chunk] = obs

        return new_states, observations, rewards, dones, truncateds, infos

    # -- Delegated single-walker methods ---------------------------------

    def reset(self, seed: int | None = None) -> DMControlState:
        return self._envs[0].reset(seed=seed)

    def step(self, action, state=None, dt=1, return_state=True):
        return self._envs[0].step(action, state=state, dt=dt, return_state=return_state)

    def get_state(self) -> DMControlState:
        return self._envs[0].get_state()

    def clone_state(self) -> DMControlState:
        return self._envs[0].clone_state()

    def set_state(self, state: DMControlState):
        self._envs[0].set_state(state)

    def restore_state(self, state: DMControlState):
        self._envs[0].restore_state(state)

    def render(self) -> np.ndarray:
        return self._envs[0].render()

    def close(self):
        """Shut down the thread pool and close all environment instances."""
        self._executor.shutdown(wait=False)
        for env in self._envs:
            env.close()
