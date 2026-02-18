"""FractalGas base class and WalkerState dataclass.

Provides the shared algorithm skeleton for Atari and robotic fractal gas
implementations. Subclasses override a handful of hooks (action init,
frame rendering, non-tuple reset handling) to specialise behaviour.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


def _make_object_array(items) -> np.ndarray:
    """Create a 1D object array from *items* without collapsing same-shape sub-arrays.

    ``np.array(list_of_same_shape_arrays, dtype=object)`` creates a 2D object
    array, which breaks downstream code that expects ``arr[i]`` to return the
    original numpy array (not an object-dtype row).  This helper pre-allocates
    a 1D object array and fills it element-by-element.
    """
    n = len(items)
    arr = np.empty(n, dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def _clone_tensor(data: Tensor, companions: Tensor, will_clone: Tensor) -> Tensor:
    """Clone tensor data based on cloning decisions."""
    cloned = data.clone()
    cloned[will_clone] = data[companions[will_clone]]
    return cloned


@dataclass
class WalkerState:
    """Container for all walker data.

    Stores environment states, observations, rewards, and other walker attributes.
    Observations are stored as tensors for efficient distance computation.

    Attributes:
        states: Environment states [N], dtype=object
        observations: Observations as tensors [N, ...] (e.g., RAM [N, 128])
        rewards: Cumulative rewards [N]
        step_rewards: Last step rewards [N]
        dones: Terminal flags [N]
        truncated: Truncation flags [N]
        actions: Last actions [N]
        dt: Frame skip values [N]
        infos: Info dicts [N]
        virtual_rewards: Virtual rewards from fitness [N]
    """

    states: np.ndarray  # dtype=object
    observations: Tensor
    rewards: Tensor
    step_rewards: Tensor
    dones: Tensor
    truncated: Tensor
    actions: np.ndarray
    dt: np.ndarray
    infos: list
    virtual_rewards: Tensor | None = None

    @property
    def N(self) -> int:
        """Number of walkers."""
        return len(self.states)

    @property
    def alive(self) -> Tensor:
        """Boolean mask of alive walkers (not done and not truncated)."""
        return ~(self.dones | self.truncated)

    @property
    def device(self) -> str:
        """Device of tensor data."""
        return self.observations.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of tensor data."""
        return self.observations.dtype

    def inject(self, source: "WalkerState", positions: slice | np.ndarray) -> "WalkerState":
        """Overwrite walkers at `positions` with data from `source`.

        Returns a new WalkerState (defensive copy of affected fields).
        """
        states = self.states.copy()
        observations = self.observations.clone()
        rewards = self.rewards.clone()
        step_rewards = self.step_rewards.clone()
        dones = self.dones.clone()
        truncated = self.truncated.clone()
        actions = self.actions.copy()
        dt = self.dt.copy()
        infos = list(self.infos)

        # Numpy/list fields
        states[positions] = _make_object_array([self._copy_state(s) for s in source.states])
        actions[positions] = source.actions.copy()
        dt[positions] = source.dt.copy()
        idx = (
            list(range(*positions.indices(self.N))) if isinstance(positions, slice) else positions
        )
        for i, j in enumerate(idx):
            infos[j] = source.infos[i]

        # Tensor fields
        observations[positions] = source.observations.clone()
        rewards[positions] = source.rewards.clone()
        step_rewards[positions] = source.step_rewards.clone()
        dones[positions] = source.dones.clone()
        truncated[positions] = source.truncated.clone()

        vr = self.virtual_rewards.clone() if self.virtual_rewards is not None else None
        if vr is not None and source.virtual_rewards is not None:
            vr[positions] = source.virtual_rewards.clone()

        return WalkerState(
            states=states,
            observations=observations,
            rewards=rewards,
            step_rewards=step_rewards,
            dones=dones,
            truncated=truncated,
            actions=actions,
            dt=dt,
            infos=infos,
            virtual_rewards=vr,
        )

    @staticmethod
    def _copy_state(state):
        return state.copy() if hasattr(state, "copy") else state

    def clone(self, companions: Tensor, will_clone: Tensor) -> "WalkerState":
        """Create new WalkerState with cloned walker data.

        Args:
            companions: Companion indices [N]
            will_clone: Boolean mask [N], True for walkers that will clone

        Returns:
            New WalkerState with cloned data
        """
        # Clone numpy arrays (states, actions, dt)
        new_states = self.states.copy()
        new_states[will_clone.cpu().numpy()] = self.states[companions[will_clone].cpu().numpy()]

        new_actions = self.actions.copy()
        new_actions[will_clone.cpu().numpy()] = self.actions[companions[will_clone].cpu().numpy()]

        new_dt = self.dt.copy()
        new_dt[will_clone.cpu().numpy()] = self.dt[companions[will_clone].cpu().numpy()]

        # Clone infos list
        new_infos = [
            self.infos[companions[i].item()] if will_clone[i] else self.infos[i]
            for i in range(self.N)
        ]

        # Clone tensor data
        new_observations = _clone_tensor(self.observations, companions, will_clone)
        new_rewards = _clone_tensor(self.rewards, companions, will_clone)
        new_step_rewards = _clone_tensor(self.step_rewards, companions, will_clone)
        new_dones = _clone_tensor(self.dones, companions, will_clone)
        new_truncated = _clone_tensor(self.truncated, companions, will_clone)

        # Clone virtual rewards if present
        new_virtual_rewards = (
            _clone_tensor(self.virtual_rewards, companions, will_clone)
            if self.virtual_rewards is not None
            else None
        )

        return WalkerState(
            states=new_states,
            observations=new_observations,
            rewards=new_rewards,
            step_rewards=new_step_rewards,
            dones=new_dones,
            truncated=new_truncated,
            actions=new_actions,
            dt=new_dt,
            infos=new_infos,
            virtual_rewards=new_virtual_rewards,
        )


class FractalGas:
    """Base fractal gas algorithm.

    Uses uniform companion selection, distance-based fitness, and random actions.
    Subclasses must implement :meth:`_init_actions` and :meth:`_render_walker_frame`.

    Args:
        env: plangym / dm_control environment with step_batch method
        N: Number of walkers
        dist_coef: Distance coefficient for fitness (default 1.0)
        reward_coef: Reward coefficient for fitness (default 1.0)
        use_cumulative_reward: Use cumulative rewards for fitness (default False)
        dt_range: Range for frame skip values (default uses cls.DEFAULT_DT_RANGE)
        action_sampler: Optional custom action sampling function
        device: Device for tensor operations ('cpu' or 'cuda')
        dtype: Data type for tensors (default torch.float32)
        seed: Random seed for reproducibility
        record_frames: Record best walker frames for visualization (default False)
        n_elite: Number of elite walkers to preserve (default 0)
    """

    DEFAULT_DT_RANGE: tuple[int, int] = (1, 4)

    def __init__(
        self,
        env: Any,
        N: int,
        dist_coef: float = 1.0,
        reward_coef: float = 1.0,
        use_cumulative_reward: bool = False,
        dt_range: tuple[int, int] | None = None,
        action_sampler: Any = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int | None = None,
        record_frames: bool = False,
        n_elite: int = 0,
        death_condition=None,
    ):
        # Lazy imports to avoid circular dependency through videogames/__init__.py
        from fragile.fractalai.videogames.cloning import FractalCloningOperator
        from fragile.fractalai.videogames.kinetic import RandomActionOperator

        if dt_range is None:
            dt_range = self.DEFAULT_DT_RANGE
        self.env = env
        self.N = N
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.record_frames = record_frames
        self.n_elite = n_elite
        self.death_condition = death_condition
        self._elite_walkers: WalkerState | None = None

        # Initialize operators
        self.clone_op = FractalCloningOperator(
            dist_coef=dist_coef,
            reward_coef=reward_coef,
            use_cumulative_reward=use_cumulative_reward,
            device=device,
        )

        self.kinetic_op = RandomActionOperator(
            env=env,
            dt_range=dt_range,
            action_sampler=action_sampler,
            seed=seed,
        )

        # Metrics tracking
        self.total_steps = 0
        self.total_clones = 0
        self.iteration_count = 0

        # Set random seeds
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _init_actions(self) -> np.ndarray:
        """Return initial zero-filled action array for N walkers.

        Subclasses override to set appropriate shape and dtype.
        """
        raise NotImplementedError

    def _render_walker_frame(self, state) -> np.ndarray:
        """Render a visual frame for the given environment state.

        Returns:
            RGB array [H, W, 3] uint8
        """
        raise NotImplementedError

    def _handle_non_tuple_reset(self, reset_data) -> tuple[Any, Any, dict]:
        """Handle non-tuple return from env.reset().

        Default implementation treats *reset_data* as a dict-based env state
        (Atari pattern). Robotic subclass overrides for DMControlState.
        """
        state = reset_data
        info: dict = {}
        observation = None
        if isinstance(state, dict):
            observation = state.get("obs")
        if observation is None:
            observation = self._extract_observation_from_env()
        if observation is None:
            # Legacy fallback: infer observation through a no-op kinetic step.
            states = _make_object_array([self._copy_state(state) for _ in range(self.N)])
            actions = self._init_actions()
            dt = np.ones(self.N, dtype=int)
            _, obs_list, _, _, _, _ = self.kinetic_op.apply(states, actions, dt)
            observation = obs_list[0]
        return state, observation, info

    def _extract_observation_from_env(self) -> Any | None:
        """Try reading the current observation from env.get_state()."""
        if not hasattr(self.env, "get_state"):
            return None
        state_data = self.env.get_state()
        if isinstance(state_data, dict):
            return state_data.get("obs")
        return None

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def reset(self, initial_state: tuple | None = None) -> WalkerState:
        """Reset environment and initialize walkers.

        Args:
            initial_state: Optional ``(env_state, observation, info)`` tuple.
                If provided, all walkers start from this state instead of
                calling ``env.reset()``.

        Returns:
            Initial WalkerState with N walkers
        """
        if initial_state is not None:
            init_state, init_obs, init_info = initial_state
        else:
            init_state, init_obs, init_info = self._reset_env_with_state()

        # Replicate initial state and observation N times
        states = _make_object_array([self._copy_state(init_state) for _ in range(self.N)])
        obs_list = [self._copy_observation(init_obs) for _ in range(self.N)]

        # Convert observations to tensors
        observations = torch.tensor(np.array(obs_list), device=self.device, dtype=self.dtype)

        # Initialize all walker arrays
        rewards = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        step_rewards = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        dones = torch.zeros(self.N, device=self.device, dtype=torch.bool)
        truncated = torch.zeros(self.N, device=self.device, dtype=torch.bool)
        actions = self._init_actions()
        dt = np.ones(self.N, dtype=int)
        infos = [
            init_info.copy() if hasattr(init_info, "copy") else init_info for _ in range(self.N)
        ]

        # Reset metrics
        self.total_steps = 0
        self.total_clones = 0
        self.iteration_count = 0
        self._elite_walkers = None

        return WalkerState(
            states=states,
            observations=observations,
            rewards=rewards,
            step_rewards=step_rewards,
            dones=dones,
            truncated=truncated,
            actions=actions,
            dt=dt,
            infos=infos,
            virtual_rewards=None,
        )

    def step(
        self, state: WalkerState, actions: np.ndarray | None = None
    ) -> tuple[WalkerState, dict]:
        """Perform one iteration of the fractal gas algorithm.

        Steps:
        1. Calculate fitness (virtual rewards) using distance and rewards
        2. Decide cloning based on virtual reward comparison
        3. Clone walker states
        4. Apply random actions (kinetic operator)
        5. Update cumulative rewards and termination flags

        Args:
            state: Current walker state
            actions: Optional pre-computed actions [N]. If provided, these are
                used instead of sampling random actions.

        Returns:
            new_state: Updated walker state
            info: Dictionary with iteration metrics
        """
        # 0. Inject elites into first n_elite positions
        if self.n_elite > 0 and self._elite_walkers is not None:
            state = state.inject(self._elite_walkers, slice(0, self.n_elite))

        # 1. Calculate fitness (pass both cumulative and step rewards)
        virtual_rewards, _fitness_companions = self.clone_op.calculate_fitness(
            state.observations,
            state.rewards,  # cumulative
            state.step_rewards,  # step rewards
            state.alive,
        )

        # 2. Decide cloning
        clone_companions, will_clone = self.clone_op.decide_cloning(virtual_rewards, state.alive)

        # 3. Clone state
        state_after_clone = state.clone(clone_companions, will_clone)
        state_after_clone.virtual_rewards = virtual_rewards

        # Track cloning
        num_cloned = will_clone.sum().item()
        self.total_clones += num_cloned

        # 4. Apply kinetic operator (random actions, or provided actions)
        new_states_list, obs_np, step_rewards_np, dones_np, truncated_np, infos = (
            self.kinetic_op.apply(state_after_clone.states, actions=actions)
        )
        # step_batch returns lists; convert back to numpy object array for indexing
        new_states = _make_object_array(new_states_list)

        # 5. Convert numpy arrays to tensors
        observations = torch.tensor(obs_np, device=self.device, dtype=self.dtype)
        step_rewards_tensor = torch.tensor(step_rewards_np, device=self.device, dtype=self.dtype)
        dones_tensor = torch.tensor(dones_np, device=self.device, dtype=torch.bool)
        truncated_tensor = torch.tensor(truncated_np, device=self.device, dtype=torch.bool)

        # 6. Update cumulative rewards
        cumulative_rewards = state_after_clone.rewards + step_rewards_tensor

        # 7. Create new WalkerState
        new_state = WalkerState(
            states=new_states,
            observations=observations,
            rewards=cumulative_rewards,
            step_rewards=step_rewards_tensor,
            dones=dones_tensor,
            truncated=truncated_tensor,
            actions=self.kinetic_op.last_actions,
            dt=self.kinetic_op.last_dt,
            infos=infos,
            virtual_rewards=virtual_rewards,
        )

        # Apply custom death condition
        if self.death_condition is not None:
            custom_dead = self.death_condition(new_state)
            new_state.dones = new_state.dones | custom_dead

        # Update metrics
        self.total_steps += self.N
        self.iteration_count += 1

        # Collect info
        info = {
            "iteration": self.iteration_count,
            "num_cloned": num_cloned,
            "alive_count": new_state.alive.sum().item(),
            "mean_reward": cumulative_rewards.mean().item(),
            "max_reward": cumulative_rewards.max().item(),
            "min_reward": cumulative_rewards.min().item(),
            "mean_virtual_reward": virtual_rewards.mean().item(),
            "max_virtual_reward": virtual_rewards.max().item(),
            "min_virtual_reward": float(virtual_rewards.min().item()),
            "mean_dt": float(self.kinetic_op.last_dt.mean()),
            "min_dt": int(self.kinetic_op.last_dt.min()),
            "max_dt": int(self.kinetic_op.last_dt.max()),
            # Cloning data for tree history recording
            "clone_companions": clone_companions,
            "will_clone": will_clone,
            "_state_before_clone": state,
            "_state_after_clone": state_after_clone,
        }

        # 8. Update elite buffer
        if self.n_elite > 0:
            self._update_elites(new_state)
            elite_max = self._elite_walkers.rewards.max().item()
            info["max_reward"] = max(elite_max, info["max_reward"])

        # 9. Record best walker frame (if enabled)
        if self.record_frames:
            best_idx = new_state.rewards.argmax().item()
            best_state = new_state.states[best_idx]
            best_frame = None
            if hasattr(best_state, "rgb_frame") and best_state.rgb_frame is not None:
                best_frame = best_state.rgb_frame
            else:
                best_frame = self._render_walker_frame(best_state)
            info["best_frame"] = best_frame
            info["best_walker_idx"] = best_idx

        return new_state, info

    def _update_elites(self, state: WalkerState):
        """Update elite buffer with the best n_elite walkers overall."""
        n = self.n_elite

        if self._elite_walkers is None:
            # First call: take top n from current population
            _, top_idx = state.rewards.topk(min(n, state.N))
            self._elite_walkers = self._extract_walkers(state, top_idx)
            return

        # Concatenate elite and current rewards, pick global top n
        all_rewards = torch.cat([self._elite_walkers.rewards, state.rewards])
        _, top_idx = all_rewards.topk(min(n, len(all_rewards)))

        n_elite_current = self._elite_walkers.N
        # Split indices into those from elite buffer vs current state
        elite_mask = top_idx < n_elite_current
        elite_idx = top_idx[elite_mask]
        state_idx = top_idx[~elite_mask] - n_elite_current

        parts = []
        if elite_idx.numel() > 0:
            parts.append(self._extract_walkers(self._elite_walkers, elite_idx))
        if state_idx.numel() > 0:
            parts.append(self._extract_walkers(state, state_idx))

        if len(parts) == 1:
            self._elite_walkers = parts[0]
        else:
            self._elite_walkers = self._concat_walkers(parts[0], parts[1])

    @staticmethod
    def _extract_walkers(state: WalkerState, indices: Tensor) -> WalkerState:
        """Extract a subset of walkers by index into a new WalkerState."""
        idx_np = indices.cpu().numpy()
        return WalkerState(
            states=_make_object_array([
                state.states[i].copy() if hasattr(state.states[i], "copy") else state.states[i]
                for i in idx_np
            ]),
            observations=state.observations[indices].clone(),
            rewards=state.rewards[indices].clone(),
            step_rewards=state.step_rewards[indices].clone(),
            dones=state.dones[indices].clone(),
            truncated=state.truncated[indices].clone(),
            actions=state.actions[idx_np].copy(),
            dt=state.dt[idx_np].copy(),
            infos=[state.infos[i] for i in idx_np],
            virtual_rewards=(
                state.virtual_rewards[indices].clone()
                if state.virtual_rewards is not None
                else None
            ),
        )

    @staticmethod
    def _concat_walkers(a: WalkerState, b: WalkerState) -> WalkerState:
        """Concatenate two WalkerStates along the walker dimension."""
        return WalkerState(
            states=np.concatenate([a.states, b.states]),
            observations=torch.cat([a.observations, b.observations]),
            rewards=torch.cat([a.rewards, b.rewards]),
            step_rewards=torch.cat([a.step_rewards, b.step_rewards]),
            dones=torch.cat([a.dones, b.dones]),
            truncated=torch.cat([a.truncated, b.truncated]),
            actions=np.concatenate([a.actions, b.actions]),
            dt=np.concatenate([a.dt, b.dt]),
            infos=a.infos + b.infos,
            virtual_rewards=(
                torch.cat([a.virtual_rewards, b.virtual_rewards])
                if a.virtual_rewards is not None and b.virtual_rewards is not None
                else None
            ),
        )

    def run(
        self,
        max_iterations: int = 1000,
        stop_when_all_dead: bool = False,
        initial_state: tuple | None = None,
    ) -> tuple[WalkerState, list[dict]]:
        """Run the fractal gas algorithm for multiple iterations.

        Args:
            max_iterations: Maximum number of iterations to run
            stop_when_all_dead: If True, stop when all walkers are dead
            initial_state: Optional ``(env_state, observation, info)`` tuple

        Returns:
            final_state: Final walker state
            history: List of info dicts from each iteration
        """
        state = self.reset(initial_state=initial_state)
        history = []

        for _ in range(max_iterations):
            state, info = self.step(state)
            history.append(info)

            # Check stopping condition
            if stop_when_all_dead and not state.alive.any():
                break

        return state, history

    def run_with_tree(
        self,
        max_iterations: int = 1000,
        stop_when_all_dead: bool = False,
        task_label: str = "",
        initial_state: tuple | None = None,
    ):
        """Run the algorithm and record every step in an :class:`AtariTreeHistory`.

        Same loop as :meth:`run` but returns a graph-backed tree that
        captures cloning lineage instead of flat info-dict lists.

        Args:
            max_iterations: Maximum number of iterations to run.
            stop_when_all_dead: If True, stop when all walkers are dead.
            task_label: Name stored in the tree metadata.
            initial_state: Optional ``(env_state, observation, info)`` tuple.

        Returns:
            AtariTreeHistory with the full run recorded.
        """
        from fragile.fractalai.videogames.atari_tree_history import AtariTreeHistory

        state = self.reset(initial_state=initial_state)
        tree = AtariTreeHistory(N=self.N, game_name=task_label, max_iterations=max_iterations)
        tree.record_initial_atari_state(state)

        for _ in range(max_iterations):
            prev_state = state
            state, info = self.step(prev_state)

            tree.record_atari_step(
                state_before=prev_state,
                state_after_clone=info["_state_after_clone"],
                state_final=state,
                info=info,
                clone_companions=info["clone_companions"],
                will_clone=info["will_clone"],
                best_frame=info.get("best_frame"),
            )

            if stop_when_all_dead and not state.alive.any():
                break

        return tree

    def get_best_walker(self, state: WalkerState) -> tuple[int, float]:
        """Get the index and reward of the best walker.

        Args:
            state: Current walker state

        Returns:
            idx: Index of best walker
            reward: Reward of best walker
        """
        best_idx = state.rewards.argmax().item()
        best_reward = state.rewards[best_idx].item()
        return best_idx, best_reward

    def _reset_env_with_state(self) -> tuple[Any, Any, dict]:
        """Reset env and normalize output to (state, observation, info)."""
        try:
            reset_data = self.env.reset(return_state=True)
        except TypeError:
            # Fallback for non-plangym-style mocks.
            reset_data = self.env.reset()

        if isinstance(reset_data, tuple):
            if len(reset_data) == 3:
                state, observation, info = reset_data
            elif len(reset_data) == 2:
                state, observation = reset_data
                info = {}
            else:
                msg = f"Unexpected reset output length from env.reset: {len(reset_data)}"
                raise RuntimeError(msg)
        else:
            state, observation, info = self._handle_non_tuple_reset(reset_data)

        if observation is None:
            msg = "Could not obtain an initial observation from env.reset."
            raise RuntimeError(msg)
        if not isinstance(info, dict):
            info = {}
        return state, observation, info

    @staticmethod
    def _copy_state(state: Any) -> Any:
        """Return a defensive copy of an environment state when supported."""
        return state.copy() if hasattr(state, "copy") else state

    @staticmethod
    def _copy_observation(observation: Any) -> Any:
        """Return a defensive copy of an observation when supported."""
        return observation.copy() if hasattr(observation, "copy") else observation
