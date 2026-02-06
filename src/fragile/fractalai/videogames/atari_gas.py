"""Fractal gas algorithm for Atari games with uniform companion selection."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fragile.fractalai.videogames.cloning import clone_walker_data, FractalCloningOperator
from fragile.fractalai.videogames.kinetic import RandomActionOperator


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
        new_states[will_clone.cpu().numpy()] = self.states[
            companions[will_clone].cpu().numpy()
        ]

        new_actions = self.actions.copy()
        new_actions[will_clone.cpu().numpy()] = self.actions[
            companions[will_clone].cpu().numpy()
        ]

        new_dt = self.dt.copy()
        new_dt[will_clone.cpu().numpy()] = self.dt[companions[will_clone].cpu().numpy()]

        # Clone infos list
        new_infos = [
            self.infos[companions[i].item()] if will_clone[i] else self.infos[i]
            for i in range(self.N)
        ]

        # Clone tensor data
        new_observations = clone_walker_data(self.observations, companions, will_clone)
        new_rewards = clone_walker_data(self.rewards, companions, will_clone)
        new_step_rewards = clone_walker_data(self.step_rewards, companions, will_clone)
        new_dones = clone_walker_data(self.dones, companions, will_clone)
        new_truncated = clone_walker_data(self.truncated, companions, will_clone)

        # Clone virtual rewards if present
        new_virtual_rewards = (
            clone_walker_data(self.virtual_rewards, companions, will_clone)
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


class AtariFractalGas:
    """Fractal gas algorithm for Atari games.

    Uses uniform companion selection, distance-based fitness, and random actions.
    Simpler alternative to Euclidean gas with no embeddings or neighbor finding.

    Args:
        env: plangym environment with step_batch method
        N: Number of walkers
        dist_coef: Distance coefficient for fitness (default 1.0)
        reward_coef: Reward coefficient for fitness (default 1.0)
        use_cumulative_reward: Use cumulative rewards for fitness (default False)
        dt_range: Range for frame skip values (default (1, 4))
        action_sampler: Optional custom action sampling function
        device: Device for tensor operations ('cpu' or 'cuda')
        dtype: Data type for tensors (default torch.float32)
        seed: Random seed for reproducibility
        record_frames: Record best walker frames for visualization (default False)
    """

    def __init__(
        self,
        env: Any,
        N: int,
        dist_coef: float = 1.0,
        reward_coef: float = 1.0,
        use_cumulative_reward: bool = False,
        dt_range: tuple[int, int] = (1, 4),
        action_sampler: Any = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int | None = None,
        record_frames: bool = False,
    ):
        self.env = env
        self.N = N
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.record_frames = record_frames

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

    def reset(self) -> WalkerState:
        """Reset environment and initialize walkers.

        Returns:
            Initial WalkerState with N walkers
        """
        init_state, init_obs, init_info = self._reset_env_with_state()

        # Replicate initial state and observation N times
        states = np.array([self._copy_state(init_state) for _ in range(self.N)], dtype=object)
        obs_list = [self._copy_observation(init_obs) for _ in range(self.N)]

        # Convert observations to tensors
        observations = torch.tensor(np.array(obs_list), device=self.device, dtype=self.dtype)

        # Initialize all walker arrays
        rewards = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        step_rewards = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        dones = torch.zeros(self.N, device=self.device, dtype=torch.bool)
        truncated = torch.zeros(self.N, device=self.device, dtype=torch.bool)
        actions = np.zeros(self.N, dtype=int)
        dt = np.ones(self.N, dtype=int)
        infos = [init_info.copy() if hasattr(init_info, "copy") else init_info for _ in range(self.N)]

        # Reset metrics
        self.total_steps = 0
        self.total_clones = 0
        self.iteration_count = 0

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

    def step(self, state: WalkerState) -> tuple[WalkerState, dict]:
        """Perform one iteration of the fractal gas algorithm.

        Steps:
        1. Calculate fitness (virtual rewards) using distance and rewards
        2. Decide cloning based on virtual reward comparison
        3. Clone walker states
        4. Apply random actions (kinetic operator)
        5. Update cumulative rewards and termination flags

        Args:
            state: Current walker state

        Returns:
            new_state: Updated walker state
            info: Dictionary with iteration metrics
        """
        # 1. Calculate fitness (pass both cumulative and step rewards)
        virtual_rewards, _fitness_companions = self.clone_op.calculate_fitness(
            state.observations,
            state.rewards,  # cumulative
            state.step_rewards,  # step rewards
            state.alive,
        )

        # 2. Decide cloning
        clone_companions, will_clone = self.clone_op.decide_cloning(
            virtual_rewards, state.alive
        )

        # 3. Clone state
        state_after_clone = state.clone(clone_companions, will_clone)
        state_after_clone.virtual_rewards = virtual_rewards

        # Track cloning
        num_cloned = will_clone.sum().item()
        self.total_clones += num_cloned

        # 4. Apply kinetic operator (random actions)
        new_states, obs_np, step_rewards_np, dones_np, truncated_np, infos = (
            self.kinetic_op.apply(state_after_clone.states)
        )

        # 5. Convert numpy arrays to tensors
        observations = torch.tensor(obs_np, device=self.device, dtype=self.dtype)
        step_rewards_tensor = torch.tensor(
            step_rewards_np, device=self.device, dtype=self.dtype
        )
        dones_tensor = torch.tensor(dones_np, device=self.device, dtype=torch.bool)
        truncated_tensor = torch.tensor(
            truncated_np, device=self.device, dtype=torch.bool
        )

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
        }

        # 8. Record best walker frame (if enabled)
        if self.record_frames:
            best_idx = new_state.rewards.argmax().item()
            best_frame = self._render_walker_frame(new_state.states[best_idx])
            info["best_frame"] = best_frame
            info["best_walker_idx"] = best_idx

        return new_state, info

    def run(
        self, max_iterations: int = 1000, stop_when_all_dead: bool = False
    ) -> tuple[WalkerState, list[dict]]:
        """Run the fractal gas algorithm for multiple iterations.

        Args:
            max_iterations: Maximum number of iterations to run
            stop_when_all_dead: If True, stop when all walkers are dead

        Returns:
            final_state: Final walker state
            history: List of info dicts from each iteration
        """
        state = self.reset()
        history = []

        for _ in range(max_iterations):
            state, info = self.step(state)
            history.append(info)

            # Check stopping condition
            if stop_when_all_dead and not state.alive.any():
                break

        return state, history

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

    def _render_walker_frame(self, state) -> np.ndarray:
        """Render visual frame for a walker state.

        Args:
            state: Environment state to render

        Returns:
            RGB array [H, W, 3] uint8, or zeros if rendering fails
        """
        # Save current environment state
        saved_state = None
        if hasattr(self.env, "get_state"):
            saved_state = self._copy_state(self.env.get_state())
        elif hasattr(self.env, "clone_state"):
            saved_state = self._copy_state(self.env.clone_state())

        try:
            # Set environment to walker's state
            if hasattr(self.env, "set_state"):
                self.env.set_state(state)
            elif hasattr(self.env, "restore_state"):
                self.env.restore_state(state)

            # Render frame
            if hasattr(self.env, "render"):
                frame = self.env.render()
                if frame is not None and isinstance(frame, np.ndarray):
                    return frame.astype(np.uint8)

            # Fallback to blank frame
            return np.zeros((210, 160, 3), dtype=np.uint8)

        except Exception:
            # If rendering fails, return blank frame
            return np.zeros((210, 160, 3), dtype=np.uint8)

        finally:
            # Restore environment state
            if saved_state is not None:
                if hasattr(self.env, "set_state"):
                    self.env.set_state(saved_state)
                elif hasattr(self.env, "restore_state"):
                    self.env.restore_state(saved_state)

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
            state = reset_data
            info = {}
            observation = self._extract_observation_from_env()
            if observation is None:
                # Legacy fallback: infer observation through a no-op kinetic step.
                states = np.array([self._copy_state(state) for _ in range(self.N)], dtype=object)
                actions = np.zeros(self.N, dtype=int)
                dt = np.ones(self.N, dtype=int)
                _, obs_list, _, _, _, _ = self.kinetic_op.apply(states, actions, dt)
                observation = obs_list[0]

        if observation is None:
            msg = "Could not obtain an initial observation from env.reset."
            raise RuntimeError(msg)
        if not isinstance(info, dict):
            info = {}
        return state, observation, info

    def _extract_observation_from_env(self) -> Any | None:
        """Try reading the current observation from env.get_state()."""
        if not hasattr(self.env, "get_state"):
            return None
        state_data = self.env.get_state()
        if isinstance(state_data, dict):
            return state_data.get("obs")
        return None

    @staticmethod
    def _copy_state(state: Any) -> Any:
        """Return a defensive copy of an environment state when supported."""
        return state.copy() if hasattr(state, "copy") else state

    @staticmethod
    def _copy_observation(observation: Any) -> Any:
        """Return a defensive copy of an observation when supported."""
        return observation.copy() if hasattr(observation, "copy") else observation
