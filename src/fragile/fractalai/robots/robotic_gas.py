"""Fractal gas algorithm for DM Control environments with uniform companion selection."""

from typing import Any

import numpy as np
import torch

from fragile.fractalai.videogames.atari_gas import WalkerState
from fragile.fractalai.videogames.cloning import FractalCloningOperator
from fragile.fractalai.videogames.kinetic import RandomActionOperator


class RoboticFractalGas:
    """Fractal gas algorithm for DM Control continuous-control environments.

    Uses uniform companion selection, distance-based fitness, and random actions
    sampled from continuous action spaces. Analogous to AtariFractalGas but
    adapted for dm_control environments.

    Args:
        env: DMControlEnv instance (or any env with step_batch, action_space.sample)
        N: Number of walkers
        dist_coef: Distance coefficient for fitness (default 1.0)
        reward_coef: Reward coefficient for fitness (default 1.0)
        use_cumulative_reward: Use cumulative rewards for fitness (default False)
        dt_range: Range for action repeat values (default (1, 1))
        action_sampler: Optional custom action sampling function
        device: Device for tensor operations ('cpu' or 'cuda')
        dtype: Data type for tensors (default torch.float32)
        seed: Random seed for reproducibility
        record_frames: Record best walker frames for visualization (default False)
        n_elite: Number of elite walkers to preserve (default 0)
    """

    def __init__(
        self,
        env: Any,
        N: int,
        dist_coef: float = 1.0,
        reward_coef: float = 1.0,
        use_cumulative_reward: bool = False,
        dt_range: tuple[int, int] = (1, 1),
        action_sampler: Any = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int | None = None,
        record_frames: bool = False,
        n_elite: int = 0,
    ):
        self.env = env
        self.N = N
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.record_frames = record_frames
        self.n_elite = n_elite
        self._elite_walkers: WalkerState | None = None

        # Initialize operators (reuse from videogames module)
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

        # Continuous actions: shape [N, action_dim]
        action_shape = self.env.action_space.shape
        actions = np.zeros((self.N, *action_shape), dtype=np.float64)
        dt = np.ones(self.N, dtype=int)
        infos = [init_info.copy() if hasattr(init_info, "copy") else {} for _ in range(self.N)]

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
        # 0. Inject elites into first n_elite positions
        if self.n_elite > 0 and self._elite_walkers is not None:
            state = state.inject(self._elite_walkers, slice(0, self.n_elite))

        # 1. Calculate fitness
        virtual_rewards, _fitness_companions = self.clone_op.calculate_fitness(
            state.observations,
            state.rewards,
            state.step_rewards,
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

        # 4. Apply kinetic operator (random actions from continuous space)
        new_states_list, obs_np, step_rewards_np, dones_np, truncated_np, infos = (
            self.kinetic_op.apply(state_after_clone.states)
        )
        new_states = np.array(new_states_list, dtype=object)

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

        # 8. Update elite buffer
        if self.n_elite > 0:
            self._update_elites(new_state)
            elite_max = self._elite_walkers.rewards.max().item()
            if elite_max > info["max_reward"]:
                info["max_reward"] = elite_max

        # 9. Record best walker frame
        if self.record_frames:
            best_idx = new_state.rewards.argmax().item()
            best_frame = self._render_walker_frame(new_state.states[best_idx])
            info["best_frame"] = best_frame
            info["best_walker_idx"] = best_idx

        return new_state, info

    def _update_elites(self, state: WalkerState):
        """Update elite buffer with the best n_elite walkers overall."""
        n = self.n_elite

        if self._elite_walkers is None:
            _, top_idx = state.rewards.topk(min(n, state.N))
            self._elite_walkers = self._extract_walkers(state, top_idx)
            return

        all_rewards = torch.cat([self._elite_walkers.rewards, state.rewards])
        _, top_idx = all_rewards.topk(min(n, len(all_rewards)))

        n_elite_current = self._elite_walkers.N
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
    def _extract_walkers(state: WalkerState, indices) -> WalkerState:
        """Extract a subset of walkers by index into a new WalkerState."""
        idx_np = indices.cpu().numpy()
        return WalkerState(
            states=np.array(
                [state.states[i].copy() if hasattr(state.states[i], "copy") else state.states[i] for i in idx_np],
                dtype=object,
            ),
            observations=state.observations[indices].clone(),
            rewards=state.rewards[indices].clone(),
            step_rewards=state.step_rewards[indices].clone(),
            dones=state.dones[indices].clone(),
            truncated=state.truncated[indices].clone(),
            actions=state.actions[idx_np].copy(),
            dt=state.dt[idx_np].copy(),
            infos=[state.infos[i] for i in idx_np],
            virtual_rewards=state.virtual_rewards[indices].clone() if state.virtual_rewards is not None else None,
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

            if stop_when_all_dead and not state.alive.any():
                break

        return state, history

    def get_best_walker(self, state: WalkerState) -> tuple[int, float]:
        """Get the index and reward of the best walker."""
        best_idx = state.rewards.argmax().item()
        best_reward = state.rewards[best_idx].item()
        return best_idx, best_reward

    def _render_walker_frame(self, state) -> np.ndarray:
        """Render visual frame for a walker state.

        Args:
            state: DMControlState to render

        Returns:
            RGB array [H, W, 3] uint8, or zeros if rendering fails
        """
        saved_state = None
        if hasattr(self.env, "get_state"):
            saved_state = self._copy_state(self.env.get_state())
        elif hasattr(self.env, "clone_state"):
            saved_state = self._copy_state(self.env.clone_state())

        try:
            if hasattr(self.env, "set_state"):
                self.env.set_state(state)
            elif hasattr(self.env, "restore_state"):
                self.env.restore_state(state)

            if hasattr(self.env, "render"):
                frame = self.env.render()
                if frame is not None and isinstance(frame, np.ndarray):
                    return frame.astype(np.uint8)

            return np.zeros((480, 480, 3), dtype=np.uint8)

        except Exception:
            return np.zeros((480, 480, 3), dtype=np.uint8)

        finally:
            if saved_state is not None:
                if hasattr(self.env, "set_state"):
                    self.env.set_state(saved_state)
                elif hasattr(self.env, "restore_state"):
                    self.env.restore_state(saved_state)

    def _reset_env_with_state(self) -> tuple:
        """Reset env and normalize output to (state, observation, info)."""
        try:
            reset_data = self.env.reset(return_state=True)
        except TypeError:
            reset_data = self.env.reset()

        if isinstance(reset_data, tuple):
            if len(reset_data) == 3:
                state, observation, info = reset_data
            elif len(reset_data) == 2:
                state, observation = reset_data
                info = {}
            else:
                raise RuntimeError(f"Unexpected reset output length: {len(reset_data)}")
        else:
            # DMControlEnv.reset() returns a DMControlState directly
            state = reset_data
            observation = state.observation if hasattr(state, "observation") else None
            info = {}
            if observation is None:
                observation = self._extract_observation_from_env()

        if observation is None:
            raise RuntimeError("Could not obtain an initial observation from env.reset.")
        if not isinstance(info, dict):
            info = {}
        return state, observation, info

    def _extract_observation_from_env(self):
        """Try reading the current observation from env.get_state()."""
        if not hasattr(self.env, "get_state"):
            return None
        state_data = self.env.get_state()
        if isinstance(state_data, dict):
            return state_data.get("obs")
        if hasattr(state_data, "observation"):
            return state_data.observation
        return None

    @staticmethod
    def _copy_state(state):
        """Return a defensive copy of an environment state."""
        return state.copy() if hasattr(state, "copy") else state

    @staticmethod
    def _copy_observation(observation):
        """Return a defensive copy of an observation."""
        return observation.copy() if hasattr(observation, "copy") else observation
