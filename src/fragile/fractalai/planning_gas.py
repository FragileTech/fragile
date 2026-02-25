"""Two-level planning fractal gas.

Uses an inner fractal gas as a planning oracle to select each action in an
outer sequential trajectory. At each outer step, N inner walkers explore from
the current state for ``tau_inner`` iterations. The best action is selected by
averaging the "root actions" (the first action each surviving walker's lineage
took at the tree root).
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from fragile.fractalai.fractal_gas import _make_object_array


@dataclass
class PlanningTrajectory:
    """Recorded trajectory from a planning run.

    Attributes:
        states: Outer env states (one per step + initial).
        observations: Outer observations.
        actions: Selected actions (one per step).
        rewards: Per-step rewards.
        cumulative_rewards: Running total of rewards.
        dones: Terminal flags per step.
        planning_infos: Inner gas statistics per step.
        frames: Rendered frames (if record_frames was set).
    """

    states: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    cumulative_rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    planning_infos: list = field(default_factory=list)
    frames: list | None = None

    @property
    def total_reward(self) -> float:
        """Sum of per-step rewards."""
        return float(sum(self.rewards))

    @property
    def num_steps(self) -> int:
        """Number of outer steps taken."""
        return len(self.actions)

    def to_planning_history(self, N: int, env_name: str) -> "PlanningHistory":
        """Convert to a PlanningHistory preserving all outer-loop data.

        Args:
            N: Number of inner walkers used.
            env_name: Environment name.

        Returns:
            PlanningHistory instance.
        """
        return PlanningHistory.from_trajectory(self, N, env_name)


@dataclass
class PlanningHistory:
    """Outer-loop statistics from a planning run.

    Preserves per-step rewards, actions, inner planning quality,
    and termination flags -- data lost in AtariHistory/RoboticHistory.
    """

    iterations: list[int]
    step_rewards: list[float]
    cumulative_rewards: list[float]
    actions: list
    dones: list[bool]

    # Inner planning quality
    inner_alive_counts: list[int]
    inner_mean_rewards: list[float]
    inner_max_rewards: list[float]
    inner_iterations: list[int]

    # Frames
    best_frames: list[np.ndarray | None]

    # Config
    N: int
    max_steps: int
    env_name: str

    @classmethod
    def from_trajectory(cls, traj: PlanningTrajectory, N: int, env_name: str) -> "PlanningHistory":
        """Build from a PlanningTrajectory.

        Args:
            traj: Recorded trajectory from a planning run.
            N: Number of inner walkers.
            env_name: Environment name.

        Returns:
            PlanningHistory instance.
        """
        n = traj.num_steps
        return cls(
            iterations=list(range(n)),
            step_rewards=list(traj.rewards),
            cumulative_rewards=list(traj.cumulative_rewards),
            actions=list(traj.actions),
            dones=list(traj.dones),
            inner_alive_counts=[pi.get("alive_count", N) for pi in traj.planning_infos],
            inner_mean_rewards=[pi.get("inner_mean_reward", 0.0) for pi in traj.planning_infos],
            inner_max_rewards=[pi.get("inner_max_reward", 0.0) for pi in traj.planning_infos],
            inner_iterations=[pi.get("inner_iterations", 0) for pi in traj.planning_infos],
            best_frames=list(traj.frames) if traj.frames else [None] * n,
            N=N,
            max_steps=n,
            env_name=env_name,
        )

    def to_atari_history(self):
        """Convert to AtariHistory for the Atari visualizer."""
        from fragile.fractalai.videogames.atari_history import AtariHistory

        n = len(self.iterations)
        return AtariHistory(
            iterations=list(self.iterations),
            rewards_mean=list(self.cumulative_rewards),
            rewards_max=list(self.cumulative_rewards),
            rewards_min=list(self.step_rewards),
            rewards_std=[0.0] * n,
            alive_counts=list(self.inner_alive_counts),
            num_cloned=[0] * n,
            virtual_rewards_mean=list(self.inner_mean_rewards),
            virtual_rewards_max=list(self.inner_max_rewards),
            virtual_rewards_min=[0.0] * n,
            dt_mean=[1.0] * n,
            dt_min=[1] * n,
            dt_max=[1] * n,
            best_frames=list(self.best_frames),
            best_rewards=list(self.cumulative_rewards),
            best_indices=[0] * n,
            N=self.N,
            max_iterations=self.max_steps,
            game_name=self.env_name,
        )

    def to_robotic_history(self):
        """Convert to RoboticHistory for the robotics visualizer."""
        from fragile.fractalai.robots.robotic_history import RoboticHistory

        n = len(self.iterations)
        return RoboticHistory(
            iterations=list(self.iterations),
            rewards_mean=list(self.cumulative_rewards),
            rewards_max=list(self.cumulative_rewards),
            rewards_min=list(self.step_rewards),
            alive_counts=list(self.inner_alive_counts),
            num_cloned=[0] * n,
            virtual_rewards_mean=list(self.inner_mean_rewards),
            virtual_rewards_max=list(self.inner_max_rewards),
            virtual_rewards_min=[0.0] * n,
            dt_mean=[1.0] * n,
            dt_min=[1] * n,
            dt_max=[1] * n,
            best_frames=list(self.best_frames),
            best_rewards=list(self.cumulative_rewards),
            best_indices=[0] * n,
            N=self.N,
            max_iterations=self.max_steps,
            task_name=self.env_name,
        )

    @property
    def has_frames(self) -> bool:
        """Check if frames were recorded."""
        return self.best_frames[0] is not None if self.best_frames else False


class PlanningFractalGas:
    """Two-level fractal gas planner.

    At each outer step an inner :class:`FractalGas` explores from the current
    state for *tau_inner* iterations and the best root action is selected.

    Args:
        env: Environment used for both inner planning and outer stepping.
        N: Number of inner walkers.
        tau_inner: Inner planning horizon (iterations).
        inner_gas_cls: FractalGas subclass for the inner gas.  Defaults to
            :class:`AtariFractalGas`.
        dist_coef: Distance coefficient forwarded to the inner gas.
        reward_coef: Reward coefficient forwarded to the inner gas.
        use_cumulative_reward: Use cumulative rewards in the inner gas.
        dt_range: Frame-skip range forwarded to the inner gas.
        action_sampler: Optional custom action sampler for the inner gas.
        device: Torch device.
        dtype: Torch dtype.
        seed: Random seed.
        record_frames: Record rendered frames from the outer trajectory.
        n_elite: Number of elite walkers in the inner gas.
        outer_dt: Frame-skip value for the outer environment step.
    """

    def __init__(
        self,
        env: Any,
        N: int,
        tau_inner: int,
        inner_gas_cls=None,
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
        outer_dt: int = 1,
        death_condition=None,
        inertial: bool = False,
        inertial_noise_std: float = 0.1,
    ):
        if inner_gas_cls is None:
            from fragile.fractalai.videogames.atari_gas import AtariFractalGas

            inner_gas_cls = AtariFractalGas

        self.env = env
        self.N = N
        self.tau_inner = tau_inner
        self.outer_dt = outer_dt
        self.record_frames = record_frames
        self.device = device
        self.dtype = dtype

        self.inner_gas = inner_gas_cls(
            env=env,
            N=N,
            dist_coef=dist_coef,
            reward_coef=reward_coef,
            use_cumulative_reward=use_cumulative_reward,
            dt_range=dt_range,
            action_sampler=action_sampler,
            device=device,
            dtype=dtype,
            seed=seed,
            record_frames=False,
            n_elite=n_elite,
            death_condition=death_condition,
            inertial=inertial,
            inertial_noise_std=inertial_noise_std,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> tuple[Any, Any, dict]:
        """Reset the outer environment.

        Returns:
            (state, observation, info) tuple from the environment.
        """
        return self.inner_gas._reset_env_with_state()

    def plan_action(self, state: Any, obs: Any, info: dict) -> tuple[Any, dict]:
        """Run the inner fractal gas and select an action.

        Args:
            state: Current outer environment state.
            obs: Current observation.
            info: Current info dict.

        Returns:
            action: The selected action.
            plan_info: Dictionary with planning statistics.
        """
        inner_state = self.inner_gas.reset(initial_state=(state, obs, info))
        root_actions: np.ndarray | None = None
        initial_actions = self._sample_initial_actions(inner_state.N)

        for k in range(self.tau_inner):
            if k == 0:
                inner_state, step_info = self.inner_gas.step(inner_state, actions=initial_actions)
            else:
                inner_state, step_info = self.inner_gas.step(inner_state)

            if k == 0:
                # After the first kinetic step each walker has its first action
                root_actions = inner_state.actions.copy()
            else:
                # Propagate root actions through cloning that happened at the
                # start of this step (before kinetic).
                wc = step_info["will_clone"].cpu().numpy()
                companions = step_info["clone_companions"].cpu().numpy()
                root_actions[wc] = root_actions[companions[wc]]

        action = self._select_action(root_actions, inner_state.alive, inner_state.rewards)

        plan_info = {
            "alive_count": inner_state.alive.sum().item(),
            "inner_max_reward": inner_state.rewards.max().item(),
            "inner_mean_reward": inner_state.rewards.mean().item(),
            "inner_iterations": self.tau_inner,
        }
        return action, plan_info

    def step(
        self, state: Any, obs: Any, info: dict
    ) -> tuple[Any, Any, float, bool, bool, dict, dict]:
        """Plan an action and apply it to the outer environment.

        Args:
            state: Current outer state.
            obs: Current observation.
            info: Current info dict.

        Returns:
            new_state, new_obs, reward, done, truncated, new_info, step_info
        """
        action, plan_info = self.plan_action(state, obs, info)
        new_state, new_obs, reward, done, truncated, new_info = self._outer_step(state, action)
        step_info = {"action": action, "plan_info": plan_info}
        return new_state, new_obs, reward, done, truncated, new_info, step_info

    def run(self, max_steps: int = 100, render: bool = False) -> PlanningTrajectory:
        """Run the full planning loop.

        Args:
            max_steps: Maximum number of outer steps.
            render: If True, record rendered frames (requires record_frames).

        Returns:
            PlanningTrajectory with the recorded trajectory.
        """
        state, obs, info = self.reset()

        traj = PlanningTrajectory(
            frames=[] if (render or self.record_frames) else None,
        )
        traj.states.append(state)
        traj.observations.append(obs)

        cum_reward = 0.0
        for _ in range(max_steps):
            new_state, new_obs, reward, done, truncated, new_info, step_info = self.step(
                state, obs, info
            )

            cum_reward += float(reward)
            traj.actions.append(step_info["action"])
            traj.rewards.append(float(reward))
            traj.cumulative_rewards.append(cum_reward)
            traj.dones.append(bool(done))
            traj.planning_infos.append(step_info["plan_info"])
            traj.states.append(new_state)
            traj.observations.append(new_obs)

            state, obs, info = new_state, new_obs, new_info

            if done or truncated:
                break

        return traj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample_initial_actions(self, N: int) -> np.ndarray:
        """Sample N random actions from the environment's action space."""
        return self.inner_gas.kinetic_op.sample_actions(N)

    def _select_action(
        self,
        root_actions: np.ndarray,
        alive_mask: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Any:
        """Select the best action from ALL walkers' root actions.

        For discrete (integer) actions, returns the majority vote (mode).
        For continuous actions, returns the mean across all walkers.
        """
        if root_actions.dtype.kind == "i":
            # Discrete: majority vote (most common action)
            if root_actions.ndim == 1:
                counts = np.bincount(root_actions)
                action = int(counts.argmax())
            else:
                # Multi-dimensional discrete: vote per dimension
                action = np.array(
                    [int(np.bincount(root_actions[:, d]).argmax()) for d in range(root_actions.shape[1])]
                )
        else:
            # Continuous: mean of all walkers
            action = root_actions.mean(axis=0)

        return action

    def _outer_step(self, state: Any, action: Any) -> tuple[Any, Any, float, bool, bool, dict]:
        """Apply a single action to the outer environment."""
        states = _make_object_array([state])
        actions = np.array([action])
        dt = np.array([self.outer_dt], dtype=int)

        result = self.env.step_batch(states=states, actions=actions, dt=dt, return_state=True)

        if len(result) == 5:
            new_states, observations, rewards, dones, infos = result
            truncated = False
        else:
            new_states, observations, rewards, dones, truncated_arr, infos = result
            truncated = bool(truncated_arr[0])

        new_state = new_states[0]
        new_obs = observations[0]
        reward = float(rewards[0])
        done = bool(dones[0])
        new_info = infos[0] if isinstance(infos, list) else {}

        return new_state, new_obs, reward, done, truncated, new_info
