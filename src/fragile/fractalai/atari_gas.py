from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field
import torch
from torch import Tensor

from fragile.fractalai.core.euclidean_gas import CloningParams, SwarmState, VectorizedOps


class AtariSwarmState(SwarmState):
    """Swarm state that tracks Atari env metadata alongside Euclidean embeddings.

    Attributes:
        x: Position embeddings [N, d]
        v: Velocity embeddings [N, d]
        env_states: Environment states for each walker [N]
        observations: Raw observations from environment [N, ...]
        rewards: Cumulative rewards [N]
        step_rewards: Rewards from last step [N]
        dones: Terminal state flags [N]
        truncated: Truncation flags [N]
        actions: Last actions taken [N]
        dts: Number of times each action was applied consecutively [N]
        infos: Info dictionaries from environment [N]
    """

    def __init__(
        self,
        x: Tensor,
        v: Tensor,
        env_states: Sequence[Any],
        observations: Tensor,
        rewards: Tensor,
        dones: Tensor,
        truncated: Tensor,
        actions: Tensor,
        dts: Tensor,
        infos: Sequence[Any] | None = None,
        step_rewards: Tensor | None = None,
    ) -> None:
        super().__init__(x, v)
        env_states = np.asarray(env_states, dtype=object)
        if env_states.shape[0] != self.N:
            msg = f"Expected {self.N} env states, got {env_states.shape[0]}"
            raise ValueError(msg)

        observations = observations.to(device=self.device, dtype=self.dtype)
        if observations.shape[0] != self.N:
            msg = f"Expected observations with first dim {self.N}, got {observations.shape[0]}"
            raise ValueError(msg)

        self.env_states = env_states
        self.observations = observations
        self.rewards = rewards.to(device=self.device, dtype=self.dtype)
        if step_rewards is None:
            step_rewards = torch.zeros_like(self.rewards)
        self.step_rewards = step_rewards.to(device=self.device, dtype=self.dtype)
        self.dones = dones.to(device=self.device, dtype=torch.bool)
        self.truncated = truncated.to(device=self.device, dtype=torch.bool)
        self.actions = actions.to(device=self.device, dtype=torch.long)
        self.dts = dts.to(device=self.device, dtype=torch.long)

        if infos is None:
            self.infos = [{} for _ in range(self.N)]
        else:
            infos = list(infos)
            if len(infos) != self.N:
                msg = f"Expected {self.N} info entries, got {len(infos)}"
                raise ValueError(msg)
            self.infos = [info.copy() if hasattr(info, "copy") else info for info in infos]

    @property
    def obs_shape(self) -> tuple[int, ...]:
        """Shape of raw observations (excluding walker dimension)."""
        return tuple(self.observations.shape[1:])

    def clone(self) -> AtariSwarmState:
        """Deep copy the Atari swarm state."""
        return AtariSwarmState(
            x=self.x.clone(),
            v=self.v.clone(),
            env_states=self.env_states.copy(),
            observations=self.observations.clone(),
            rewards=self.rewards.clone(),
            step_rewards=self.step_rewards.clone(),
            dones=self.dones.clone(),
            truncated=self.truncated.clone(),
            actions=self.actions.clone(),
            dts=self.dts.clone(),
            infos=[info.copy() if hasattr(info, "copy") else info for info in self.infos],
        )

    def copy_from(self, other: AtariSwarmState, mask: Tensor) -> None:
        """Copy full walker state (including env metadata) from another swarm."""
        super().copy_from(other, mask)
        if not mask.any():
            return
        indices = torch.where(mask)[0].tolist()
        for idx in indices:
            self.env_states[idx] = other.env_states[idx]
            self.observations[idx] = other.observations[idx]
            self.rewards[idx] = other.rewards[idx]
            self.step_rewards[idx] = other.step_rewards[idx]
            self.dones[idx] = other.dones[idx]
            self.truncated[idx] = other.truncated[idx]
            self.actions[idx] = other.actions[idx]
            self.dts[idx] = other.dts[idx]
            self.infos[idx] = (
                other.infos[idx].copy() if hasattr(other.infos[idx], "copy") else other.infos[idx]
            )


class AtariGasParams(BaseModel):
    """Configuration for Atari gas experiments."""

    model_config = {"arbitrary_types_allowed": True}

    N: int = Field(gt=0, description="Number of walkers")
    env: Any = Field(description="Initialized plangym environment")
    cloning: CloningParams = Field(description="Cloning operator parameters")
    device: str = Field("cpu", description="PyTorch device")
    dtype: str = Field("float32", description="PyTorch dtype")
    dt_range: tuple[int, int] | None = Field(
        (1, 4),
        description=(
            "Inclusive range for random dt sampling. dt represents the number of times each "
            "action is applied consecutively in the environment. Set to None to disable random "
            "dt and use dt=1 for all steps."
        ),
    )
    action_sampler: Callable[[int], Sequence[int]] | None = Field(
        default=None, description="Optional custom action sampler"
    )
    dt_sampler: Callable[[int], Sequence[int]] | None = Field(
        default=None,
        description=(
            "Optional custom dt sampler. Takes number of walkers as input and "
            "returns a sequence of dt values (number of times each action is "
            "applied consecutively in the environment)."
        ),
    )
    observation_transform: Callable[[Any], np.ndarray] | None = Field(
        default=None, description="Transform raw observations before tensor conversion"
    )
    step_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs forwarded to env.step_batch"
    )
    freeze_best: bool = Field(
        True,
        description=(
            "Keep the highest cumulative reward walker unperturbed across "
            "cloning and kinetic steps."
        ),
    )

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return corresponding torch dtype."""
        return torch.float64 if self.dtype == "float64" else torch.float32


class AtariCloningOperator:
    """Cloning operator that reuses Euclidean pairing logic for Atari states."""

    def __init__(self, params: CloningParams, device: torch.device, dtype: torch.dtype) -> None:
        self.params = params
        self.device = device
        self.dtype = dtype
        self.last_companions: Tensor | None = None
        self.last_distances: Tensor | None = None

    def apply(self, state: AtariSwarmState) -> AtariSwarmState:
        """Select cloning companions and replicate env-ready state."""
        dist_sq = VectorizedOps.algorithmic_distance_squared(state, self.params.lambda_alg)
        epsilon = max(float(self.params.sigma_x), 1e-8)
        companions = VectorizedOps.find_companions(dist_sq, epsilon=epsilon)

        self.last_companions = companions
        self.last_distances = dist_sq

        comp_idx = companions.detach().cpu().numpy()
        env_states = np.asarray(state.env_states[comp_idx], dtype=object)
        observations = state.observations[companions].clone()
        x_clone = observations.reshape(state.N, -1)

        if self.params.sigma_x > 0:
            jitter = torch.randn_like(x_clone, device=self.device) * self.params.sigma_x
            x_clone += jitter

        return AtariSwarmState(
            x=x_clone,
            v=state.v[companions].clone(),
            env_states=env_states,
            observations=observations,
            rewards=state.rewards[companions].clone(),
            step_rewards=state.step_rewards[companions].clone(),
            dones=state.dones[companions].clone(),
            truncated=state.truncated[companions].clone(),
            actions=state.actions[companions].clone(),
            dts=state.dts[companions].clone(),
            infos=[state.infos[i] for i in comp_idx],
        )


class RandomActionOperator:
    """Apply random actions (uniform policy) to a batch of Atari states."""

    def __init__(
        self,
        env: Any,
        device: torch.device,
        dtype: torch.dtype,
        action_sampler: Callable[[int], Sequence[int]] | None,
        dt_sampler: Callable[[int], Sequence[int]] | None,
        dt_range: tuple[int, int] | None,
        observation_transform: Callable[[Any], np.ndarray] | None,
        step_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.env = env
        self.device = device
        self.dtype = dtype
        self._action_sampler = action_sampler
        self._dt_sampler = dt_sampler
        self._dt_range = dt_range
        self._observation_transform = observation_transform
        self._step_kwargs = dict(step_kwargs or {})

    def transform_observations(self, observations: Sequence[Any]) -> Tensor:
        """Expose observation processing for initialization."""
        return self._transform_observations(observations)

    def apply(self, state: AtariSwarmState) -> AtariSwarmState:
        """Sample random actions, step env, and update swarm tensors."""
        actions = self._sample_actions(state.N)
        dts = self._sample_dt(state.N)
        step_data = self._step_environment(actions=actions, states=state.env_states, dts=dts)

        if len(step_data) == 5:
            new_states, observations, rewards, dones, infos = step_data
            truncated = [False] * len(dones)
        elif len(step_data) == 6:
            new_states, observations, rewards, dones, truncated, infos = step_data
        else:
            msg = f"Unexpected number of outputs from env.step_batch: {len(step_data)}"
            raise RuntimeError(msg)

        obs_tensor = self._transform_observations(observations)
        x_next = obs_tensor.reshape(state.N, -1)
        v_next = x_next - state.x

        step_rewards_tensor = torch.tensor(
            np.asarray(rewards, dtype=np.float32), dtype=self.dtype, device=self.device
        )
        cumulative_rewards = state.rewards + step_rewards_tensor
        dones_tensor = torch.tensor(
            np.asarray(dones, dtype=np.bool_), dtype=torch.bool, device=self.device
        )
        truncated_tensor = torch.tensor(
            np.asarray(truncated, dtype=np.bool_), dtype=torch.bool, device=self.device
        )
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        if dts is None:
            dts_tensor = torch.ones(state.N, dtype=torch.long, device=self.device)
        else:
            dts_tensor = torch.tensor(dts, dtype=torch.long, device=self.device)

        infos_list = list(infos)
        env_states = np.asarray(new_states, dtype=object)

        return AtariSwarmState(
            x=x_next,
            v=v_next,
            env_states=env_states,
            observations=obs_tensor,
            rewards=cumulative_rewards,
            step_rewards=step_rewards_tensor,
            dones=dones_tensor,
            truncated=truncated_tensor,
            actions=actions_tensor,
            dts=dts_tensor,
            infos=infos_list,
        )

    def _sample_actions(self, n_walkers: int) -> np.ndarray:
        if self._action_sampler is not None:
            actions = self._action_sampler(n_walkers)
        else:
            actions = [self._sample_action() for _ in range(n_walkers)]
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (n_walkers,):
            actions = actions.reshape(n_walkers)
        return actions

    def _sample_action(self) -> int:
        if hasattr(self.env, "sample_action"):
            return int(self.env.sample_action())
        if hasattr(self.env, "action_space"):
            return int(self.env.action_space.sample())
        msg = "Environment must expose sample_action or action_space.sample()"
        raise AttributeError(msg)

    def _sample_dt(self, n_walkers: int) -> np.ndarray | None:
        """Sample dt values (number of times to apply each action consecutively).

        Args:
            n_walkers: Number of dt values to sample.

        Returns:
            Array of dt values [n_walkers], or None if dt sampling is disabled.
            Each dt value specifies how many times the corresponding action is
            applied consecutively in the environment.
        """
        if self._dt_sampler is not None:
            dts = self._dt_sampler(n_walkers)
        elif self._dt_range is not None:
            low, high = self._dt_range
            dts = np.random.randint(low, high + 1, size=n_walkers, dtype=np.int64)
        else:
            return None
        dts = np.asarray(dts, dtype=np.int64)
        if dts.shape == ():
            dts = np.full(n_walkers, int(dts), dtype=np.int64)
        return dts

    def _transform_observations(self, observations: Sequence[Any]) -> Tensor:
        if self._observation_transform is not None:
            processed = [self._observation_transform(obs) for obs in observations]
        else:
            processed = observations
        obs_array = np.asarray(processed)
        if obs_array.dtype == np.object_:
            obs_array = np.stack(processed)
        return torch.tensor(obs_array, dtype=self.dtype, device=self.device)

    def _step_environment(
        self,
        *,
        actions: np.ndarray,
        states: np.ndarray,
        dts: np.ndarray | None,
    ) -> Sequence[Any]:
        """Step the environment with actions, optionally repeating each action dt times.

        The dt parameter controls how many times each action is applied consecutively
        in the environment before returning. This is passed to plangym's step/step_batch
        methods via the 'dt' keyword argument.
        """
        kwargs = dict(self._step_kwargs)
        kwargs.update({"actions": actions, "states": states})
        if dts is not None:
            # dt tells plangym how many times to apply each action consecutively
            kwargs["dt"] = dts

        if hasattr(self.env, "step_batch"):
            return self.env.step_batch(**kwargs)
        return self._step_sequential(**kwargs)

    def _step_sequential(
        self,
        *,
        actions: Sequence[int],
        states: Sequence[Any],
        dt: np.ndarray | None = None,
    ) -> Sequence[Any]:
        results = []
        for idx, action in enumerate(actions):
            state = None if states is None else states[idx]
            dt_i = int(dt[idx]) if dt is not None else 1
            data = self.env.step(action=action, state=state, dt=dt_i, return_state=True)
            results.append(data)
        # Transpose list of tuples -> tuple of lists
        return [list(x) for x in zip(*results)]


class AtariGas:
    """Reimplementation of the Euclidean gas for Atari environments."""

    def __init__(self, params: AtariGasParams) -> None:
        self.params = params
        self.device = torch.device(params.device)
        self.dtype = params.torch_dtype
        self.env = params.env

        self.cloning_op = AtariCloningOperator(params.cloning, self.device, self.dtype)
        self.kinetic_op = RandomActionOperator(
            env=self.env,
            device=self.device,
            dtype=self.dtype,
            action_sampler=params.action_sampler,
            dt_sampler=params.dt_sampler,
            dt_range=params.dt_range,
            observation_transform=params.observation_transform,
            step_kwargs=params.step_kwargs,
        )
        self.freeze_best = params.freeze_best

    def _freeze_mask(self, state: AtariSwarmState) -> Tensor | None:
        if not self.freeze_best:
            return None
        rewards = state.rewards
        if rewards.numel() == 0:
            return None
        best_idx = torch.argmax(rewards)
        mask = torch.zeros(state.N, dtype=torch.bool, device=rewards.device)
        mask[best_idx] = True
        return mask

    def should_terminate(self, state: AtariSwarmState) -> tuple[bool, str]:
        """Check if the algorithm should terminate early.

        Returns:
            Tuple of (should_stop, reason):
            - should_stop: True if algorithm should terminate
            - reason: Human-readable explanation for termination
        """
        # Check if all walkers are dead
        if state.dones.all():
            return True, "All walkers are dead"

        # When freeze_best is enabled, check if only the frozen walker is alive
        if self.freeze_best:
            n_alive = (~state.dones).sum().item()
            if n_alive == 1:
                best_idx = torch.argmax(state.rewards)
                if not state.dones[best_idx]:
                    return True, "Only frozen best walker remains alive"

        return False, ""

    def initialize_state(self) -> AtariSwarmState:
        """Reset environment and build the initial swarm."""
        base_state, observation, info = self.env.reset()
        env_states = np.asarray(
            [self._copy_state(base_state) for _ in range(self.params.N)], dtype=object
        )
        observations = self.kinetic_op.transform_observations([
            self._copy_observation(observation) for _ in range(self.params.N)
        ])
        x_init = observations.reshape(self.params.N, -1)
        v_init = torch.zeros_like(x_init, device=self.device, dtype=self.dtype)
        rewards = torch.zeros(self.params.N, device=self.device, dtype=self.dtype)
        step_rewards = torch.zeros(self.params.N, device=self.device, dtype=self.dtype)
        dones = torch.zeros(self.params.N, device=self.device, dtype=torch.bool)
        truncated = torch.zeros(self.params.N, device=self.device, dtype=torch.bool)
        actions = torch.full((self.params.N,), fill_value=-1, device=self.device, dtype=torch.long)
        dts = torch.ones(self.params.N, device=self.device, dtype=torch.long)

        infos = [info.copy() if hasattr(info, "copy") else info for _ in range(self.params.N)]

        return AtariSwarmState(
            x=x_init,
            v=v_init,
            env_states=env_states,
            observations=observations,
            rewards=rewards,
            step_rewards=step_rewards,
            dones=dones,
            truncated=truncated,
            actions=actions,
            dts=dts,
            infos=infos,
        )

    def step(self, state: AtariSwarmState) -> tuple[AtariSwarmState, AtariSwarmState, Tensor]:
        """Run one cloning + random action iteration.

        Returns:
            Tuple of (cloned_state, next_state, companions):
            - cloned_state: State after cloning operator
            - next_state: State after kinetic operator (random actions)
            - companions: Companion indices from cloning [N]

        Note: Use should_terminate(next_state) to check if algorithm should stop.
        """
        freeze_mask = self._freeze_mask(state)
        reference_state = state.clone() if freeze_mask is not None else None

        cloned_state = self.cloning_op.apply(state)
        companions = self.cloning_op.last_companions
        if companions is None:
            msg = "Cloning operator did not produce companion indices."
            raise RuntimeError(msg)
        if freeze_mask is not None and freeze_mask.any():
            cloned_state.copy_from(reference_state, freeze_mask)
        next_state = self.kinetic_op.apply(cloned_state)
        if freeze_mask is not None and freeze_mask.any():
            next_state.copy_from(reference_state, freeze_mask)
        return cloned_state, next_state, companions

    def run(self, n_steps: int, state: AtariSwarmState | None = None) -> dict[str, Tensor]:
        """Execute multiple iterations, tracking Atari-specific telemetry.

        The run terminates early if:
        - All walkers are dead
        - Only the frozen best walker remains alive (when freeze_best=True)

        Returns:
            Dictionary containing trajectories and metrics. Note that arrays may contain
            fewer than n_steps if early termination occurred. Check 'actual_steps' to
            see how many steps were actually executed.
        """
        state = state if state is not None else self.initialize_state()
        N = state.N
        d = state.x.shape[1]
        obs_shape = state.obs_shape

        x_traj = torch.zeros(n_steps + 1, N, d, device=self.device, dtype=self.dtype)
        obs_traj = torch.zeros((n_steps + 1, N, *obs_shape), device=self.device, dtype=self.dtype)
        cumulative_rewards = torch.zeros(n_steps, N, device=self.device, dtype=self.dtype)
        step_rewards = torch.zeros(n_steps, N, device=self.device, dtype=self.dtype)
        actions = torch.zeros(n_steps, N, device=self.device, dtype=torch.long)
        dones = torch.zeros(n_steps, N, device=self.device, dtype=torch.bool)
        truncated = torch.zeros(n_steps, N, device=self.device, dtype=torch.bool)
        dts = torch.zeros(n_steps, N, device=self.device, dtype=torch.long)
        companions = torch.zeros(n_steps, N, device=self.device, dtype=torch.long)

        x_traj[0] = state.x
        obs_traj[0] = state.observations

        actual_steps = 0
        termination_reason = ""

        for t in range(n_steps):
            _cloned_state, state, companion_idx = self.step(state)
            x_traj[t + 1] = state.x
            obs_traj[t + 1] = state.observations
            cumulative_rewards[t] = state.rewards
            step_rewards[t] = state.step_rewards
            actions[t] = state.actions
            dones[t] = state.dones
            truncated[t] = state.truncated
            dts[t] = state.dts
            companions[t] = companion_idx

            actual_steps = t + 1

            # Check for early termination
            should_stop, reason = self.should_terminate(state)
            if should_stop:
                termination_reason = reason
                break

        # Trim arrays to actual steps taken
        return {
            "x": x_traj[: actual_steps + 1],
            "observations": obs_traj[: actual_steps + 1],
            "rewards": cumulative_rewards[:actual_steps],
            "step_rewards": step_rewards[:actual_steps],
            "actions": actions[:actual_steps],
            "dones": dones[:actual_steps],
            "truncated": truncated[:actual_steps],
            "dts": dts[:actual_steps],
            "companions": companions[:actual_steps],
            "actual_steps": torch.tensor(actual_steps, dtype=torch.long),
            "terminated_early": torch.tensor(actual_steps < n_steps, dtype=torch.bool),
            "termination_reason": termination_reason,
        }

    @staticmethod
    def _copy_state(state: Any) -> Any:
        return state.copy() if hasattr(state, "copy") else state

    @staticmethod
    def _copy_observation(observation: Any) -> Any:
        return observation.copy() if hasattr(observation, "copy") else observation
