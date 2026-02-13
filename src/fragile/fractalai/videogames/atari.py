"""Lightweight gymnasium Atari wrapper with state cloning support."""

from dataclasses import dataclass

# Register ALE environments
import ale_py  # noqa: F401
import gymnasium as gym
import numpy as np


@dataclass
class AtariState:
    """Container for Atari environment state.

    Stores ALE state for cloning/restoring, observation, and optional RGB frame.

    Attributes:
        ale_state: ALE state object from ale.cloneState()
        observation: Current observation (RAM or image depending on obs_type)
        rgb_frame: RGB frame for visualization (optional)
    """

    ale_state: object  # ALEState from ale.cloneState()
    observation: np.ndarray
    rgb_frame: np.ndarray | None = None

    def copy(self) -> "AtariState":
        """Create a copy of this state.

        Note: ALEState objects are immutable, so we share the reference.
        """
        return AtariState(
            ale_state=self.ale_state,
            observation=self.observation.copy(),
            rgb_frame=self.rgb_frame.copy() if self.rgb_frame is not None else None,
        )


class AtariEnv:
    """Lightweight gymnasium Atari wrapper with state cloning support.

    Provides the interface expected by AtariFractalGas:
    - reset() -> (state, observation, info)
    - step_batch(states, actions, dt) -> (new_states, observations, rewards, dones, truncated, infos)
    - step(action, state, dt, return_state) -> (state, observation, reward, done, truncated, info)
    - action_space.sample() for random action sampling

    State management uses ALE's cloneState/restoreState for efficient checkpointing.

    Args:
        name: Environment name (e.g., "ALE/Pong-v5")
        obs_type: Observation type - "ram", "rgb", or "grayscale"
        render_mode: Gymnasium render mode (default "rgb_array" for headless)
        include_rgb: Always capture RGB frames for visualization (default True)
        **kwargs: Additional arguments passed to gymnasium.make()
    """

    def __init__(
        self,
        name: str,
        obs_type: str = "ram",
        render_mode: str = "rgb_array",
        include_rgb: bool = True,
        **kwargs,
    ):
        self.name = name
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.include_rgb = include_rgb

        # Create gymnasium environment
        self.env = gym.make(name, obs_type=obs_type, render_mode=render_mode, **kwargs)
        self.action_space = self.env.action_space

        # Access underlying ALE for state cloning
        self._ale = self.env.unwrapped.ale

    def _get_observation(self) -> np.ndarray:
        """Get current observation based on obs_type."""
        if self.obs_type == "ram":
            return self._ale.getRAM()
        if self.obs_type == "rgb":
            return self._ale.getScreenRGB()
        if self.obs_type == "grayscale":
            return self._ale.getScreenGrayscale()
        raise ValueError(f"Unknown obs_type: {self.obs_type}")

    def _get_rgb_frame(self) -> np.ndarray | None:
        """Get RGB frame for visualization if enabled."""
        if self.include_rgb:
            return self._ale.getScreenRGB()
        return None

    def _clone_state(self, observation: np.ndarray | None = None) -> AtariState:
        """Clone current environment state.

        Args:
            observation: Pre-computed observation (avoids redundant computation)

        Returns:
            AtariState containing ALE state, observation, and optional RGB frame
        """
        ale_state = self._ale.cloneState(include_rng=True)
        if observation is None:
            observation = self._get_observation()
        rgb_frame = self._get_rgb_frame()
        return AtariState(
            ale_state=ale_state,
            observation=observation,
            rgb_frame=rgb_frame,
        )

    def _restore_state(self, state: AtariState) -> None:
        """Restore environment to a previously cloned state.

        Args:
            state: AtariState to restore
        """
        self._ale.restoreState(state.ale_state)

    def reset(self, seed: int | None = None, **kwargs) -> AtariState:
        """Reset environment and return initial state.

        Compatible with plangym interface: returns just the state.
        Use get_state()["obs"] to get the observation.

        Args:
            seed: Optional random seed

        Returns:
            state: Initial AtariState (with .copy() method)
        """
        obs, _info = self.env.reset(seed=seed)
        return self._clone_state(observation=obs)

    def step(
        self,
        action: int,
        state: AtariState | None = None,
        dt: int = 1,
        return_state: bool = True,
    ) -> tuple[AtariState | None, np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: Action to take
            state: State to restore before stepping (optional)
            dt: Number of times to repeat the action (frame skip)
            return_state: Whether to return the new state

        Returns:
            new_state: New AtariState (or None if return_state=False)
            observation: Observation after action
            reward: Accumulated reward across dt frames
            done: Terminal flag (OR'd across frames)
            truncated: Truncation flag (OR'd across frames)
            info: Info dictionary from last frame
        """
        # Restore state if provided
        if state is not None:
            self._restore_state(state)

        # Apply action dt times, accumulating rewards
        total_reward = 0.0
        done = False
        truncated = False
        info = {}

        for _ in range(dt):
            obs, reward, step_done, step_truncated, info = self.env.step(action)
            total_reward += reward
            done = done or step_done
            truncated = truncated or step_truncated
            if done or truncated:
                break

        # Clone state if requested
        new_state = self._clone_state(observation=obs) if return_state else None

        return new_state, obs, total_reward, done, truncated, info

    def step_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        dt: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Step multiple walkers sequentially.

        ALE is single-threaded, so we process walkers one at a time.

        Args:
            states: Array of AtariState objects [N], dtype=object
            actions: Array of actions [N]
            dt: Array of frame skip values [N] (default: all 1s)

        Returns:
            new_states: Array of new AtariState objects [N], dtype=object
            observations: Array of observations [N, ...]
            rewards: Array of rewards [N]
            dones: Array of terminal flags [N]
            truncated: Array of truncation flags [N]
            infos: List of info dicts [N]
        """
        N = len(states)

        if dt is None:
            dt = np.ones(N, dtype=int)

        # Pre-allocate outputs
        new_states = np.empty(N, dtype=object)
        observations = []
        rewards = np.zeros(N, dtype=np.float32)
        dones = np.zeros(N, dtype=bool)
        truncateds = np.zeros(N, dtype=bool)
        infos = []

        for i in range(N):
            new_state, obs, reward, done, trunc, info = self.step(
                action=int(actions[i]),
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

        # Stack observations
        observations = np.array(observations)

        return new_states, observations, rewards, dones, truncateds, infos

    def render(self) -> np.ndarray | None:
        """Render current frame.

        Returns:
            RGB array if render_mode is 'rgb_array', else None
        """
        return self.env.render()

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

    def clone_state(self) -> AtariState:
        """Clone current environment state (convenience method).

        Returns:
            Current AtariState
        """
        return self._clone_state()

    def restore_state(self, state: AtariState) -> None:
        """Restore environment state (convenience method).

        Args:
            state: AtariState to restore
        """
        self._restore_state(state)

    def set_state(self, state: AtariState) -> None:
        """Alias for restore_state (for compatibility).

        Args:
            state: AtariState to restore
        """
        self._restore_state(state)

    def get_state(self) -> dict:
        """Get current state as dictionary (for compatibility).

        Returns:
            Dict with 'state' (AtariState) and 'obs' (observation)
        """
        state = self._clone_state()
        return {"state": state, "obs": state.observation}
