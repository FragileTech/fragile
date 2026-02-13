"""Fractal gas algorithm for DM Control environments with uniform companion selection."""

from typing import Any

import numpy as np

from fragile.fractalai.fractal_gas import FractalGas, WalkerState  # noqa: F401


class RoboticFractalGas(FractalGas):
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

    DEFAULT_DT_RANGE = (1, 1)

    def _init_actions(self) -> np.ndarray:
        action_shape = self.env.action_space.shape
        return np.zeros((self.N, *action_shape), dtype=np.float64)

    def _render_walker_frame(self, state) -> np.ndarray:
        """Render visual frame for a DM Control walker state.

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

    def _handle_non_tuple_reset(self, reset_data) -> tuple[Any, Any, dict]:
        """Handle DMControlState returned directly from env.reset()."""
        state = reset_data
        observation = state.observation if hasattr(state, "observation") else None
        info: dict = {}
        if observation is None:
            observation = self._extract_observation_from_env()
        return state, observation, info

    def _extract_observation_from_env(self) -> Any | None:
        """Try reading the current observation from env.get_state()."""
        if not hasattr(self.env, "get_state"):
            return None
        state_data = self.env.get_state()
        if isinstance(state_data, dict):
            return state_data.get("obs")
        if hasattr(state_data, "observation"):
            return state_data.observation
        return None

    def run_with_tree(
        self,
        max_iterations: int = 1000,
        stop_when_all_dead: bool = False,
        task_name: str = "",
        task_label: str = "",
        initial_state: tuple | None = None,
    ):
        """Run with tree history recording.

        Args:
            max_iterations: Maximum number of iterations to run.
            stop_when_all_dead: If True, stop when all walkers are dead.
            task_name: Backward-compatible alias for *task_label*.
            task_label: Name stored in the tree metadata.
            initial_state: Optional ``(env_state, observation, info)`` tuple.

        Returns:
            AtariTreeHistory with the full run recorded.
        """
        return super().run_with_tree(
            max_iterations=max_iterations,
            stop_when_all_dead=stop_when_all_dead,
            task_label=task_name or task_label,
            initial_state=initial_state,
        )
