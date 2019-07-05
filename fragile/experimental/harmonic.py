from typing import Tuple

import numpy as np

from fragile.core.base_classes import BaseEnvironment, BaseModel, BaseStates
from fragile.core.states import States


class HarmonicOscillator(BaseEnvironment):
    def __init__(
        self, n_dims: int = 1, k: float = 1.0, m: float = 1.0, length: float = 1.0, init_state=None
    ):
        self._n_dims = n_dims
        self.k = k
        self.m = m
        self.length = length
        self.init_state = init_state

    @property
    def n_actions(self):
        return self._n_dims

    def get_params_dict(self) -> dict:
        """Return a dictionary containing the param_dict to build an instance
        of States that can handle all the data generated by the environment.

        The size of the observations corresponds to the state space of position and momentum of
        the particle in each dimension.
        """
        params = {
            "states": {"size": tuple([self.n_actions * 2 + 4]), "dtype": np.int64},
            "observs": {"size": tuple([self.n_actions * 2 + 4]), "dtype": np.float32},
            "rewards": {"dtype": np.float32},
            "ends": {"dtype": np.bool_},
        }
        return params

    def step(
        self,
        actions: np.ndarray,
        env_states: BaseStates,
        n_repeat_action: [int, np.ndarray] = 1,
        *args,
        **kwargs
    ) -> BaseStates:
        """
        Sets the environment to the target states by applying the specified actions an arbitrary
        number of time steps.

        Args:
            actions: Vector containing the actions that will be applied to the target states.
            env_states: BaseStates class containing the state data to be set on the Environment.
            n_repeat_action: Number of times that an action will be applied. If it is an array
                it corresponds to the different dts of each walker.
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            States containing the information that describes the new state of the Environment.
        """

        new_states, observs, rewards, ends = self._step_harmonic_oscillator(
            actions=actions, states=env_states.states, n_repeat_action=n_repeat_action
        )

        new_state = self._get_new_states(new_states, observs, rewards, ends, len(actions))
        return new_state

    def _step_harmonic_oscillator(self, actions, states, n_repeat_action: float = 0.1) -> tuple:
        old_pos = states[:, : self.n_actions].reshape(-1, self.n_actions)
        old_velocity = states[:, self.n_actions : self.n_actions * 2].reshape(-1, self.n_actions)
        old_delta_energy = states[:, -4]
        old_action = states[:, -3]

        assert old_pos.shape == old_velocity.shape, (old_pos.shape, old_velocity.shape)
        assert old_pos.shape[1] == self.n_actions
        assert old_pos.shape == actions.shape

        old_pot_e = 0.5 * self.k * (old_pos ** 2).sum(axis=1)
        old_kin_e = (old_velocity ** 2).sum(axis=1) * self.m

        assert old_pot_e.shape == old_kin_e.shape, (old_pot_e.shape, old_kin_e.shape)

        old_energy = old_pot_e + old_kin_e

        assert old_pot_e.shape[0] == actions.shape[0]
        assert old_kin_e.shape[0] == actions.shape[0]
        assert old_energy.shape[0] == actions.shape[0]

        actions = -self.k * old_pos + actions
        new_velocity = old_velocity + actions * n_repeat_action
        new_position = (
            old_pos.reshape(-1, self.n_actions)
            + old_velocity.reshape(-1, self.n_actions) * n_repeat_action
            + 0.5 * actions.reshape(-1, self.n_actions) * (n_repeat_action ** 2)
        )
        new_position = new_position
        # .view(-1, self.n_actions)
        delta_pos = new_position - old_pos

        new_pot_e = 0.5 * self.k * (new_position ** 2).sum(axis=1)
        new_kin_e = (new_velocity ** 2).sum(axis=1) * self.m
        new_energy = new_kin_e + new_pot_e

        delta_potential_e = new_pot_e - old_pot_e
        delta_kinetic_e = new_kin_e - old_kin_e
        delta_energy = new_energy - old_energy

        assert delta_energy.shape[0] == actions.shape[0]

        assert new_position.shape[0] == actions.shape[0], new_position.shape

        assert new_position.shape == old_pos.shape
        assert new_velocity.shape == old_velocity.shape

        ends = delta_energy > 3
        action = new_kin_e - old_kin_e

        states[:, : self.n_actions] = new_position
        states[:, self.n_actions : self.n_actions * 2] = new_velocity

        states[:, -4] = delta_energy.flatten() + old_delta_energy.flatten()
        states[:, -3] = action.flatten() + old_action.flatten()
        states[:, -2] = new_pot_e.flatten()
        states[:, -1] = new_kin_e.flatten()

        rewards = -(old_action.flatten() + action.flatten()) - abs(
            delta_energy.flatten() + old_delta_energy.flatten()
        )

        return states, states.copy(), rewards, ends

    # @profile
    def reset(self, batch_size: int = 1, states=None) -> BaseStates:
        """
        Resets the environment to the start of a new episode and returns an
        States instance describing the state of the Environment.
        Args:
            batch_size: Number of walkers that the returned state will have.
            states: Ignored.

        Returns:
            States instance describing the state of the Environment. The first
            dimension of the data tensors (number of walkers) will be equal to
            batch_size.
        """
        init_val = (
            self.init_state if self.init_state is not None else np.zeros(self.n_actions * 2 + 4)
        )

        states = np.vstack([init_val.copy() for _ in range(batch_size)])
        observs = states.copy()
        rewards = np.zeros(batch_size, dtype=np.float32)
        ends = np.zeros(batch_size, dtype=np.uint8)
        new_states = self._get_new_states(states, observs, rewards, ends, batch_size)
        return new_states

    # @profile
    def _get_new_states(self, states, observs, rewards, ends, batch_size) -> BaseStates:
        state = States(state_dict=self.get_params_dict(), batch_size=batch_size)
        state.update(states=states, observs=observs, rewards=rewards, ends=ends)
        return state


class GausianPerturbator(BaseModel):
    def __init__(
        self,
        n_actions: int,
        dt: float = 0.01,
        scale: float = 1.0,
        loc=0.0,
        max_jump: float = 3.0,
        *args,
        **kwargs
    ):
        super(GausianPerturbator, self).__init__(*args, **kwargs)
        self._n_actions = n_actions
        self.dt = dt
        self.max_jump = max_jump
        self.dist = lambda x: np.random.normal(size=x, loc=loc, scale=scale)

    def get_params_dict(self) -> dict:
        params = {
            "actions": {"size": tuple([self._n_actions]), "dtype": np.float32},
            "init_actions": {"size": tuple([self._n_actions]), "dtype": np.float32},
            "dt": {"size": tuple([self._n_actions]), "dtype": np.float32},
        }
        return params

    @property
    def n_actions(self):
        return self._n_actions

    def reset(self, batch_size: int = 1, *args, **kwargs) -> Tuple[np.ndarray, BaseStates]:
        """

        Args:
            batch_size:
            *args:
            **kwargs:

        Returns:

        """

        model_states = States(state_dict=self.get_params_dict(), batch_size=batch_size)
        actions = self.dist((batch_size, self.n_actions))
        actions = np.clip(actions, -self.max_jump, self.max_jump)
        model_states.update(
            dt=np.ones(batch_size) * self.dt, actions=actions, init_actions=actions
        )
        return actions, model_states

    def predict(
        self,
        env_states: BaseStates = None,
        batch_size: int = None,
        model_states: BaseStates = None,
    ) -> Tuple:
        """

        Args:
            env_states:
            batch_size:
            model_states:

        Returns:

        """
        if batch_size is None and env_states is None:
            raise ValueError("env_states and batch_size cannot be both None.")
        size = len(env_states.rewards) if env_states is not None else batch_size
        actions = self.dist((size, self.n_actions))
        actions = np.clip(actions, -self.max_jump, self.max_jump)
        return actions.reshape((size, self.n_actions)), model_states

    def calculate_dt(self, model_states: BaseStates, env_states: BaseStates) -> Tuple:
        """

        Args:
            model_states:
            env_states:

        Returns:

        """
        n_walkers = len(env_states.rewards)
        dt = np.ones(n_walkers) * self.dt
        model_states.update(dt=dt)
        return dt, model_states