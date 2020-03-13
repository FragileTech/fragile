import copy

import numpy as np
from plangym.env import Environment as PlangymEnv

from fragile.core.base_classes import BaseEnvironment
from fragile.core.states import StateDict, StatesEnv, StatesModel


class Environment(BaseEnvironment):
    """
    The Environment is in charge of stepping the walkers, acting as an state \
    transition function.

    For every different problem a new :class:`Environment` needs to be implemented \
    following the :class:`BaseEnvironment` interface.
    """

    def __init__(self, states_shape: tuple, observs_shape: tuple):
        """
        Initialize an :class:`Environment`.

        Args:
            states_shape: Shape of the internal state of the :class:`Environment`
            observs_shape: Shape of the observations state of the :class:`Environment`.
        """
        self._states_shape = states_shape
        self._observs_shape = observs_shape

    @property
    def states_shape(self) -> tuple:
        """Return the shape of the internal state of the :class:`Environment`."""
        return self._states_shape

    @property
    def observs_shape(self) -> tuple:
        """Return the shape of the observations state of the :class:`Environment`."""
        return self._observs_shape

    def get_params_dict(self) -> StateDict:
        """Return a dictionary containing the param_dict to build an instance \
        of :class:`StatesEnv` that can handle all the data generated by a \
        :class:`Environment`.
        """
        params = {
            "states": {"size": self.states_shape, "dtype": np.int64},
            "observs": {"size": self.observs_shape, "dtype": np.float32},
            "rewards": {"dtype": np.float32},
            "ends": {"dtype": np.bool_},
        }
        return params

    def states_from_data(self, batch_size, states, observs, rewards, ends, **kwargs) -> StatesEnv:
        """Return a new :class:`StatesEnv` object containing the data generated \
        by the environment."""
        ends = np.array(ends, dtype=np.bool_)
        rewards = np.array(rewards, dtype=np.float32)
        observs = np.array(observs)
        states = np.array(states)
        state = super(Environment, self).states_from_data(
            batch_size=batch_size,
            states=states,
            observs=observs,
            rewards=rewards,
            ends=ends,
            **kwargs
        )
        custom_death = self.calculate_custom_death(state)
        state.ends = np.logical_or(state.ends, custom_death)
        return state


class DiscreteEnv(Environment):
    """The DiscreteEnv acts as an interface with `plangym` discrete actions.

    It can interact with any environment that accepts discrete actions and \
    follows the interface of `plangym`.
    """

    def __init__(self, env: PlangymEnv):
        """
        Initialize a :class:`DiscreteEnv`.

        Args:
           env: Instance of :class:`plangym.Environment`.
        """
        self._env = env
        self._n_actions = self._env.action_space.n
        super(DiscreteEnv, self).__init__(
            states_shape=self._env.get_state().shape,
            observs_shape=self._env.observation_space.shape,
        )

    @property
    def n_actions(self) -> int:
        """Return the number of different discrete actions that can be taken in the environment."""
        return self._n_actions

    # @profile
    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """
        Set the environment to the target states by applying the specified \
        actions an arbitrary number of time steps.

        Args:
            model_states: States representing the data to be used to act on the environment..
            env_states: States representing the data to be set in the environment.

        Returns:
            States containing the information that describes the new state of the Environment.

        """
        actions = model_states.actions.astype(np.int32)
        n_repeat_actions = model_states.dt if hasattr(model_states, "dt") else 1
        new_states, observs, rewards, ends, infos = self._env.step_batch(
            actions=actions, states=env_states.states, n_repeat_action=n_repeat_actions
        )

        new_state = self.states_from_data(len(actions), new_states, observs, rewards, ends)
        return new_state

    # @profile
    def reset(self, batch_size: int = 1, **kwargs) -> StatesEnv:
        """
        Reset the environment to the start of a new episode and returns a new \
        States instance describing the state of the Environment.

        Args:
            batch_size: Number of walkers that the returned state will have.
            **kwargs: Ignored. This environment resets without using any external data.

        Returns:
            States instance describing the state of the Environment. The first \
            dimension of the data tensors (number of walkers) will be equal to \
            batch_size.

        """
        state, obs = self._env.reset()
        states = np.array([copy.deepcopy(state) for _ in range(batch_size)])
        observs = np.array([copy.deepcopy(obs) for _ in range(batch_size)])
        rewards = np.zeros(batch_size, dtype=np.float32)
        ends = np.zeros(batch_size, dtype=np.uint8)
        new_states = self.states_from_data(batch_size, states, observs, rewards, ends)
        return new_states
