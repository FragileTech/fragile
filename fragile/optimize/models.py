from typing import Tuple

import numpy as np
import torch

from fragile.core.base_classes import BaseEnvironment, BaseStates
from fragile.core.models import RandomContinous
from fragile.core.states import States
from fragile.core.utils import device, relativize_np, to_numpy, to_tensor
from fragile.optimize.encoder import Encoder
from fragile.optimize.env import Function


class UnitaryContinuous(RandomContinous):
    def sample(self, batch_size: int = 1):
        val = super(UnitaryContinuous, self).sample(batch_size=batch_size)
        axis = 1 if len(val.shape) <= 2 else tuple(range(1, len(val.shape)))
        norm = np.linalg.norm(val, axis=axis)
        div = norm.reshape(-1, 1) if axis == 1 else np.expand_dims(np.expand_dims(norm, 1), 1)
        return val / div


class RandomNormal(RandomContinous):
    def __init__(self, env: Function = None, loc: float = 0, scale: float = 1, *args, **kwargs):
        kwargs["shape"] = kwargs.get(
            "shape", env.shape if isinstance(env, BaseEnvironment) else None
        )
        try:
            super(RandomNormal, self).__init__(env=env, *args, **kwargs)
        except Exception as e:
            print(args, kwargs)
            raise e
        self._shape = self.bounds.shape
        self._n_dims = self.bounds.shape
        self.loc = loc
        self.scale = scale

    def sample(self, batch_size: int = 1, loc: float = None, scale: float = None):
        loc = self.loc if loc is None else loc
        scale = self.scale if scale is None else scale
        high = (
            self.bounds.high
            if self.bounds.dtype.kind == "f"
            else self.bounds.high.astype("int64") + 1
        )
        data = np.clip(
            self.np_random.normal(
                size=tuple([batch_size]) + self.shape, loc=loc, scale=scale
            ).astype(self.bounds.dtype),
            self.bounds.low,
            high,
        )
        return to_tensor(data, device=device, dtype=torch.float32)

    def calculate_dt(
        self, model_states: BaseStates, env_states: BaseStates
    ) -> Tuple[np.ndarray, BaseStates]:
        """

        Args:
            model_states:
            env_states:

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.
        """
        dt = np.ones(shape=tuple(env_states.rewards.shape)) * self.mean_dt
        dt = np.clip(dt, self.min_dt, self.max_dt)
        dt = to_tensor(dt, device=device, dtype=torch.float32).reshape(-1, 1)
        model_states.update(dt=dt)
        return dt, model_states

    def reset(self, batch_size: int = 1, *args, **kwargs) -> Tuple[np.ndarray, BaseStates]:
        """

        Args:
            batch_size:
            *args:
            **kwargs:

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.
        """

        model_states = States(state_dict=self.get_params_dict(), n_walkers=batch_size)
        actions = super(RandomNormal, self).sample(batch_size=batch_size)
        model_states.update(dt=np.ones(batch_size), actions=actions, init_actions=actions)
        return actions, model_states


class EncoderSampler(RandomNormal):
    def __init__(self, env: Function = None, walkers: "MapperWalkers" = None, *args, **kwargs):
        kwargs["shape"] = kwargs.get(
            "shape", env.shape if isinstance(env, BaseEnvironment) else None
        )
        try:
            super(EncoderSampler, self).__init__(env=env, *args, **kwargs)
        except Exception as e:
            print(args, kwargs)
            raise e
        self._shape = self.bounds.shape
        self._n_dims = self.bounds.shape
        self._walkers = walkers
        self.bases = None

    @property
    def encoder(self):
        return self._walkers.encoder

    @property
    def walkers(self):
        return self._walkers

    def set_walkers(self, encoder: Encoder):
        self._walkers = encoder

    def sample(self, batch_size: int = 1):
        if self.encoder is None:
            raise ValueError("You must first set the encoder before calling sample()")
        if len(self.encoder) <= 5:
            return RandomNormal.sample(self, batch_size=batch_size)
        data = self._sample_encoder(batch_size)
        return to_tensor(data, device=device, dtype=torch.float32)

    def _sample_encoder(self, batch_size: int = 1):
        samples = to_tensor(
            super(EncoderSampler, self).sample(batch_size=batch_size, loc=0.0, scale=1.0),
            device=device,
            dtype=torch.float32,
        )
        self.bases = self.encoder.get_bases()
        perturbation = torch.abs(self.bases.mean(0)) * samples  # (samples - self.mean_dt) * 0.01 /
        # self.std_dt
        return perturbation / 2

    def calculate_dt(
        self, model_states: BaseStates, env_states: BaseStates
    ) -> Tuple[np.ndarray, BaseStates]:
        """

        Args:
            model_states:
            env_states:

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.
        """
        dt = np.ones(shape=tuple(env_states.rewards.shape))
        # * self.mean_dt
        model_states.update(dt=dt)
        return dt, model_states


class BestDtEncoderSamper(EncoderSampler):
    def calculate_dt(
        self, model_states: BaseStates, env_states: BaseStates
    ) -> Tuple[np.ndarray, BaseStates]:
        """

        Args:
            model_states:
            env_states:

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.
        """
        if True:  # self.bases is None:
            return super(BestDtEncoderSamper, self).calculate_dt(
                model_states=model_states, env_states=env_states
            )
        best = self.walkers.best_found
        dist = to_numpy(torch.sqrt((self.walkers.observs - best) ** 2))
        max_mod = torch.abs(self.bases.max(0))

        dist = relativize_np(dist)
        # dist = (dist - dist.mean()) / dist.std()
        dt = np.ones(shape=tuple(env_states.rewards.shape)) * self.map_range(
            dist, max_=to_numpy(max_mod)
        )

        # * self.mean_dt
        model_states.update(dt=dt)
        return dt, model_states

    @staticmethod
    def map_range(x, max_: float = 1, min_: float = 0.0):
        normed = (x - x.min()) / (x.max() - x.min())
        return normed * (max_ - min_) + min_
