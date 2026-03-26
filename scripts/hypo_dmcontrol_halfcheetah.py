#!/usr/bin/env python3
"""Gradient-free hypostructure RL for dm_control cheetah/run or Gymnasium HalfCheetah.

This script adapts a fixed-reservoir / exact-readout style learner to continuous control.
The recurrent core stays fixed; learning is performed with:

1. exact ridge value fitting;
2. reward-weighted regression (RWR) for a Gaussian action policy;
3. optional action-atlas charts (mixture of local linear controllers);
4. hypostructure-style defect penalties and optional response-signature block search.

The code is intentionally self-contained and conservative about dependencies.

Examples
--------
# dm_control cheetah/run (preferred if dm_control is installed)
python hypo_dmcontrol_halfcheetah.py \
  --backend dm_control --dm-domain cheetah --dm-task run \
  --iterations 12 --train-episodes 6 --eval-episodes 3 --horizon 250

# Gymnasium HalfCheetah fallback
python hypo_dmcontrol_halfcheetah.py \
  --backend gymnasium --gym-env HalfCheetah-v5 \
  --iterations 12 --train-episodes 6 --eval-episodes 3 --horizon 250
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Environment adapters
# -----------------------------------------------------------------------------


def try_import_dm_control() -> Any:
    try:
        from dm_control import suite  # type: ignore

        return suite
    except Exception:
        return None


def try_import_gymnasium() -> Any:
    try:
        import gymnasium as gym  # type: ignore

        return gym
    except Exception:
        return None


class EnvAdapter:
    """Small adapter giving a uniform reset/step API and flattened observations."""

    def __init__(self, action_low: np.ndarray, action_high: np.ndarray, obs_dim: int, act_dim: int):
        self.action_low = action_low.astype(np.float64)
        self.action_high = action_high.astype(np.float64)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class DMControlAdapter(EnvAdapter):
    def __init__(self, domain: str, task: str):
        suite = try_import_dm_control()
        if suite is None:
            raise RuntimeError("dm_control is not installed. Install dm_control or use --backend gymnasium.")
        self.domain = domain
        self.task = task
        self.env = suite.load(domain_name=domain, task_name=task)
        spec = self.env.action_spec()
        action_low = np.asarray(spec.minimum, dtype=np.float64)
        action_high = np.asarray(spec.maximum, dtype=np.float64)
        ts = self.env.reset()
        obs = self._flatten_obs(ts.observation)
        super().__init__(action_low=action_low, action_high=action_high, obs_dim=len(obs), act_dim=action_low.size)
        self._last_timestep = ts

    @staticmethod
    def _flatten_obs(obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            return np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in obs.values()]).astype(np.float64)
        return np.asarray(obs, dtype=np.float64).ravel()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        # dm_control suite tasks do not expose a standard reset(seed=...) call on the env API.
        # We still accept the argument for signature compatibility.
        del seed
        self._last_timestep = self.env.reset()
        return self._flatten_obs(self._last_timestep.observation)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        clipped = np.clip(action, self.action_low, self.action_high)
        ts = self.env.step(clipped)
        self._last_timestep = ts
        obs = self._flatten_obs(ts.observation)
        reward = float(ts.reward or 0.0)
        done = bool(ts.last())
        info = {"discount": None if ts.discount is None else float(ts.discount)}
        return obs, reward, done, info

    def close(self) -> None:
        return None


class GymnasiumAdapter(EnvAdapter):
    def __init__(self, env_id: str):
        gym = try_import_gymnasium()
        if gym is None:
            raise RuntimeError("gymnasium is not installed. Install gymnasium[mujoco] or use --backend dm_control.")
        self.env_id = env_id
        self.env = gym.make(env_id)
        action_low = np.asarray(self.env.action_space.low, dtype=np.float64)
        action_high = np.asarray(self.env.action_space.high, dtype=np.float64)
        obs, _ = self.env.reset()
        obs = np.asarray(obs, dtype=np.float64).ravel()
        super().__init__(action_low=action_low, action_high=action_high, obs_dim=len(obs), act_dim=action_low.size)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        obs, _ = self.env.reset(seed=seed)
        return np.asarray(obs, dtype=np.float64).ravel()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        clipped = np.clip(action, self.action_low, self.action_high)
        obs, reward, terminated, truncated, info = self.env.step(clipped)
        done = bool(terminated or truncated)
        return np.asarray(obs, dtype=np.float64).ravel(), float(reward), done, dict(info)

    def close(self) -> None:
        self.env.close()


def make_env(backend: str, dm_domain: str, dm_task: str, gym_env: str) -> EnvAdapter:
    if backend == "dm_control":
        return DMControlAdapter(domain=dm_domain, task=dm_task)
    if backend == "gymnasium":
        return GymnasiumAdapter(env_id=gym_env)
    raise ValueError(f"Unknown backend: {backend}")


# -----------------------------------------------------------------------------
# Reservoir utilities
# -----------------------------------------------------------------------------


def deterministic_seed(obj: Any, base: int = 0) -> int:
    data = repr((obj, base)).encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


@dataclass
class ReservoirConfig:
    n_reservoir: int = 64
    spectral_radius: float = 0.95
    leak: float = 0.25
    input_scale: float = 0.25
    bias_scale: float = 0.10
    topology: str = "orthogonal"  # orthogonal | cycle | sparse
    sparsity: float = 0.08
    include_prev_action: bool = True
    include_prev_reward: bool = True


class FixedReservoir:
    def __init__(self, input_dim: int, config: ReservoirConfig, seed: int = 0):
        self.input_dim = int(input_dim)
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.W, self.U, self.b = self._make_weights()
        self.state = np.zeros(config.n_reservoir, dtype=np.float64)

    def _spectral_radius_power(self, W: np.ndarray, n_iter: int = 30) -> float:
        v = np.ones(W.shape[0], dtype=np.float64)
        v /= np.linalg.norm(v) + 1e-12
        for _ in range(n_iter):
            v = W @ v
            norm = np.linalg.norm(v)
            if norm < 1e-12:
                return 0.0
            v /= norm
        return float(np.linalg.norm(W @ v))

    def _scale_radius(self, W: np.ndarray, target: float) -> np.ndarray:
        radius = self._spectral_radius_power(W)
        if radius <= 1e-12:
            return W
        return W * (target / radius)

    def _make_weights(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        rng = self.rng
        n = cfg.n_reservoir
        if cfg.topology == "cycle":
            W = np.zeros((n, n), dtype=np.float64)
            idx = np.arange(n - 1)
            W[idx + 1, idx] = 1.0
            W[0, -1] = 1.0
            signs = rng.choice([-1.0, 1.0], size=n)
            W = W * signs[:, None]
            W = self._scale_radius(W, cfg.spectral_radius)
        elif cfg.topology == "orthogonal":
            A = rng.normal(size=(n, n))
            Q, _ = np.linalg.qr(A)
            W = Q * cfg.spectral_radius
        elif cfg.topology == "sparse":
            mask = rng.random((n, n)) < cfg.sparsity
            W = rng.normal(size=(n, n)) * mask
            W = self._scale_radius(W, cfg.spectral_radius)
        else:
            raise ValueError(f"Unknown reservoir topology: {cfg.topology}")

        U = rng.normal(scale=cfg.input_scale, size=(n, self.input_dim))
        b = rng.normal(scale=cfg.bias_scale, size=(n,))
        return W.astype(np.float64), U.astype(np.float64), b.astype(np.float64)

    @property
    def dim(self) -> int:
        return int(self.config.n_reservoir)

    def reset(self) -> None:
        self.state[...] = 0.0

    def step(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {x.size}")
        pre = self.W @ self.state + self.U @ x + self.b
        self.state = (1.0 - self.config.leak) * self.state + self.config.leak * np.tanh(pre)
        return self.state.copy()


# -----------------------------------------------------------------------------
# Policy, value fitting, and exact linear algebra
# -----------------------------------------------------------------------------


def weighted_ridge_multioutput(X: np.ndarray, Y: np.ndarray, weights: np.ndarray, reg: float) -> np.ndarray:
    """Solve argmin_B sum_i w_i ||X_i B - Y_i||^2 + reg ||B||^2.

    Returns B with shape (n_features, n_outputs).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be rank-2 arrays.")
    if X.shape[0] != Y.shape[0] or X.shape[0] != w.shape[0]:
        raise ValueError("Mismatched batch sizes in weighted ridge solve.")

    Xw = X * w[:, None]
    A = X.T @ Xw + reg * np.eye(X.shape[1], dtype=np.float64)
    B = np.linalg.solve(A, Xw.T @ Y)
    return B


class RidgeValueModel:
    def __init__(self, reg: float = 1.0):
        self.reg = float(reg)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.reg)
        self.is_fit = False

    def fit(self, Z: np.ndarray, returns: np.ndarray) -> None:
        Zs = self.scaler.fit_transform(np.asarray(Z, dtype=np.float64))
        self.model = Ridge(alpha=self.reg)
        self.model.fit(Zs, np.asarray(returns, dtype=np.float64))
        self.is_fit = True

    def predict(self, Z: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            return np.zeros(len(Z), dtype=np.float64)
        Zs = self.scaler.transform(np.asarray(Z, dtype=np.float64))
        return np.asarray(self.model.predict(Zs), dtype=np.float64)


@dataclass
class PolicyConfig:
    n_action_charts: int = 1
    action_reg: float = 1e-1
    value_reg: float = 1.0
    initial_std: float = 0.50
    min_std: float = 0.08
    max_std: float = 1.00
    blend: float = 0.35
    advantage_quantile: float = 0.60
    temperature_floor: float = 0.10
    varentropy_eta: float = 0.08
    varentropy_gamma: float = 0.50


class ActionAtlasPolicy:
    """Mixture of local linear Gaussian controllers with derivative-free updates.

    Charts correspond to local control regimes. For n_action_charts=1 the controller reduces
    to a single linear Gaussian policy on top of the reservoir state.
    """

    def __init__(self, feature_dim: int, act_dim: int, action_low: np.ndarray, action_high: np.ndarray, cfg: PolicyConfig, seed: int = 0):
        self.feature_dim = int(feature_dim)
        self.act_dim = int(act_dim)
        self.action_low = np.asarray(action_low, dtype=np.float64)
        self.action_high = np.asarray(action_high, dtype=np.float64)
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.scaler = StandardScaler().fit(np.zeros((2, feature_dim), dtype=np.float64))
        self.centroids = np.zeros((cfg.n_action_charts, feature_dim), dtype=np.float64)
        self.W = np.zeros((cfg.n_action_charts, act_dim, feature_dim + 1), dtype=np.float64)
        self.log_std = np.log(np.ones((cfg.n_action_charts, act_dim), dtype=np.float64) * cfg.initial_std)
        self.cognitive_temperature = 1.0

    def _normalize(self, Z: np.ndarray) -> np.ndarray:
        return self.scaler.transform(np.asarray(Z, dtype=np.float64))

    def fit_scaler(self, Z: np.ndarray) -> None:
        self.scaler.fit(np.asarray(Z, dtype=np.float64))

    def assign_batch(self, Z_norm: np.ndarray) -> np.ndarray:
        if self.centroids.shape[0] == 1:
            return np.zeros(len(Z_norm), dtype=int)
        d = ((Z_norm[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d, axis=1).astype(int)

    def act(self, z: np.ndarray, sample: bool = True) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        z_norm = self._normalize(np.asarray(z, dtype=np.float64).reshape(1, -1))[0]
        label = int(self.assign_batch(z_norm[None, :])[0])
        phi = np.concatenate([z_norm, np.ones(1, dtype=np.float64)], axis=0)
        mean = self.W[label] @ phi
        std = np.exp(self.log_std[label])
        if sample:
            noise = self.rng.normal(scale=std)
            action = mean + noise
        else:
            action = mean
        action = np.clip(action, self.action_low, self.action_high)
        return action.astype(np.float64), label, mean.astype(np.float64), std.astype(np.float64)

    @staticmethod
    def _varentropy(advantages: np.ndarray, temperature: float) -> float:
        temp = max(float(temperature), 1e-6)
        logits = advantages / temp
        logits = logits - np.max(logits)
        p = np.exp(np.clip(logits, -30.0, 30.0))
        p /= p.sum() + 1e-12
        surprisal = -np.log(np.maximum(p, 1e-12))
        return float(np.var(surprisal))

    def update_from_batch(self, Z: np.ndarray, actions: np.ndarray, advantages: np.ndarray, chart_hint: Optional[np.ndarray] = None) -> Dict[str, float]:
        cfg = self.cfg
        Z = np.asarray(Z, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        advantages = np.asarray(advantages, dtype=np.float64).reshape(-1)

        self.fit_scaler(Z)
        Z_norm = self._normalize(Z)

        # Update centroids using elite/positive-advantage states when multiple charts are enabled.
        if cfg.n_action_charts > 1:
            elite_mask = advantages >= np.quantile(advantages, cfg.advantage_quantile)
            Z_fit = Z_norm[elite_mask] if elite_mask.sum() >= cfg.n_action_charts else Z_norm
            if chart_hint is not None and len(np.unique(chart_hint)) >= min(cfg.n_action_charts, len(Z_fit)):
                # Warm-start centroids from prior chart usage when available.
                self.centroids = np.stack([Z_fit[np.asarray(chart_hint[elite_mask] if elite_mask.sum() >= cfg.n_action_charts else chart_hint) == k].mean(axis=0)
                                           if np.any((chart_hint[elite_mask] if elite_mask.sum() >= cfg.n_action_charts else chart_hint) == k)
                                           else Z_fit[self.rng.integers(len(Z_fit))]
                                           for k in range(cfg.n_action_charts)], axis=0)
            km = KMeans(n_clusters=cfg.n_action_charts, n_init=5, random_state=self.seed)
            km.fit(Z_fit)
            self.centroids = km.cluster_centers_.astype(np.float64)
        else:
            self.centroids[...] = 0.0

        labels = self.assign_batch(Z_norm)

        # Reward-weighted regression (RWR) / advantage-weighted behavioral cloning.
        temp = max(cfg.temperature_floor, float(np.std(advantages) + 1e-3))
        weights = np.exp(np.clip((advantages - np.max(advantages)) / temp, -20.0, 0.0))
        weights /= weights.mean() + 1e-12

        X = np.concatenate([Z_norm, np.ones((len(Z_norm), 1), dtype=np.float64)], axis=1)
        updated_charts = 0
        for k in range(cfg.n_action_charts):
            mask = labels == k
            # Ensure each chart has enough support to fit a local linear model.
            if mask.sum() < max(8, X.shape[1] // 3):
                continue
            B = weighted_ridge_multioutput(X[mask], actions[mask], weights[mask], reg=cfg.action_reg)
            new_W = B.T  # (act_dim, feat+1)
            pred = X[mask] @ B
            residual = actions[mask] - pred
            w = weights[mask]
            variance = (w[:, None] * residual**2).sum(axis=0) / (w.sum() + 1e-12)
            new_std = np.clip(np.sqrt(np.maximum(variance, 1e-4)), cfg.min_std, cfg.max_std)

            # Relative-trust-region-style damping: do not fully overwrite the controller.
            self.W[k] = (1.0 - cfg.blend) * self.W[k] + cfg.blend * new_W
            current_std = np.exp(self.log_std[k])
            mixed_std = (1.0 - cfg.blend) * current_std + cfg.blend * new_std
            self.log_std[k] = np.log(np.clip(mixed_std, cfg.min_std, cfg.max_std))
            updated_charts += 1

        # Varentropy brake: slow cooling when the policy is near a bifurcation/high-uncertainty regime.
        V_H = self._varentropy(advantages, temperature=max(self.cognitive_temperature, cfg.temperature_floor))
        self.cognitive_temperature = max(
            cfg.temperature_floor,
            self.cognitive_temperature * (1.0 - cfg.varentropy_eta / (1.0 + cfg.varentropy_gamma * V_H)),
        )

        return {
            "updated_charts": float(updated_charts),
            "varentropy": float(V_H),
            "policy_temperature": float(self.cognitive_temperature),
            "mean_std": float(np.exp(self.log_std).mean()),
        }


# -----------------------------------------------------------------------------
# Trajectory handling
# -----------------------------------------------------------------------------


@dataclass
class StepRecord:
    obs: np.ndarray
    prev_action: np.ndarray
    prev_reward: float
    action: np.ndarray
    reward: float
    done: bool
    latent: np.ndarray
    chart: int


@dataclass
class EpisodeBatch:
    steps: List[List[StepRecord]]

    @property
    def episode_returns(self) -> List[float]:
        return [float(sum(s.reward for s in ep)) for ep in self.steps]

    @property
    def n_steps(self) -> int:
        return int(sum(len(ep) for ep in self.steps))


@dataclass
class FlattenedBatch:
    Z: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    returns: np.ndarray
    charts: np.ndarray
    prev_rewards: np.ndarray
    prev_actions: np.ndarray



def compute_discounted_returns(batch: EpisodeBatch, gamma: float) -> np.ndarray:
    returns: List[float] = []
    for ep in batch.steps:
        G = 0.0
        ep_returns = [0.0] * len(ep)
        for i in reversed(range(len(ep))):
            G = ep[i].reward + gamma * G
            ep_returns[i] = float(G)
            if ep[i].done:
                G = 0.0
        returns.extend(ep_returns)
    return np.asarray(returns, dtype=np.float64)



def flatten_batch(batch: EpisodeBatch, gamma: float) -> FlattenedBatch:
    latents: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    dones: List[bool] = []
    charts: List[int] = []
    prev_rewards: List[float] = []
    prev_actions: List[np.ndarray] = []
    for ep in batch.steps:
        for step in ep:
            latents.append(step.latent)
            actions.append(step.action)
            rewards.append(step.reward)
            dones.append(step.done)
            charts.append(step.chart)
            prev_rewards.append(step.prev_reward)
            prev_actions.append(step.prev_action)
    return FlattenedBatch(
        Z=np.asarray(latents, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.float64),
        rewards=np.asarray(rewards, dtype=np.float64),
        dones=np.asarray(dones, dtype=bool),
        returns=compute_discounted_returns(batch, gamma=gamma),
        charts=np.asarray(charts, dtype=int),
        prev_rewards=np.asarray(prev_rewards, dtype=np.float64),
        prev_actions=np.asarray(prev_actions, dtype=np.float64),
    )


# -----------------------------------------------------------------------------
# Hypostructure diagnostics and response-signature-style search
# -----------------------------------------------------------------------------


@dataclass
class Diagnostics:
    K_D: float
    K_C: float
    K_LS: float
    K_TOP: float
    K_BOUND: float
    mean_return: float
    score: float


DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    "return": 1.0,
    "K_D": 0.15,
    "K_C": 0.05,
    "K_LS": 0.10,
    "K_TOP": 0.08,
    "K_BOUND": 0.03,
}


class LocalLinearWorldModel:
    """Local linear model used only for diagnostics, not gradient-based training."""

    def __init__(self, n_charts: int, reg: float = 1e-2):
        self.n_charts = int(n_charts)
        self.reg = float(reg)
        self.models: Dict[int, np.ndarray] = {}

    def fit(self, Z: np.ndarray, A: np.ndarray, labels: np.ndarray, Z_next: np.ndarray) -> None:
        self.models = {}
        X = np.concatenate([Z, A, np.ones((len(Z), 1), dtype=np.float64)], axis=1)
        for k in range(self.n_charts):
            mask = labels == k
            if mask.sum() < max(8, X.shape[1] // 3):
                continue
            B = weighted_ridge_multioutput(X[mask], Z_next[mask], np.ones(mask.sum(), dtype=np.float64), reg=self.reg)
            self.models[k] = B

    def one_step_errors(self, Z: np.ndarray, A: np.ndarray, labels: np.ndarray, Z_next: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        X = np.concatenate([Z, A, np.ones((len(Z), 1), dtype=np.float64)], axis=1)
        errs = np.zeros(len(Z), dtype=np.float64)
        radii: List[float] = []
        for k, B in self.models.items():
            mask = labels == k
            if not np.any(mask):
                continue
            pred = X[mask] @ B
            errs[mask] = ((pred - Z_next[mask]) ** 2).mean(axis=1)
            A_local = B[: Z.shape[1], :].T
            eigvals = np.linalg.eigvals(A_local)
            radii.append(float(np.max(np.abs(eigvals))))
        return errs, radii



def compute_diagnostics(batch: EpisodeBatch, flat: FlattenedBatch, n_charts: int, score_weights: Dict[str, float]) -> Diagnostics:
    episode_returns = np.asarray(batch.episode_returns, dtype=np.float64)
    mean_return = float(episode_returns.mean()) if len(episode_returns) else 0.0

    energy = np.sum(flat.Z**2, axis=1)
    if len(energy) > 1:
        # Reconstruct next-state pairs per episode.
        K_D_terms: List[float] = []
        Z_curr: List[np.ndarray] = []
        A_curr: List[np.ndarray] = []
        L_curr: List[int] = []
        Z_next: List[np.ndarray] = []
        for ep in batch.steps:
            for i in range(len(ep) - 1):
                z0 = ep[i].latent
                z1 = ep[i + 1].latent
                K_D_terms.append(max(0.0, float(np.dot(z1, z1) - np.dot(z0, z0))))
                Z_curr.append(z0)
                A_curr.append(ep[i].action)
                L_curr.append(ep[i].chart)
                Z_next.append(z1)
        K_D = float(np.mean(K_D_terms)) if K_D_terms else 0.0
        if Z_curr:
            Zc = np.asarray(Z_curr, dtype=np.float64)
            Ac = np.asarray(A_curr, dtype=np.float64)
            Lc = np.asarray(L_curr, dtype=int)
            Zn = np.asarray(Z_next, dtype=np.float64)
            world = LocalLinearWorldModel(n_charts=n_charts, reg=1e-2)
            world.fit(Zc, Ac, Lc, Zn)
            pred_errs, spectral_radii = world.one_step_errors(Zc, Ac, Lc, Zn)
            K_TOP = float(pred_errs.mean())
            # LS proxy: penalize unstable local Jacobians.
            K_LS = float(np.mean([max(0.0, r - 1.0) for r in spectral_radii])) if spectral_radii else 0.0
        else:
            K_TOP = 0.0
            K_LS = 0.0
    else:
        K_D = 0.0
        K_TOP = 0.0
        K_LS = 0.0

    # Capacity/conditioning penalty from latent covariance spectrum.
    Z_centered = flat.Z - flat.Z.mean(axis=0, keepdims=True)
    cov = (Z_centered.T @ Z_centered) / max(1, len(Z_centered) - 1)
    try:
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        condition = float(eigvals.max() / eigvals.min())
        K_C = float(max(0.0, math.log(condition) - 6.0) / 6.0)
    except np.linalg.LinAlgError:
        K_C = 1.0

    # Boundary / motor penalty: too much saturation means poor control allocation.
    sat = np.abs(flat.actions) >= (1.0 - 1e-3)
    K_BOUND = float(sat.mean())

    score = (
        score_weights["return"] * mean_return
        - score_weights["K_D"] * K_D
        - score_weights["K_C"] * K_C
        - score_weights["K_LS"] * K_LS
        - score_weights["K_TOP"] * K_TOP
        - score_weights["K_BOUND"] * K_BOUND
    )
    return Diagnostics(K_D=K_D, K_C=K_C, K_LS=K_LS, K_TOP=K_TOP, K_BOUND=K_BOUND, mean_return=mean_return, score=float(score))


@dataclass
class SearchConfig:
    enabled: bool = True
    warmup_episodes: int = 4
    warmup_horizon: int = 150
    search_every: int = 4


BLOCK_SPACE: Dict[str, List[Dict[str, Any]]] = {
    "dyn": [
        {"spectral_radius": 0.75, "leak": 0.20, "topology": "orthogonal"},
        {"spectral_radius": 0.90, "leak": 0.25, "topology": "orthogonal"},
        {"spectral_radius": 0.98, "leak": 0.35, "topology": "cycle"},
    ],
    "cap": [
        {"n_reservoir": 48, "input_scale": 0.20},
        {"n_reservoir": 64, "input_scale": 0.25},
        {"n_reservoir": 96, "input_scale": 0.35},
    ],
    "top": [
        {"n_action_charts": 1},
        {"n_action_charts": 2},
        {"n_action_charts": 4},
    ],
    "ls": [
        {"action_reg": 3e-1, "value_reg": 1.0},
        {"action_reg": 1e-1, "value_reg": 1.0},
        {"action_reg": 1e-1, "value_reg": 3.0},
    ],
}


@dataclass
class CandidateConfig:
    reservoir: ReservoirConfig
    policy: PolicyConfig



def merge_candidate(base: CandidateConfig, updates: Dict[str, Any]) -> CandidateConfig:
    r_kwargs = asdict(base.reservoir)
    p_kwargs = asdict(base.policy)
    for k, v in updates.items():
        if k in r_kwargs:
            r_kwargs[k] = v
        elif k in p_kwargs:
            p_kwargs[k] = v
        else:
            raise KeyError(f"Unknown config key in block search: {k}")
    return CandidateConfig(reservoir=ReservoirConfig(**r_kwargs), policy=PolicyConfig(**p_kwargs))


@dataclass
class BlockSearchResult:
    chosen_block: str
    chosen_config: CandidateConfig
    response_signature: Dict[str, float]
    best_scores: Dict[str, float]


class OfflineResponseSignatureSearch:
    def __init__(self, action_low: np.ndarray, action_high: np.ndarray, gamma: float, seed: int = 0):
        self.action_low = np.asarray(action_low, dtype=np.float64)
        self.action_high = np.asarray(action_high, dtype=np.float64)
        self.gamma = float(gamma)
        self.seed = int(seed)

    def _evaluate_candidate(self, episodes: List[List[StepRecord]], obs_dim: int, act_dim: int, cfg: CandidateConfig) -> float:
        input_dim = obs_dim + act_dim + 1
        reservoir = FixedReservoir(input_dim=input_dim, config=cfg.reservoir, seed=deterministic_seed(cfg, self.seed))
        batch_steps: List[List[StepRecord]] = []
        for ep in episodes:
            ep_new: List[StepRecord] = []
            reservoir.reset()
            for st in ep:
                x = np.concatenate([st.obs, st.prev_action, np.asarray([st.prev_reward], dtype=np.float64)], axis=0)
                z = reservoir.step(x)
                ep_new.append(StepRecord(
                    obs=st.obs,
                    prev_action=st.prev_action,
                    prev_reward=st.prev_reward,
                    action=st.action,
                    reward=st.reward,
                    done=st.done,
                    latent=z,
                    chart=0,
                ))
            batch_steps.append(ep_new)
        batch = EpisodeBatch(batch_steps)
        flat = flatten_batch(batch, gamma=self.gamma)
        value = RidgeValueModel(reg=cfg.policy.value_reg)
        value.fit(flat.Z, flat.returns)
        advantages = flat.returns - value.predict(flat.Z)
        policy = ActionAtlasPolicy(
            feature_dim=flat.Z.shape[1],
            act_dim=flat.actions.shape[1],
            action_low=self.action_low,
            action_high=self.action_high,
            cfg=cfg.policy,
            seed=deterministic_seed((cfg, "policy"), self.seed),
        )
        policy.update_from_batch(flat.Z, flat.actions, advantages)
        # Reassign charts after the fitted controller is available.
        Z_norm = policy._normalize(flat.Z)
        labels = policy.assign_batch(Z_norm)
        flat_search = FlattenedBatch(
            Z=flat.Z,
            actions=flat.actions,
            rewards=flat.rewards,
            dones=flat.dones,
            returns=flat.returns,
            charts=labels,
            prev_rewards=flat.prev_rewards,
            prev_actions=flat.prev_actions,
        )
        # Copy labels into episode structure for diagnostics.
        idx = 0
        diag_eps: List[List[StepRecord]] = []
        for ep in batch.steps:
            diag_ep: List[StepRecord] = []
            for st in ep:
                diag_ep.append(StepRecord(
                    obs=st.obs,
                    prev_action=st.prev_action,
                    prev_reward=st.prev_reward,
                    action=st.action,
                    reward=st.reward,
                    done=st.done,
                    latent=st.latent,
                    chart=int(labels[idx]),
                ))
                idx += 1
            diag_eps.append(diag_ep)
        diag = compute_diagnostics(EpisodeBatch(diag_eps), flat_search, n_charts=cfg.policy.n_action_charts, score_weights=DEFAULT_SCORE_WEIGHTS)
        # Convert score to risk so that smaller is better.
        return float(-diag.score)

    def run(self, current: CandidateConfig, episodes: List[List[StepRecord]], obs_dim: int, act_dim: int) -> BlockSearchResult:
        response_signature: Dict[str, float] = {}
        best_scores: Dict[str, float] = {}
        best_cfg_by_block: Dict[str, CandidateConfig] = {}
        for block, candidates in BLOCK_SPACE.items():
            best_risk: Optional[float] = None
            best_cfg: Optional[CandidateConfig] = None
            for updates in candidates:
                proposal = merge_candidate(current, updates)
                risk = self._evaluate_candidate(episodes, obs_dim=obs_dim, act_dim=act_dim, cfg=proposal)
                if best_risk is None or risk < best_risk:
                    best_risk = risk
                    best_cfg = proposal
            assert best_risk is not None and best_cfg is not None
            response_signature[block] = float(best_risk)
            best_scores[block] = float(-best_risk)
            best_cfg_by_block[block] = best_cfg
        chosen_block = min(response_signature, key=response_signature.get)
        return BlockSearchResult(
            chosen_block=chosen_block,
            chosen_config=best_cfg_by_block[chosen_block],
            response_signature=response_signature,
            best_scores=best_scores,
        )


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    backend: str = "dm_control"
    dm_domain: str = "cheetah"
    dm_task: str = "run"
    gym_env: str = "HalfCheetah-v5"
    seed: int = 0
    gamma: float = 0.99
    iterations: int = 12
    train_episodes: int = 6
    eval_episodes: int = 3
    horizon: int = 250
    search: SearchConfig = field(default_factory=SearchConfig)


@dataclass
class IterationLog:
    iteration: int
    train_mean_return: float
    eval_mean_return: float
    train_episode_returns: List[float]
    eval_episode_returns: List[float]
    diagnostics: Dict[str, float]
    policy_stats: Dict[str, float]
    response_signature: Optional[Dict[str, float]] = None
    chosen_block: Optional[str] = None
    config_snapshot: Optional[Dict[str, Any]] = None


class HypoRLTrainer:
    def __init__(self, env: EnvAdapter, cfg: TrainConfig, reservoir_cfg: ReservoirConfig, policy_cfg: PolicyConfig):
        self.env = env
        self.cfg = cfg
        self.candidate_cfg = CandidateConfig(reservoir=reservoir_cfg, policy=policy_cfg)
        self.input_dim = env.obs_dim + env.act_dim + 1
        self.reservoir = FixedReservoir(self.input_dim, reservoir_cfg, seed=cfg.seed)
        self.policy = ActionAtlasPolicy(
            feature_dim=self.reservoir.dim,
            act_dim=env.act_dim,
            action_low=env.action_low,
            action_high=env.action_high,
            cfg=policy_cfg,
            seed=cfg.seed,
        )
        self.value_model = RidgeValueModel(reg=policy_cfg.value_reg)
        self.logs: List[IterationLog] = []

    def _rebuild_models(self) -> None:
        self.reservoir = FixedReservoir(self.input_dim, self.candidate_cfg.reservoir, seed=self.cfg.seed)
        self.policy = ActionAtlasPolicy(
            feature_dim=self.reservoir.dim,
            act_dim=self.env.act_dim,
            action_low=self.env.action_low,
            action_high=self.env.action_high,
            cfg=self.candidate_cfg.policy,
            seed=self.cfg.seed,
        )
        self.value_model = RidgeValueModel(reg=self.candidate_cfg.policy.value_reg)

    def _collect(self, episodes: int, horizon: int, sample: bool, seed_offset: int = 0) -> EpisodeBatch:
        steps_all: List[List[StepRecord]] = []
        for ep_idx in range(episodes):
            obs = self.env.reset(seed=self.cfg.seed + seed_offset + ep_idx)
            self.reservoir.reset()
            prev_action = np.zeros(self.env.act_dim, dtype=np.float64)
            prev_reward = 0.0
            episode_steps: List[StepRecord] = []
            for _ in range(horizon):
                x = np.concatenate([obs, prev_action, np.asarray([prev_reward], dtype=np.float64)], axis=0)
                z = self.reservoir.step(x)
                action, chart, _, _ = self.policy.act(z, sample=sample)
                next_obs, reward, done, _info = self.env.step(action)
                episode_steps.append(
                    StepRecord(
                        obs=obs.copy(),
                        prev_action=prev_action.copy(),
                        prev_reward=float(prev_reward),
                        action=action.copy(),
                        reward=float(reward),
                        done=bool(done),
                        latent=z.copy(),
                        chart=int(chart),
                    )
                )
                obs = next_obs
                prev_action = action
                prev_reward = reward
                if done:
                    break
            steps_all.append(episode_steps)
        return EpisodeBatch(steps_all)

    def _policy_state(self) -> Dict[str, Any]:
        return {
            "centroids": self.policy.centroids.tolist(),
            "W": self.policy.W.tolist(),
            "log_std": self.policy.log_std.tolist(),
            "cognitive_temperature": float(self.policy.cognitive_temperature),
            "scaler_mean": self.policy.scaler.mean_.tolist(),
            "scaler_scale": self.policy.scaler.scale_.tolist(),
        }

    def _value_state(self) -> Dict[str, Any]:
        if not self.value_model.is_fit:
            return {}
        return {
            "coef": np.asarray(self.value_model.model.coef_, dtype=np.float64).tolist(),
            "intercept": float(np.asarray(self.value_model.model.intercept_, dtype=np.float64)),
            "scaler_mean": self.value_model.scaler.mean_.tolist(),
            "scaler_scale": self.value_model.scaler.scale_.tolist(),
        }

    def _apply_response_signature_search(self, warmup_batch: EpisodeBatch) -> BlockSearchResult:
        searcher = OfflineResponseSignatureSearch(
            action_low=self.env.action_low,
            action_high=self.env.action_high,
            gamma=self.cfg.gamma,
            seed=self.cfg.seed,
        )
        result = searcher.run(
            current=self.candidate_cfg,
            episodes=warmup_batch.steps,
            obs_dim=self.env.obs_dim,
            act_dim=self.env.act_dim,
        )
        self.candidate_cfg = result.chosen_config
        self._rebuild_models()
        return result

    def train(self) -> Dict[str, Any]:
        search_history: List[Dict[str, Any]] = []
        if self.cfg.search.enabled:
            warmup = self._collect(
                episodes=self.cfg.search.warmup_episodes,
                horizon=self.cfg.search.warmup_horizon,
                sample=True,
                seed_offset=1000,
            )
            search_result = self._apply_response_signature_search(warmup)
            search_history.append(
                {
                    "phase": "initial",
                    "chosen_block": search_result.chosen_block,
                    "response_signature": search_result.response_signature,
                    "best_scores": search_result.best_scores,
                    "candidate_config": {
                        "reservoir": asdict(self.candidate_cfg.reservoir),
                        "policy": asdict(self.candidate_cfg.policy),
                    },
                }
            )

        best_eval = float("-inf")
        best_snapshot: Optional[Dict[str, Any]] = None

        for iteration in range(1, self.cfg.iterations + 1):
            train_batch = self._collect(
                episodes=self.cfg.train_episodes,
                horizon=self.cfg.horizon,
                sample=True,
                seed_offset=10000 * iteration,
            )
            flat = flatten_batch(train_batch, gamma=self.cfg.gamma)
            self.value_model = RidgeValueModel(reg=self.candidate_cfg.policy.value_reg)
            self.value_model.fit(flat.Z, flat.returns)
            advantages = flat.returns - self.value_model.predict(flat.Z)
            policy_stats = self.policy.update_from_batch(flat.Z, flat.actions, advantages, chart_hint=flat.charts)

            # Refresh chart labels for diagnostics after the policy update.
            labels = self.policy.assign_batch(self.policy._normalize(flat.Z))
            flat_diag = FlattenedBatch(
                Z=flat.Z,
                actions=flat.actions,
                rewards=flat.rewards,
                dones=flat.dones,
                returns=flat.returns,
                charts=labels,
                prev_rewards=flat.prev_rewards,
                prev_actions=flat.prev_actions,
            )
            idx = 0
            diag_eps: List[List[StepRecord]] = []
            for ep in train_batch.steps:
                new_ep: List[StepRecord] = []
                for st in ep:
                    new_ep.append(
                        StepRecord(
                            obs=st.obs,
                            prev_action=st.prev_action,
                            prev_reward=st.prev_reward,
                            action=st.action,
                            reward=st.reward,
                            done=st.done,
                            latent=st.latent,
                            chart=int(labels[idx]),
                        )
                    )
                    idx += 1
                diag_eps.append(new_ep)
            diagnostics = compute_diagnostics(EpisodeBatch(diag_eps), flat_diag, n_charts=self.candidate_cfg.policy.n_action_charts, score_weights=DEFAULT_SCORE_WEIGHTS)

            response_signature: Optional[Dict[str, float]] = None
            chosen_block: Optional[str] = None
            if self.cfg.search.enabled and self.cfg.search.search_every > 0 and iteration % self.cfg.search.search_every == 0:
                search_result = self._apply_response_signature_search(train_batch)
                response_signature = search_result.response_signature
                chosen_block = search_result.chosen_block
                search_history.append(
                    {
                        "phase": f"iter_{iteration}",
                        "chosen_block": chosen_block,
                        "response_signature": response_signature,
                        "best_scores": search_result.best_scores,
                        "candidate_config": {
                            "reservoir": asdict(self.candidate_cfg.reservoir),
                            "policy": asdict(self.candidate_cfg.policy),
                        },
                    }
                )
                # Re-fit policy/value quickly on the same batch under the new feature/controller family.
                train_batch = self._collect(
                    episodes=max(2, self.cfg.train_episodes // 2),
                    horizon=min(self.cfg.horizon, 150),
                    sample=True,
                    seed_offset=20000 * iteration,
                )
                flat = flatten_batch(train_batch, gamma=self.cfg.gamma)
                self.value_model = RidgeValueModel(reg=self.candidate_cfg.policy.value_reg)
                self.value_model.fit(flat.Z, flat.returns)
                advantages = flat.returns - self.value_model.predict(flat.Z)
                policy_stats = self.policy.update_from_batch(flat.Z, flat.actions, advantages, chart_hint=flat.charts)

            eval_batch = self._collect(
                episodes=self.cfg.eval_episodes,
                horizon=self.cfg.horizon,
                sample=False,
                seed_offset=30000 * iteration,
            )
            log = IterationLog(
                iteration=iteration,
                train_mean_return=float(np.mean(train_batch.episode_returns)),
                eval_mean_return=float(np.mean(eval_batch.episode_returns)),
                train_episode_returns=train_batch.episode_returns,
                eval_episode_returns=eval_batch.episode_returns,
                diagnostics=asdict(diagnostics),
                policy_stats=policy_stats,
                response_signature=response_signature,
                chosen_block=chosen_block,
                config_snapshot={
                    "reservoir": asdict(self.candidate_cfg.reservoir),
                    "policy": asdict(self.candidate_cfg.policy),
                },
            )
            self.logs.append(log)
            if log.eval_mean_return > best_eval:
                best_eval = log.eval_mean_return
                best_snapshot = {
                    "iteration": iteration,
                    "candidate_config": {
                        "reservoir": asdict(self.candidate_cfg.reservoir),
                        "policy": asdict(self.candidate_cfg.policy),
                    },
                    "reservoir_state": {
                        "W": self.reservoir.W.tolist(),
                        "U": self.reservoir.U.tolist(),
                        "b": self.reservoir.b.tolist(),
                    },
                    "policy_state": self._policy_state(),
                    "value_state": self._value_state(),
                }
            print(
                f"[iter {iteration:02d}] train={log.train_mean_return:.3f} "
                f"eval={log.eval_mean_return:.3f} "
                f"std={log.policy_stats['mean_std']:.3f} "
                f"score={log.diagnostics['score']:.3f}",
                flush=True,
            )

        summary_best_eval = max((log.eval_mean_return for log in self.logs), default=float("-inf"))
        return {
            "environment": {
                "backend": self.cfg.backend,
                "dm_domain": self.cfg.dm_domain,
                "dm_task": self.cfg.dm_task,
                "gym_env": self.cfg.gym_env,
                "obs_dim": self.env.obs_dim,
                "act_dim": self.env.act_dim,
            },
            "train_config": {
                "seed": self.cfg.seed,
                "gamma": self.cfg.gamma,
                "iterations": self.cfg.iterations,
                "train_episodes": self.cfg.train_episodes,
                "eval_episodes": self.cfg.eval_episodes,
                "horizon": self.cfg.horizon,
                "search": asdict(self.cfg.search),
            },
            "final_candidate_config": {
                "reservoir": asdict(self.candidate_cfg.reservoir),
                "policy": asdict(self.candidate_cfg.policy),
            },
            "search_history": search_history,
            "iterations": [asdict(log) for log in self.logs],
            "best_snapshot": best_snapshot,
            "final_policy_state": self._policy_state(),
            "final_value_state": self._value_state(),
            "final_reservoir_state": {
                "W": self.reservoir.W.tolist(),
                "U": self.reservoir.U.tolist(),
                "b": self.reservoir.b.tolist(),
            },
            "summary": {
                "best_eval_mean_return": float(summary_best_eval),
                "final_eval_mean_return": float(self.logs[-1].eval_mean_return if self.logs else float("nan")),
                "initial_eval_mean_return": float(self.logs[0].eval_mean_return if self.logs else float("nan")),
                "n_logs": len(self.logs),
            },
        }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient-free hypostructure RL on dm_control cheetah/run or Gymnasium HalfCheetah")
    parser.add_argument("--backend", type=str, default="dm_control", choices=["dm_control", "gymnasium"])
    parser.add_argument("--dm-domain", type=str, default="cheetah")
    parser.add_argument("--dm-task", type=str, default="run")
    parser.add_argument("--gym-env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--train-episodes", type=int, default=6)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--disable-search", action="store_true")
    parser.add_argument("--search-every", type=int, default=4)
    parser.add_argument("--warmup-episodes", type=int, default=4)
    parser.add_argument("--warmup-horizon", type=int, default=150)
    parser.add_argument("--n-reservoir", type=int, default=64)
    parser.add_argument("--spectral-radius", type=float, default=0.95)
    parser.add_argument("--leak", type=float, default=0.25)
    parser.add_argument("--input-scale", type=float, default=0.25)
    parser.add_argument("--topology", type=str, default="orthogonal", choices=["orthogonal", "cycle", "sparse"])
    parser.add_argument("--n-action-charts", type=int, default=1)
    parser.add_argument("--action-reg", type=float, default=1e-1)
    parser.add_argument("--value-reg", type=float, default=1.0)
    parser.add_argument("--initial-std", type=float, default=0.50)
    parser.add_argument("--output-json", type=str, default="hypo_dmcontrol_halfcheetah_results.json")
    parser.add_argument("--output-model-npz", type=str, default=None, help="Optional path to save best/final learned parameters as .npz")
    return parser.parse_args(argv)



def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    train_cfg = TrainConfig(
        backend=args.backend,
        dm_domain=args.dm_domain,
        dm_task=args.dm_task,
        gym_env=args.gym_env,
        seed=args.seed,
        gamma=args.gamma,
        iterations=args.iterations,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        horizon=args.horizon,
        search=SearchConfig(
            enabled=not args.disable_search,
            warmup_episodes=args.warmup_episodes,
            warmup_horizon=args.warmup_horizon,
            search_every=args.search_every,
        ),
    )
    reservoir_cfg = ReservoirConfig(
        n_reservoir=args.n_reservoir,
        spectral_radius=args.spectral_radius,
        leak=args.leak,
        input_scale=args.input_scale,
        topology=args.topology,
    )
    policy_cfg = PolicyConfig(
        n_action_charts=args.n_action_charts,
        action_reg=args.action_reg,
        value_reg=args.value_reg,
        initial_std=args.initial_std,
    )

    env = make_env(args.backend, args.dm_domain, args.dm_task, args.gym_env)
    trainer = HypoRLTrainer(env=env, cfg=train_cfg, reservoir_cfg=reservoir_cfg, policy_cfg=policy_cfg)
    t0 = time.time()
    try:
        results = trainer.train()
    finally:
        env.close()
    results["wall_clock_seconds"] = float(time.time() - t0)

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(results, indent=2))
    if args.output_model_npz:
        best = results.get("best_snapshot") or {}
        np.savez_compressed(
            args.output_model_npz,
            final_reservoir_W=np.asarray(results["final_reservoir_state"]["W"], dtype=np.float64),
            final_reservoir_U=np.asarray(results["final_reservoir_state"]["U"], dtype=np.float64),
            final_reservoir_b=np.asarray(results["final_reservoir_state"]["b"], dtype=np.float64),
            final_policy_W=np.asarray(results["final_policy_state"]["W"], dtype=np.float64),
            final_policy_log_std=np.asarray(results["final_policy_state"]["log_std"], dtype=np.float64),
            final_policy_centroids=np.asarray(results["final_policy_state"]["centroids"], dtype=np.float64),
            final_policy_scaler_mean=np.asarray(results["final_policy_state"]["scaler_mean"], dtype=np.float64),
            final_policy_scaler_scale=np.asarray(results["final_policy_state"]["scaler_scale"], dtype=np.float64),
            best_iteration=np.asarray(best.get("iteration", -1), dtype=np.int64),
            best_policy_W=np.asarray(best.get("policy_state", {}).get("W", []), dtype=np.float64),
            best_policy_log_std=np.asarray(best.get("policy_state", {}).get("log_std", []), dtype=np.float64),
            best_policy_centroids=np.asarray(best.get("policy_state", {}).get("centroids", []), dtype=np.float64),
            best_policy_scaler_mean=np.asarray(best.get("policy_state", {}).get("scaler_mean", []), dtype=np.float64),
            best_policy_scaler_scale=np.asarray(best.get("policy_state", {}).get("scaler_scale", []), dtype=np.float64),
        )
    print(f"\nSaved results to {out_path}")
    print(json.dumps(results["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
