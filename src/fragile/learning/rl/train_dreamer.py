"""Geometric Dreamer: Phase 4 model-based RL on the Poincare ball.

Trains a critic-induced control field with a geometric world model and a
deterministic action boundary map on dm_control environments.

The encoder uses the same Phase 1 reconstruction/atlas objective as
``train_joint.py`` plus the shared-codebook Markov closure/Zeno probe.
All RL heads consume detached latents; they never update the topoencoder.

Usage:
    uv run python -m fragile.learning.rl.train_dreamer
    uv run python -m fragile.learning.rl.train_dreamer --domain walker --task walk
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
import os
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fragile.learning.core.layers import FactorizedJumpOperator
from fragile.learning.core.layers.atlas import TopoEncoderPrimitives, _project_to_ball
from fragile.learning.vla.covariant_world_model import GeometricWorldModel
from fragile.learning.vla.config import VLAConfig
from fragile.learning.vla.losses import (
    DynamicsTransitionModel,
    compute_dynamics_chart_loss,
    compute_dynamics_geodesic_loss,
    compute_dynamics_markov_loss,
    compute_energy_conservation_loss,
    compute_hodge_consistency_loss,
    compute_momentum_regularization,
    compute_phase1_loss,
    compute_screened_poisson_loss,
)
from fragile.learning.vla.optim import build_encoder_param_groups
from fragile.learning.vla.shared_dyn.encoder import SharedDynTopoEncoder
from fragile.learning.vla.phase1_control import (
    init_phase1_adaptive_state,
    phase1_effective_weight_scales,
    update_phase1_adaptive_state,
)
from fragile.learning.vla.train_joint import (
    _compute_encoder_losses,
    _get_hard_routing_tau,
    _phase1_grad_breakdown,
    _use_hard_routing,
)

from fragile.fractalai.robots.death_conditions import walker_ground_death
from fragile.fractalai.robots.dm_control_env import VectorizedDMControlEnv
from fragile.fractalai.robots.robotic_gas import RoboticFractalGas

from .boundary import (
    GeometricActionBoundaryDecoder,
    GeometricActionEncoder,
    critic_control_field,
    critic_value,
    lower_control,
    raise_control,
)
from .config import DreamerConfig
from .replay_buffer import SequenceReplayBuffer
from .reward_head import RewardHead

try:
    from fragile.learning.mlflow_logging import (
        end_mlflow_run,
        log_mlflow_metrics,
    )
except ImportError:
    log_mlflow_metrics = None  # type: ignore[assignment]
    end_mlflow_run = None  # type: ignore[assignment]

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Environment helpers (dm_control)
# ---------------------------------------------------------------------------


def _make_env(domain: str, task: str):
    """Create a dm_control environment."""
    from dm_control import suite

    return suite.load(domain_name=domain, task_name=task)


def _flatten_obs(time_step) -> np.ndarray:
    """Flatten dm_control observation OrderedDict to a single vector."""
    parts = []
    for v in time_step.observation.values():
        v = np.asarray(v, dtype=np.float32).flatten()
        parts.append(v)
    return np.concatenate(parts)


def _rollout_routing_tau(hard_routing: bool, hard_routing_tau: float) -> float:
    """Use deterministic routing for deployed rollouts and evaluation."""
    if hard_routing:
        return -1.0
    return hard_routing_tau


def _clone_prefix_inplace(
    history: np.ndarray,
    companions: np.ndarray,
    will_clone: np.ndarray,
    end_col: int,
) -> None:
    """Clone the recorded prefix so rows follow walker ancestry after cloning."""
    if end_col <= 0 or not will_clone.any():
        return
    history[will_clone, :end_col] = history[companions[will_clone], :end_col]


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[torch.nn.Parameter]:
    """Return unique optimizer parameters in stable order."""
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            if id(param) in seen:
                continue
            seen.add(id(param))
            params.append(param)
    return params


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the first param-group learning rate for logging."""
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


def _shared_value_field(world_model: nn.Module, critic: nn.Module) -> bool:
    """Return whether the critic is the world-model value branch."""
    wm = _unwrap_compiled_module(world_model)
    critic_mod = _unwrap_compiled_module(critic)
    return critic_mod is getattr(wm, "potential_net", None)


def _parameter_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    """Compute an unclipped gradient norm for diagnostics."""
    grads = [p.grad.detach().norm() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack(grads), p=2))


def _make_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Cosine-anneal from the optimizer base LR to ``min_lr`` across the run."""
    t_max = max(int(total_epochs) - 1, 1)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=min_lr,
    )


def _maybe_compile_module(module: nn.Module, enabled: bool, mode: str) -> nn.Module:
    """Optionally wrap a module with ``torch.compile`` without changing defaults."""
    if not enabled or not hasattr(torch, "compile"):
        return module
    return torch.compile(module, mode=mode)


def _unwrap_compiled_module(module: nn.Module) -> nn.Module:
    """Return the original module behind ``torch.compile`` wrappers."""
    return getattr(module, "_orig_mod", module)


@dataclass
class ObservationNormalizer:
    """Fixed per-dimension affine observation normalization."""

    mean: torch.Tensor
    std: torch.Tensor
    min_std: float = 1e-3

    @classmethod
    def from_episodes(
        cls,
        episodes: list[dict[str, np.ndarray]],
        device: torch.device,
        *,
        min_std: float = 1e-3,
    ) -> ObservationNormalizer:
        if not episodes:
            msg = "Need at least one episode to estimate observation normalization stats."
            raise ValueError(msg)
        obs = np.concatenate([episode["obs"] for episode in episodes], axis=0).astype(np.float32)
        mean = torch.from_numpy(obs.mean(axis=0)).to(device=device)
        std = torch.from_numpy(obs.std(axis=0)).to(device=device).clamp(min=min_std)
        return cls(mean=mean, std=std, min_std=min_std)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor | float],
        device: torch.device,
    ) -> ObservationNormalizer:
        min_std = float(state_dict.get("min_std", 1e-3))
        mean = torch.as_tensor(state_dict["mean"], device=device, dtype=torch.float32)
        std = torch.as_tensor(state_dict["std"], device=device, dtype=torch.float32).clamp(
            min=min_std,
        )
        return cls(mean=mean, std=std, min_std=min_std)

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "min_std": float(self.min_std),
        }

    def normalize_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=obs.device, dtype=obs.dtype)
        std = self.std.to(device=obs.device, dtype=obs.dtype)
        return (obs - mean) / std

    def denormalize_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=obs.device, dtype=obs.dtype)
        std = self.std.to(device=obs.device, dtype=obs.dtype)
        return obs * std + mean

    def normalize_numpy(self, obs: np.ndarray) -> np.ndarray:
        mean = self.mean.detach().cpu().numpy()
        std = self.std.detach().cpu().numpy()
        return ((obs - mean) / std).astype(np.float32, copy=False)


@contextmanager
def _freeze_modules(*modules: nn.Module):
    """Temporarily freeze module parameters while preserving input gradients."""
    requires_grad_states: list[list[bool]] = []
    for module in modules:
        states = [param.requires_grad for param in module.parameters()]
        requires_grad_states.append(states)
        for param in module.parameters():
            param.requires_grad_(False)
    try:
        yield
    finally:
        for module, states in zip(modules, requires_grad_states, strict=False):
            for param, requires_grad in zip(module.parameters(), states, strict=False):
                param.requires_grad_(requires_grad)


def _policy_action(
    critic: nn.Module,
    action_decoder: GeometricActionBoundaryDecoder,
    z: torch.Tensor,
    rw: torch.Tensor,
    *,
    use_motor_texture: bool,
) -> dict[str, torch.Tensor]:
    """Decode the critic-induced control field into a boundary action."""
    control_metric = getattr(action_decoder, "metric", None)
    with _freeze_modules(critic):
        control_cov, control_tan, _ = critic_control_field(
            critic,
            z,
            rw,
            metric=control_metric,
            create_graph=False,
        )
    control_cov = control_cov.detach()
    control_tan = control_tan.detach()
    boundary = action_decoder(
        z.detach(),
        control_tan,
        rw.detach(),
    )
    if use_motor_texture:
        boundary_exec = action_decoder.sample_execution_action(
            z.detach(),
            control_tan,
            rw.detach(),
            macro_probs=boundary["macro_probs"],
            motor_nuisance=boundary["motor_nuisance"],
            motor_compliance=boundary["motor_compliance"],
        )
        action = boundary_exec["action"]
        action_mean = boundary_exec["action_mean"]
        log_std = boundary_exec["log_std"]
    else:
        action_mean = action_decoder.decode(
            z.detach(),
            control_tan,
            rw.detach(),
            macro_probs=boundary["macro_probs"],
            motor_nuisance=boundary["motor_nuisance"],
            motor_compliance=boundary["motor_compliance"],
        )
        log_std = boundary["log_std"]
        action = action_mean
    return {
        "action": action.detach(),
        "action_mean": action_mean.detach(),
        "control_tan": control_tan,
        "control_cov": control_cov,
        "log_std": log_std.detach(),
        "motor_macro_probs": boundary["macro_probs"].detach(),
        "motor_macro_idx": boundary["macro_idx"].detach(),
        "motor_nuisance": boundary["motor_nuisance"].detach(),
        "motor_compliance": boundary["motor_compliance"].detach(),
    }


def _collect_episode(
    env,
    critic: nn.Module | None,
    action_decoder: GeometricActionBoundaryDecoder | None,
    encoder: nn.Module,
    device: torch.device,
    control_dim: int,
    num_action_macros: int,
    obs_normalizer: ObservationNormalizer | None = None,
    action_repeat: int = 1,
    max_steps: int = 1000,
    hard_routing: bool = True,
    hard_routing_tau: float = 1.0,
    use_motor_texture: bool = True,
) -> dict[str, np.ndarray]:
    """Collect a single episode.  Uses random actions if the policy is absent."""
    obs_list, act_list, rew_list, done_list = [], [], [], []
    action_mean_list, control_tan_list, control_cov_list, control_valid_list = [], [], [], []
    motor_macro_probs_list, motor_nuisance_list, motor_compliance_list = [], [], []
    time_step = env.reset()
    action_spec = env.action_spec()
    step = 0
    routing_tau = _rollout_routing_tau(hard_routing, hard_routing_tau)

    while not time_step.last() and step < max_steps:
        obs = _flatten_obs(time_step)
        obs_list.append(obs)

        if critic is None or action_decoder is None:
            action = np.random.uniform(
                action_spec.minimum, action_spec.maximum,
                size=action_spec.shape,
            ).astype(np.float32)
            action_mean = action.copy()
            control_tan = np.zeros(control_dim, dtype=np.float32)
            control_cov = np.zeros(control_dim, dtype=np.float32)
            control_valid = 0.0
            motor_macro_probs = np.zeros(num_action_macros, dtype=np.float32)
            motor_nuisance = np.zeros(control_dim, dtype=np.float32)
            motor_compliance = np.zeros((action.shape[0], action.shape[0]), dtype=np.float32)
        else:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            if obs_normalizer is not None:
                obs_t = obs_normalizer.normalize_tensor(obs_t)
            with torch.no_grad():
                enc_out = encoder.encoder(
                    obs_t,
                    hard_routing=hard_routing,
                    hard_routing_tau=routing_tau,
                )
                rw = enc_out[4]   # [1, K] router weights
                z = enc_out[5]    # [1, D] z_geo
            action_out = _policy_action(
                critic,
                action_decoder,
                z,
                rw,
                use_motor_texture=use_motor_texture,
            )
            action = action_out["action"].squeeze(0).cpu().numpy()
            action_mean = action_out["action_mean"].squeeze(0).cpu().numpy()
            control_tan = action_out["control_tan"].squeeze(0).cpu().numpy()
            control_cov = action_out["control_cov"].squeeze(0).cpu().numpy()
            control_valid = 1.0
            motor_macro_probs = action_out["motor_macro_probs"].squeeze(0).cpu().numpy()
            motor_nuisance = action_out["motor_nuisance"].squeeze(0).cpu().numpy()
            motor_compliance = action_out["motor_compliance"].squeeze(0).cpu().numpy()
            action = np.clip(action, action_spec.minimum, action_spec.maximum)
            action_mean = np.clip(action_mean, action_spec.minimum, action_spec.maximum)

        total_reward = 0.0
        for _ in range(action_repeat):
            time_step = env.step(action)
            total_reward += time_step.reward or 0.0
            if time_step.last():
                break

        act_list.append(action)
        action_mean_list.append(action_mean)
        control_tan_list.append(control_tan)
        control_cov_list.append(control_cov)
        control_valid_list.append(np.float32(control_valid))
        motor_macro_probs_list.append(motor_macro_probs)
        motor_nuisance_list.append(motor_nuisance)
        motor_compliance_list.append(motor_compliance)
        rew_list.append(np.float32(total_reward))
        done_list.append(np.float32(time_step.last()))
        step += 1

    # Append final observation
    obs_list.append(_flatten_obs(time_step))

    return {
        "obs": np.stack(obs_list),
        "actions": np.stack(act_list + [np.zeros_like(act_list[0])]),
        "action_means": np.stack(action_mean_list + [np.zeros_like(action_mean_list[0])]),
        "controls": np.stack(control_cov_list + [np.zeros_like(control_cov_list[0])]),
        "controls_tan": np.stack(control_tan_list + [np.zeros_like(control_tan_list[0])]),
        "controls_cov": np.stack(control_cov_list + [np.zeros_like(control_cov_list[0])]),
        "control_valid": np.array(control_valid_list + [0.0], dtype=np.float32),
        "motor_macro_probs": np.stack(
            motor_macro_probs_list + [np.zeros_like(motor_macro_probs_list[0])],
        ),
        "motor_nuisance": np.stack(
            motor_nuisance_list + [np.zeros_like(motor_nuisance_list[0])],
        ),
        "motor_compliance": np.stack(
            motor_compliance_list + [np.zeros_like(motor_compliance_list[0])],
        ),
        "rewards": np.array(rew_list + [0.0], dtype=np.float32),
        "dones": np.array(done_list + [1.0], dtype=np.float32),
    }


def _eval_policy(
    env,
    critic: nn.Module,
    action_decoder: GeometricActionBoundaryDecoder,
    encoder: nn.Module,
    device: torch.device,
    action_repeat: int,
    num_episodes: int,
    max_steps: int,
    obs_normalizer: ObservationNormalizer | None = None,
    hard_routing: bool = True,
    hard_routing_tau: float = 1.0,
) -> dict[str, float]:
    """Run evaluation episodes and return summary stats."""
    rewards, lengths = [], []
    action_spec = env.action_spec()
    routing_tau = _rollout_routing_tau(hard_routing, hard_routing_tau)
    for _ in range(num_episodes):
        time_step = env.reset()
        ep_reward, ep_len = 0.0, 0
        while not time_step.last() and ep_len < max_steps:
            obs = _flatten_obs(time_step)
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                if obs_normalizer is not None:
                    obs_t = obs_normalizer.normalize_tensor(obs_t)
                enc_out = encoder.encoder(
                    obs_t,
                    hard_routing=hard_routing,
                    hard_routing_tau=routing_tau,
                )
                rw, z = enc_out[4], enc_out[5]
            action_out = _policy_action(
                critic,
                action_decoder,
                z,
                rw,
                use_motor_texture=False,
            )
            action = action_out["action"].squeeze(0).cpu().numpy()
            action = np.clip(action, action_spec.minimum, action_spec.maximum)
            for _ in range(action_repeat):
                time_step = env.step(action)
                ep_reward += time_step.reward or 0.0
                if time_step.last():
                    break
            ep_len += 1
        rewards.append(ep_reward)
        lengths.append(ep_len)
    return {
        "eval/reward_mean": float(np.mean(rewards)),
        "eval/reward_std": float(np.std(rewards)),
        "eval/length_mean": float(np.mean(lengths)),
    }


# ---------------------------------------------------------------------------
# Gas-based data collection
# ---------------------------------------------------------------------------


def _collect_gas_episodes(
    critic: nn.Module | None,
    action_decoder: GeometricActionBoundaryDecoder | None,
    model: nn.Module,
    device: torch.device,
    config: DreamerConfig,
    obs_normalizer: ObservationNormalizer | None = None,
) -> tuple[list[dict[str, np.ndarray]], dict[str, float]]:
    """Collect diverse transitions using RoboticFractalGas.

    Returns (episodes, metrics) where each episode dict has keys
    obs, actions, rewards, dones — the same format as _collect_episode.
    """
    env = VectorizedDMControlEnv(
        f"{config.domain}-{config.task}",
        n_workers=config.gas_n_env_workers,
        include_rgb=False,
    )
    death_cond = walker_ground_death if config.gas_use_death_condition else None
    gas = RoboticFractalGas(
        env=env,
        N=config.gas_walkers,
        reward_coef=config.gas_reward_coef,
        dist_coef=config.gas_dist_coef,
        n_elite=config.gas_n_elite,
        dt_range=(config.action_repeat, config.action_repeat),
        device=config.device,
        death_condition=death_cond,
    )

    state = gas.reset()
    N = config.gas_walkers
    steps = config.gas_steps
    obs_dim = state.observations.shape[1]
    action_dim = env.action_space.shape[0]

    # Pre-allocate arrays
    all_obs = np.zeros((N, steps + 1, obs_dim), dtype=np.float32)
    all_actions = np.zeros((N, steps + 1, action_dim), dtype=np.float32)
    all_action_means = np.zeros((N, steps + 1, action_dim), dtype=np.float32)
    all_controls_tan = np.zeros((N, steps + 1, config.latent_dim), dtype=np.float32)
    all_controls_cov = np.zeros((N, steps + 1, config.latent_dim), dtype=np.float32)
    all_control_valid = np.zeros((N, steps + 1), dtype=np.float32)
    num_action_macros = config.num_action_macros
    all_motor_macro_probs = np.zeros((N, steps + 1, num_action_macros), dtype=np.float32)
    all_motor_nuisance = np.zeros((N, steps + 1, config.latent_dim), dtype=np.float32)
    all_motor_compliance = np.zeros((N, steps + 1, action_dim, action_dim), dtype=np.float32)
    all_rewards = np.zeros((N, steps + 1), dtype=np.float32)
    all_dones = np.zeros((N, steps + 1), dtype=np.float32)

    all_obs[:, 0] = state.observations.cpu().numpy()
    all_dones[:, 0] = state.dones.cpu().float().numpy()
    routing_tau = _rollout_routing_tau(config.hard_routing, config.hard_routing_tau)

    for t in range(steps):
        # Compute actions from policy (or None for random)
        actions_np = None
        if critic is not None and action_decoder is not None:
            obs_t = state.observations.to(device)
            if obs_normalizer is not None:
                obs_t = obs_normalizer.normalize_tensor(obs_t)
            chunk_size = 1024
            action_chunks = []
            action_mean_chunks = []
            control_tan_chunks = []
            control_cov_chunks = []
            macro_chunks = []
            nuisance_chunks = []
            compliance_chunks = []
            for i in range(0, N, chunk_size):
                obs_chunk = obs_t[i : i + chunk_size]
                with torch.no_grad():
                    enc_out = model.encoder(
                        obs_chunk,
                        hard_routing=config.hard_routing,
                        hard_routing_tau=routing_tau,
                    )
                    rw = enc_out[4]
                    z = enc_out[5]
                action_out = _policy_action(
                    critic,
                    action_decoder,
                    z,
                    rw,
                    use_motor_texture=config.use_motor_texture,
                )
                action_chunks.append(action_out["action"].cpu().numpy())
                action_mean_chunks.append(action_out["action_mean"].cpu().numpy())
                control_tan_chunks.append(action_out["control_tan"].cpu().numpy())
                control_cov_chunks.append(action_out["control_cov"].cpu().numpy())
                macro_chunks.append(action_out["motor_macro_probs"].cpu().numpy())
                nuisance_chunks.append(action_out["motor_nuisance"].cpu().numpy())
                compliance_chunks.append(action_out["motor_compliance"].cpu().numpy())
            actions_np = np.concatenate(action_chunks, axis=0)
            all_action_means[:, t] = np.concatenate(action_mean_chunks, axis=0)
            all_controls_tan[:, t] = np.concatenate(control_tan_chunks, axis=0)
            all_controls_cov[:, t] = np.concatenate(control_cov_chunks, axis=0)
            all_motor_macro_probs[:, t] = np.concatenate(macro_chunks, axis=0)
            all_motor_nuisance[:, t] = np.concatenate(nuisance_chunks, axis=0)
            all_motor_compliance[:, t] = np.concatenate(compliance_chunks, axis=0)
            all_control_valid[:, t] = 1.0
            actions_np = np.clip(
                actions_np,
                env.action_space.minimum,
                env.action_space.maximum,
            ).astype(np.float64)

        state, step_info = gas.step(state, actions=actions_np)

        companions = step_info["clone_companions"].cpu().numpy()
        will_clone = step_info["will_clone"].cpu().numpy().astype(bool, copy=False)
        _clone_prefix_inplace(all_obs, companions, will_clone, t + 1)
        _clone_prefix_inplace(all_actions, companions, will_clone, t)
        _clone_prefix_inplace(all_action_means, companions, will_clone, t)
        _clone_prefix_inplace(all_controls_tan, companions, will_clone, t)
        _clone_prefix_inplace(all_controls_cov, companions, will_clone, t)
        _clone_prefix_inplace(all_control_valid, companions, will_clone, t)
        _clone_prefix_inplace(all_motor_macro_probs, companions, will_clone, t)
        _clone_prefix_inplace(all_motor_nuisance, companions, will_clone, t)
        _clone_prefix_inplace(all_motor_compliance, companions, will_clone, t)
        _clone_prefix_inplace(all_rewards, companions, will_clone, t)
        _clone_prefix_inplace(all_dones, companions, will_clone, t + 1)

        all_obs[:, t + 1] = state.observations.cpu().numpy()
        all_actions[:, t] = gas.kinetic_op.last_actions
        all_rewards[:, t] = state.step_rewards.cpu().numpy()
        all_dones[:, t] = state.dones.cpu().float().numpy()

    # Mark final column as done
    all_dones[:, -1] = 1.0

    # Convert each walker trajectory to an episode dict
    episodes: list[dict[str, np.ndarray]] = []
    for i in range(N):
        episodes.append({
            "obs": all_obs[i],           # [steps+1, obs_dim]
            "actions": all_actions[i],   # [steps+1, action_dim]
            "action_means": all_action_means[i],
            "controls": all_controls_cov[i],
            "controls_tan": all_controls_tan[i],
            "controls_cov": all_controls_cov[i],
            "control_valid": all_control_valid[i],
            "motor_macro_probs": all_motor_macro_probs[i],
            "motor_nuisance": all_motor_nuisance[i],
            "motor_compliance": all_motor_compliance[i],
            "rewards": all_rewards[i],   # [steps+1]
            "dones": all_dones[i],       # [steps+1]
        })

    gas_metrics = {
        "gas/max_reward": float(state.rewards.max().item()),
        "gas/mean_reward": float(state.rewards.mean().item()),
        "gas/total_clones": float(gas.total_clones),
        "gas/alive_frac": float((~state.dones).float().mean().item()),
        "gas/n_episodes": N,
        "gas/transitions": N * steps,
    }

    return episodes, gas_metrics


# ---------------------------------------------------------------------------
# Imagination
# ---------------------------------------------------------------------------


def _imagine(
    world_model: GeometricWorldModel,
    reward_head: RewardHead,
    critic: nn.Module,
    action_decoder: GeometricActionBoundaryDecoder,
    z_0: torch.Tensor,
    rw_0: torch.Tensor,
    horizon: int,
) -> dict[str, torch.Tensor]:
    """Roll out the critic-induced control field in latent space.

    Returns:
        z_states: [B, H, D] policy states before each action
        rw_states: [B, H, K] router weights before each action
        z_traj: [B, H, D]
        rw_traj: [B, H, K]
        controls_tan: [B, H, D]
        controls_cov: [B, H, D]
        motor_macro_probs: [B, H, M]
        motor_nuisance: [B, H, D]
        motor_compliance: [B, H, A, A]
        actions: [B, H, A]
        action_log_std: [B, H, A]
        rewards: [B, H]
        reward_conservative: [B, H]
        reward_nonconservative: [B, H]
        reward_curl_norm: [B, H]
        phi_eff: [B, H, 1]
    """
    z, rw = z_0, rw_0
    p = world_model.momentum_init(z_0)  # [B, D]

    z_state_list, rw_state_list = [], []
    z_list, rw_list, control_tan_list, control_cov_list, action_list = [], [], [], [], []
    macro_list, nuisance_list, compliance_list = [], [], []
    r_list, r_cons_list, r_noncons_list, r_curl_list = [], [], [], []
    texture_list, phi_list = [], []

    for _t in range(horizon):
        action_out = _policy_action(
            critic,
            action_decoder,
            z,
            rw,
            use_motor_texture=False,
        )
        z_state_list.append(z.detach())
        rw_state_list.append(rw.detach())
        control_tan_list.append(action_out["control_tan"])
        control_cov_list.append(action_out["control_cov"])
        macro_list.append(action_out["motor_macro_probs"])
        nuisance_list.append(action_out["motor_nuisance"])
        compliance_list.append(action_out["motor_compliance"])
        action_list.append(action_out["action_mean"])

        with torch.no_grad():
            reward_info = reward_head.decompose(
                z,
                action_out["action_mean"],
                rw,
                control=action_out["control_tan"],
            )
            r_hat = reward_info["reward_total"]  # [B, 1]
            step_out = world_model._rollout_transition(
                z,
                p,
                action_out["control_cov"],
                rw,
                track_energy=False,
            )
            z = step_out["z"]
            p = step_out["p"]
            rw = step_out["rw"]
            phi_eff = step_out["phi_eff"]

        z_list.append(z.detach())
        rw_list.append(rw.detach())
        r_list.append(r_hat.squeeze(-1))
        r_cons_list.append(reward_info["reward_conservative"].squeeze(-1))
        r_noncons_list.append(reward_info["reward_nonconservative"].squeeze(-1))
        r_curl_list.append(torch.linalg.norm(reward_info["reward_curl"], dim=(-2, -1)))
        texture_list.append(action_out["log_std"])
        phi_list.append(phi_eff)

    return {
        "z_states": torch.stack(z_state_list, dim=1),  # [B, H, D]
        "rw_states": torch.stack(rw_state_list, dim=1),  # [B, H, K]
        "z_traj": torch.stack(z_list, dim=1),   # [B, H, D]
        "rw_traj": torch.stack(rw_list, dim=1),   # [B, H, K]
        "controls": torch.stack(control_tan_list, dim=1),  # [B, H, D]
        "controls_tan": torch.stack(control_tan_list, dim=1),  # [B, H, D]
        "controls_cov": torch.stack(control_cov_list, dim=1),  # [B, H, D]
        "motor_macro_probs": torch.stack(macro_list, dim=1),  # [B, H, M]
        "motor_nuisance": torch.stack(nuisance_list, dim=1),  # [B, H, D]
        "motor_compliance": torch.stack(compliance_list, dim=1),  # [B, H, A, A]
        "actions": torch.stack(action_list, dim=1),  # [B, H, A]
        "rewards": torch.stack(r_list, dim=1),   # [B, H]
        "reward_conservative": torch.stack(r_cons_list, dim=1),  # [B, H]
        "reward_nonconservative": torch.stack(r_noncons_list, dim=1),  # [B, H]
        "reward_curl_norm": torch.stack(r_curl_list, dim=1),  # [B, H]
        "action_log_std": torch.stack(texture_list, dim=1),  # [B, H, A]
        "phi_eff": torch.stack(phi_list, dim=1),    # [B, H, 1]
    }


# ---------------------------------------------------------------------------
# VLAConfig bridge — create Phase 1 config from DreamerConfig
# ---------------------------------------------------------------------------


def _phase1_config(
    config: DreamerConfig,
    phase1_state=None,
) -> VLAConfig:
    """Build a VLAConfig with Phase 1 weights matching DreamerConfig."""
    scales = phase1_effective_weight_scales(config, phase1_state)
    return VLAConfig(
        input_dim=config.obs_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_charts=config.num_charts,
        codes_per_chart=config.codes_per_chart,
        w_feature_recon=config.w_feature_recon,
        w_vq=config.w_vq,
        w_entropy=config.w_entropy * scales["entropy_scale"],
        w_diversity=config.w_diversity * scales["chart_usage_scale"],
        chart_usage_entropy_low=config.chart_usage_h_low,
        chart_usage_entropy_high=config.chart_usage_h_high,
        w_chart_ot=config.w_chart_ot * scales["chart_ot_scale"],
        chart_ot_epsilon=config.chart_ot_epsilon,
        chart_ot_iters=config.chart_ot_iters,
        w_uniformity=config.w_uniformity,
        w_radial_calibration=config.w_radial_calibration,
        w_confidence_calibration=config.w_confidence_calibration,
        w_hard_routing_nll=config.w_hard_routing_nll,
        w_router_margin=config.w_router_margin,
        router_margin_target=config.router_margin_target,
        radial_quality_alpha=config.radial_quality_alpha,
        radial_vq_alpha=config.radial_vq_alpha,
        radial_quality_rank_mix=config.radial_quality_rank_mix,
        radial_recon_quality_weight=config.radial_recon_quality_weight,
        radial_quality_mix=config.radial_quality_mix,
        radial_quality_base_weight=config.radial_quality_base_weight,
        radial_calibration_rho_max=config.radial_calibration_rho_max,
        radial_calibration_band_width=config.radial_calibration_band_width,
        w_v_tangent_barrier=config.w_v_tangent_barrier,
        w_codebook_spread=config.w_codebook_spread,
        w_codebook_center=config.w_codebook_center,
        w_chart_center_mean=config.w_chart_center_mean,
        w_chart_center_radius=config.w_chart_center_radius,
        chart_center_radius_max=config.chart_center_radius_max,
        w_chart_center_sep=config.w_chart_center_sep,
        chart_center_sep_margin=config.chart_center_sep_margin,
        w_chart_collapse=config.w_chart_collapse,
        w_code_collapse=config.w_code_collapse * scales["code_usage_scale"],
        code_usage_entropy_low=config.code_usage_h_low,
        code_usage_entropy_high=config.code_usage_h_high,
        w_code_collapse_temperature=config.code_usage_temperature,
        w_window=config.w_window,
        w_window_eps_ground=config.w_window_eps_ground,
        w_consistency=config.w_consistency,
        w_perp=config.w_perp,
        lr_chart_centers_scale=config.lr_chart_centers_scale,
        lr_codebook_scale=config.lr_codebook_scale,
    )


# ---------------------------------------------------------------------------
# Atlas alignment
# ---------------------------------------------------------------------------


@torch.no_grad()
def _sync_rl_atlas(
    model: TopoEncoderPrimitives,
    world_model: GeometricWorldModel,
    critic: nn.Module,
    action_encoder: GeometricActionEncoder,
    action_decoder: GeometricActionBoundaryDecoder,
) -> None:
    """Keep all chart-conditioned RL modules on the encoder's atlas."""
    phase1_centers = getattr(model.encoder, "chart_centers", None)
    if phase1_centers is None:
        return

    safe_centers = _project_to_ball(phase1_centers.detach())
    world_model.bind_chart_centers(safe_centers, freeze=True)

    enc_centers = safe_centers.to(
        device=action_encoder.visual_chart_tok.chart_centers.device,
        dtype=action_encoder.visual_chart_tok.chart_centers.dtype,
    )
    action_encoder.visual_chart_tok.chart_centers.copy_(enc_centers)
    action_encoder.visual_chart_tok.chart_centers.requires_grad_(False)

    dec_centers = safe_centers.to(
        device=action_decoder.visual_chart_tok.chart_centers.device,
        dtype=action_decoder.visual_chart_tok.chart_centers.dtype,
    )
    action_decoder.visual_chart_tok.chart_centers.copy_(dec_centers)
    action_decoder.visual_chart_tok.chart_centers.requires_grad_(False)

    critic_centers = safe_centers.to(
        device=critic.chart_tok.chart_centers.device,
        dtype=critic.chart_tok.chart_centers.dtype,
    )
    critic.chart_tok.chart_centers.copy_(critic_centers)
    critic.chart_tok.chart_centers.requires_grad_(False)


# ---------------------------------------------------------------------------
# Per-chart diagnostics
# ---------------------------------------------------------------------------


def _per_chart_diagnostics(
    K_chart: torch.Tensor,
    K_code: torch.Tensor,
    router_weights: torch.Tensor,
    z_geo: torch.Tensor,
    num_charts: int,
    codes_per_chart: int,
) -> dict[str, float]:
    """Compute per-chart metrics from flattened encoder outputs."""
    metrics: dict[str, float] = {}

    K_chart = K_chart.long()
    K_code = K_code.long()

    counts = torch.bincount(K_chart, minlength=num_charts).to(dtype=z_geo.dtype)
    total = counts.sum().clamp(min=1.0)
    fracs = counts / total

    for k, frac in enumerate(fracs.tolist()):
        metrics[f"chart/{k}/usage"] = frac

    # Usage entropy (higher = more balanced)
    fracs_safe = fracs.clamp(min=1e-8)
    usage_entropy = -(fracs_safe * fracs_safe.log()).sum()
    metrics["chart/usage_entropy"] = float(usage_entropy)
    metrics["chart/usage_max_frac"] = float(fracs.max())
    metrics["chart/active_charts"] = float((counts > 0).sum())

    z_norms = z_geo.norm(dim=-1)
    z_norm_sums = torch.bincount(K_chart, weights=z_norms, minlength=num_charts)
    z_norm_means = z_norm_sums / counts.clamp(min=1.0)

    flat_code_idx = K_chart * codes_per_chart + K_code
    code_counts = torch.bincount(
        flat_code_idx,
        minlength=num_charts * codes_per_chart,
    ).to(dtype=z_geo.dtype).reshape(num_charts, codes_per_chart)
    code_mass = code_counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
    code_probs = code_counts / code_mass
    code_entropy = -(code_probs * code_probs.clamp(min=1e-8).log()).sum(dim=-1)
    code_perplexity = code_entropy.exp()
    active_codes = (code_counts > 0).sum(dim=-1).to(dtype=z_geo.dtype)
    active_symbols_total = float(active_codes.sum())

    for k in range(num_charts):
        metrics[f"chart/{k}/z_norm"] = float(z_norm_means[k])
        metrics[f"chart/{k}/code_entropy"] = float(code_entropy[k])
        metrics[f"chart/{k}/code_perplexity"] = float(code_perplexity[k])
        metrics[f"chart/{k}/active_codes"] = float(active_codes[k])

    # Router confidence: mean of max router weight (1.0 = perfectly confident)
    metrics["chart/router_confidence"] = float(router_weights.max(dim=-1).values.mean())

    # Top-2 gap
    if num_charts >= 2:
        top2 = router_weights.topk(2, dim=-1).values  # [B, 2]
        metrics["chart/top2_gap"] = float((top2[:, 0] - top2[:, 1]).mean())

    metrics["chart/active_symbols"] = active_symbols_total
    metrics["chart/active_symbol_fraction"] = active_symbols_total / (num_charts * codes_per_chart)

    return metrics


def _phase1_controller_metrics(
    metrics: dict[str, float],
    config: DreamerConfig,
) -> dict[str, float]:
    """Map Dreamer encoder diagnostics to the adaptive Phase 1 controller inputs."""
    active_entropies: list[float] = []
    for chart_idx in range(config.num_charts):
        active_codes = metrics.get(f"chart/{chart_idx}/active_codes", 0.0)
        if active_codes > 0.0:
            active_entropies.append(metrics.get(f"chart/{chart_idx}/code_entropy", 0.0))

    if active_entropies:
        code_entropy_mean_active = float(np.mean(active_entropies))
    else:
        code_entropy_mean_active = 0.0

    return {
        "soft_top1_prob_mean": float(metrics.get("enc/top1_prob_mean", 0.0)),
        "soft_I_XK": float(metrics.get("enc/I_XK", 0.0)),
        "hard_entropy": float(metrics.get("chart/usage_entropy", 0.0)),
        "code_entropy_mean_active": code_entropy_mean_active,
    }


def _transition_valid_mask(dones: torch.Tensor) -> torch.Tensor:
    """Mask valid replay transitions, excluding padded steps after termination."""
    if dones.numel() == 0:
        return dones
    prefix = torch.cat([torch.ones_like(dones[:, :1]), 1.0 - dones[:, :-1]], dim=1)
    return torch.cumprod(prefix, dim=1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over entries where ``mask`` is one."""
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def _discounted_return_to_go(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute discounted replay return-to-go until the first terminal step."""
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)
    running = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        running = rewards[:, t] + gamma * running * (1.0 - dones[:, t])
        returns[:, t] = running
    return returns


def _discounted_sum(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted cumulative reward over a fixed imagined horizon."""
    horizon = rewards.shape[1]
    if horizon == 0:
        return rewards.sum(dim=1)
    exponents = torch.arange(horizon, device=rewards.device, dtype=rewards.dtype)
    discounts = torch.pow(rewards.new_full((), gamma), exponents).unsqueeze(0)
    return (rewards * discounts).sum(dim=1)


def _value_calibration_error(
    values: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    num_bins: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute an ECE-style calibration error for scalar values."""
    valid = mask.bool()
    if not valid.any():
        zero = values.new_zeros(())
        return zero, zero, 0.0

    v = values[valid]
    t = targets[valid]
    v_min = v.min()
    v_max = v.max()
    if (v_max - v_min).abs() < 1e-8:
        gap = (v.mean() - t.mean()).abs()
        return gap, gap, 1.0

    edges = torch.linspace(v_min, v_max, num_bins + 1, device=v.device, dtype=v.dtype)
    cal_err = values.new_zeros(())
    cal_max = values.new_zeros(())
    nonempty = 0
    for idx in range(num_bins):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == num_bins - 1:
            in_bin = (v >= lower) & (v <= upper)
        else:
            in_bin = (v >= lower) & (v < upper)
        if not in_bin.any():
            continue
        gap = (v[in_bin].mean() - t[in_bin].mean()).abs()
        cal_err = cal_err + in_bin.float().mean() * gap
        cal_max = torch.maximum(cal_max, gap)
        nonempty += 1

    return cal_err, cal_max, float(nonempty)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def _train_step(
    model: TopoEncoderPrimitives,
    jump_op: FactorizedJumpOperator,
    dyn_trans_model: DynamicsTransitionModel,
    world_model: GeometricWorldModel,
    reward_head: RewardHead,
    critic: nn.Module,
    action_encoder: GeometricActionEncoder,
    action_decoder: GeometricActionBoundaryDecoder,
    optimizer_enc: torch.optim.Optimizer,
    optimizer_wm: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer | None,
    optimizer_boundary: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    config: DreamerConfig,
    phase1_cfg: VLAConfig,
    epoch: int,
    current_hard_routing: bool,
    current_tau: float,
    obs_normalizer: ObservationNormalizer | None = None,
    compute_diagnostics: bool = True,
) -> dict[str, float]:
    """One training iteration.

    1. Encoder: shared-dynamics TopoEncoder losses + Markov closure/Zeno
    2. Boundary: infer replay controls and decode deterministic motor actions
    3. WM + reward: dynamics losses on DETACHED encoder outputs
    4. Critic: PDE-first value-field solve with replay boundary anchors
    """
    metrics: dict[str, float] = {}
    step_t0 = time.perf_counter()

    obs_raw = batch["obs"]  # [B, T+1, obs_dim]
    obs = obs_normalizer.normalize_tensor(obs_raw) if obs_normalizer is not None else obs_raw
    actions = batch["actions"][:, :-1]  # [B, T, A]
    rewards = batch["rewards"][:, :-1]  # [B, T]
    B, T, _A = actions.shape
    H_wm = min(T, max(1, int(config.wm_prediction_horizon), int(config.imagination_horizon)))
    shared_critic = _shared_value_field(world_model, critic)

    # =====================================================================
    # 1. Encoder training (shared codebook + Markov closure)
    # =====================================================================
    t_section = time.perf_counter()
    def _encode_sequence() -> tuple[
        torch.Tensor, torch.Tensor, dict[str, float],
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        flat_obs = obs.reshape(B * (T + 1), -1)
        (
            base_loss, zn_reg_loss, enc_metrics,
            z_geo_flat, enc_w_flat, K_ch_flat, _zn_flat, _ztex_flat,
            c_bar_flat, K_code_flat, _zq_flat, v_local_flat,
        ) = _compute_encoder_losses(
            flat_obs,
            model,
            jump_op,
            config,
            epoch,
            hard_routing=current_hard_routing,
            hard_routing_tau=current_tau,
            phase1_config=phase1_cfg,
        )

        return (
            base_loss,
            zn_reg_loss,
            enc_metrics,
            z_geo_flat.reshape(B, T + 1, -1),
            enc_w_flat.reshape(B, T + 1, -1),
            K_ch_flat.reshape(B, T + 1),
            K_code_flat.reshape(B, T + 1),
            c_bar_flat.reshape(B, T + 1, -1),
            v_local_flat.reshape(B, T + 1, -1),
        )

    if config.freeze_encoder:
        with torch.no_grad():
            (
                base_loss,
                zn_reg_loss,
                enc_metrics,
                z_all,
                rw_all,
                K_all,
                K_code_all,
                c_bar_all,
                v_local_all,
            ) = _encode_sequence()
            L_dyn, dyn_metrics, _K_code_dyn = compute_dynamics_markov_loss(
                model.encoder,
                dyn_trans_model,
                v_local_all,
                rw_all,
                c_bar_all,
                K_all,
                actions,
                transition_weight=config.w_dyn_transition,
                zeno_weight=config.w_zeno,
                zeno_mode=config.zeno_mode,
            )
        metrics["enc/grad_norm"] = 0.0
        metrics["enc/router_grad_norm"] = 0.0
        metrics["enc/codebook_grad_norm"] = 0.0
        metrics["enc/centers_grad_norm"] = 0.0
        metrics["enc/val_proj_grad_norm"] = 0.0
        metrics["enc/soft_equiv_grad_norm"] = 0.0
    else:
        optimizer_enc.zero_grad()
        (
            base_loss,
            zn_reg_loss,
            enc_metrics,
            z_all,
            rw_all,
            K_all,
            K_code_all,
            c_bar_all,
            v_local_all,
        ) = _encode_sequence()
        L_dyn, dyn_metrics, _K_code_dyn = compute_dynamics_markov_loss(
            model.encoder,
            dyn_trans_model,
            v_local_all,
            rw_all,
            c_bar_all,
            K_all,
            actions,
            transition_weight=config.w_dyn_transition,
            zeno_weight=config.w_zeno,
            zeno_mode=config.zeno_mode,
        )
        L_enc = config.encoder_loss_scale * (base_loss + zn_reg_loss + L_dyn)
        L_enc.backward()
        grad_breakdown = _phase1_grad_breakdown(model)
        enc_grad = nn.utils.clip_grad_norm_(_optimizer_parameters(optimizer_enc), config.grad_clip)
        optimizer_enc.step()
        metrics["enc/L_total"] = float(L_enc)
        metrics["enc/grad_norm"] = float(enc_grad)
        for key, value in grad_breakdown.items():
            metrics[f"enc/{key}"] = value
        for key, value in dyn_metrics.items():
            metrics[f"enc/{key}"] = value

    if config.freeze_encoder:
        metrics["enc/L_total"] = float(config.encoder_loss_scale * (base_loss + zn_reg_loss + L_dyn))
        for key, value in dyn_metrics.items():
            metrics[f"enc/{key}"] = value

    for key, value in enc_metrics.items():
        metrics[f"enc/{key}"] = value
    metrics["time/encoder"] = time.perf_counter() - t_section

    t_section = time.perf_counter()
    _sync_rl_atlas(model, world_model, critic, action_encoder, action_decoder)
    metrics["time/atlas_sync"] = time.perf_counter() - t_section

    # Per-chart diagnostics (always computed, even if encoder frozen)
    if compute_diagnostics:
        t_section = time.perf_counter()
        chart_diag = _per_chart_diagnostics(
            K_all.reshape(-1),
            K_code_all.reshape(-1),
            rw_all.reshape(-1, config.num_charts),
            z_all.reshape(-1, config.latent_dim),
            config.num_charts,
            config.codes_per_chart,
        )
        metrics.update(chart_diag)
        metrics["time/chart_diag"] = time.perf_counter() - t_section

    # =====================================================================
    # 2. Boundary / control training
    # =====================================================================
    rw_all = rw_all.detach()
    z_all = z_all.detach()

    z_0 = z_all[:, 0]
    rw_0 = rw_all[:, 0]
    z_targets = z_all[:, 1:]  # [B, T, D]
    z_prev = z_all[:, :-1]
    rw_prev = rw_all[:, :-1]
    z_prev_flat = z_prev.reshape(-1, config.latent_dim)
    rw_prev_flat = rw_prev.reshape(-1, config.num_charts)
    actions_flat = actions.reshape(-1, config.action_dim)

    replay_controls = batch.get("controls")
    replay_controls_tan = batch.get("controls_tan")
    replay_controls_cov = batch.get("controls_cov")
    replay_control_valid = batch.get("control_valid")
    replay_action_means = batch.get("action_means")
    replay_motor_macro_probs = batch.get("motor_macro_probs")
    replay_motor_nuisance = batch.get("motor_nuisance")
    replay_motor_compliance = batch.get("motor_compliance")
    if replay_controls is None:
        replay_controls = torch.zeros(B, T + 1, config.latent_dim, device=actions.device)
    if replay_controls_cov is None:
        replay_controls_cov = replay_controls
    if replay_controls_tan is None:
        replay_controls_tan = raise_control(
            z_all.reshape(-1, config.latent_dim),
            replay_controls_cov.reshape(-1, config.latent_dim),
            metric=world_model.metric,
        ).reshape(B, T + 1, config.latent_dim)
    if replay_control_valid is None:
        replay_control_valid = torch.zeros(B, T + 1, device=actions.device)
    if replay_action_means is None:
        replay_action_means = batch["actions"]
    num_action_macros = config.num_action_macros
    if replay_motor_macro_probs is None:
        replay_motor_macro_probs = torch.zeros(
            B,
            T + 1,
            num_action_macros,
            device=actions.device,
        )
    if replay_motor_nuisance is None:
        replay_motor_nuisance = torch.zeros(B, T + 1, config.latent_dim, device=actions.device)
    if replay_motor_compliance is None:
        replay_motor_compliance = torch.zeros(
            B,
            T + 1,
            config.action_dim,
            config.action_dim,
            device=actions.device,
        )

    replay_controls_cov = replay_controls_cov[:, :-1]
    replay_controls_tan = replay_controls_tan[:, :-1]
    replay_control_valid = replay_control_valid[:, :-1]
    replay_action_means = replay_action_means[:, :-1]
    replay_motor_macro_probs = replay_motor_macro_probs[:, :-1]
    replay_motor_nuisance = replay_motor_nuisance[:, :-1]
    replay_motor_compliance = replay_motor_compliance[:, :-1]
    replay_controls_cov_flat = replay_controls_cov.reshape(-1, config.latent_dim)
    replay_controls_tan_flat = replay_controls_tan.reshape(-1, config.latent_dim)
    replay_control_valid_flat = replay_control_valid.reshape(-1, 1)
    replay_action_means_flat = replay_action_means.reshape(-1, config.action_dim)
    replay_motor_macro_probs_flat = replay_motor_macro_probs.reshape(-1, num_action_macros)
    replay_motor_nuisance_flat = replay_motor_nuisance.reshape(-1, config.latent_dim)
    replay_motor_compliance_flat = replay_motor_compliance.reshape(
        -1,
        config.action_dim,
        config.action_dim,
    )

    t_section = time.perf_counter()
    optimizer_boundary.zero_grad()
    inferred_boundary = action_encoder(z_prev_flat, actions_flat, rw_prev_flat)
    inferred_controls_tan = inferred_boundary["control_tan"]
    inferred_controls_cov = lower_control(
        z_prev_flat,
        inferred_controls_tan,
        metric=world_model.metric,
    )
    inferred_motor_macro_probs = inferred_boundary["macro_probs"]
    inferred_motor_nuisance = inferred_boundary["motor_nuisance"]
    inferred_motor_compliance = inferred_boundary["motor_compliance"]
    recon_actions = action_decoder.decode(
        z_prev_flat,
        inferred_controls_tan,
        rw_prev_flat,
        macro_probs=inferred_motor_macro_probs,
        motor_nuisance=inferred_motor_nuisance,
        motor_compliance=inferred_motor_compliance,
    )
    cycle_boundary = action_encoder(z_prev_flat, recon_actions.detach(), rw_prev_flat)
    target_decoded_actions = action_decoder.decode(
        z_prev_flat,
        replay_controls_tan_flat,
        rw_prev_flat,
        macro_probs=replay_motor_macro_probs_flat,
        motor_nuisance=replay_motor_nuisance_flat,
        motor_compliance=replay_motor_compliance_flat,
    )
    with _freeze_modules(critic):
        value_controls_cov, value_controls_tan, _ = critic_control_field(
            critic,
            z_prev_flat,
            rw_prev_flat,
            metric=world_model.metric,
            create_graph=False,
        )
    value_controls_cov = value_controls_cov.detach()
    value_controls_tan = value_controls_tan.detach()

    L_action_recon = (recon_actions - replay_action_means_flat).pow(2).mean()
    L_control_cycle = (cycle_boundary["control_tan"] - inferred_controls_tan.detach()).pow(2).mean()
    L_nuisance_cycle = (
        cycle_boundary["motor_nuisance"] - inferred_motor_nuisance.detach()
    ).pow(2).mean()
    L_compliance_cycle = (
        cycle_boundary["motor_compliance"] - inferred_motor_compliance.detach()
    ).pow(2).mean()
    L_boundary_cycle = L_control_cycle + L_nuisance_cycle + L_compliance_cycle
    control_valid_denom = replay_control_valid_flat.sum().clamp(min=1.0)
    control_supervise_err = F.smooth_l1_loss(
        inferred_controls_tan,
        replay_controls_tan_flat,
        reduction="none",
    ).mean(dim=-1, keepdim=True)
    L_control_supervise = (
        (control_supervise_err * replay_control_valid_flat).sum()
        / control_valid_denom
    )
    decoder_supervise_err = (
        target_decoded_actions - replay_action_means_flat
    ).pow(2).mean(dim=-1, keepdim=True)
    L_decoder_supervise = (
        (decoder_supervise_err * replay_control_valid_flat).sum()
        / control_valid_denom
    )
    macro_supervise_err = -(
        replay_motor_macro_probs_flat
        * torch.log(inferred_motor_macro_probs.clamp(min=1e-8))
    ).sum(dim=-1, keepdim=True)
    L_macro_supervise = (
        (macro_supervise_err * replay_control_valid_flat).sum()
        / control_valid_denom
    )
    nuisance_supervise_err = F.smooth_l1_loss(
        inferred_motor_nuisance,
        replay_motor_nuisance_flat,
        reduction="none",
    ).mean(dim=-1, keepdim=True)
    L_motor_nuisance_supervise = (
        (nuisance_supervise_err * replay_control_valid_flat).sum()
        / control_valid_denom
    )
    compliance_supervise_err = F.smooth_l1_loss(
        inferred_motor_compliance,
        replay_motor_compliance_flat,
        reduction="none",
    ).mean(dim=(-2, -1), keepdim=True)
    L_motor_compliance_supervise = (
        (compliance_supervise_err * replay_control_valid_flat).sum()
        / control_valid_denom
    )
    value_intent_cos = F.cosine_similarity(
        inferred_controls_tan,
        value_controls_tan,
        dim=-1,
        eps=1e-8,
    ).unsqueeze(-1)
    L_value_intent_cos = (
        ((1.0 - value_intent_cos) * replay_control_valid_flat).sum()
        / control_valid_denom
    )
    value_intent_huber_err = F.smooth_l1_loss(
        inferred_controls_tan,
        value_controls_tan,
        reduction="none",
    ).mean(dim=-1, keepdim=True)
    L_value_intent_huber = (
        (value_intent_huber_err * replay_control_valid_flat).sum()
        / control_valid_denom
    )
    L_boundary = (
        config.w_action_recon * (L_action_recon + L_decoder_supervise)
        + config.w_control_cycle * L_boundary_cycle
        + config.w_control_supervise * L_control_supervise
        + config.w_macro_supervise * L_macro_supervise
        + config.w_motor_nuisance_supervise * L_motor_nuisance_supervise
        + config.w_motor_compliance_supervise * L_motor_compliance_supervise
        + config.w_value_intent_align * (L_value_intent_cos + L_value_intent_huber)
    )
    L_boundary.backward()
    boundary_grad = nn.utils.clip_grad_norm_(
        list(action_encoder.parameters()) + list(action_decoder.parameters()),
        config.grad_clip,
    )
    optimizer_boundary.step()
    metrics["boundary/L_total"] = float(L_boundary)
    metrics["boundary/L_action_recon"] = float(L_action_recon)
    metrics["boundary/L_decoder_supervise"] = float(L_decoder_supervise)
    metrics["boundary/L_control_cycle"] = float(L_boundary_cycle)
    metrics["boundary/L_control_cycle_control"] = float(L_control_cycle)
    metrics["boundary/L_control_cycle_nuisance"] = float(L_nuisance_cycle)
    metrics["boundary/L_control_cycle_compliance"] = float(L_compliance_cycle)
    metrics["boundary/L_control_supervise"] = float(L_control_supervise)
    metrics["boundary/L_macro_supervise"] = float(L_macro_supervise)
    metrics["boundary/L_motor_nuisance_supervise"] = float(L_motor_nuisance_supervise)
    metrics["boundary/L_motor_compliance_supervise"] = float(L_motor_compliance_supervise)
    metrics["boundary/L_value_intent_cos"] = float(L_value_intent_cos)
    metrics["boundary/L_value_intent_huber"] = float(L_value_intent_huber)
    metrics["boundary/value_intent_cos"] = float(
        ((value_intent_cos * replay_control_valid_flat).sum() / control_valid_denom).detach()
    )
    metrics["boundary/value_intent_l2"] = float(
        (
            (inferred_controls_tan - value_controls_tan).pow(2).mean(dim=-1, keepdim=True)
            * replay_control_valid_flat
        ).sum()
        / control_valid_denom
    )
    inferred_roundtrip_tan = raise_control(
        z_prev_flat,
        inferred_controls_cov.detach(),
        metric=world_model.metric,
    )
    value_roundtrip_cov = lower_control(
        z_prev_flat,
        value_controls_tan,
        metric=world_model.metric,
    )
    metrics["boundary/control_raise_err"] = float(
        (inferred_roundtrip_tan - inferred_controls_tan.detach()).abs().mean()
    )
    metrics["boundary/control_lower_err"] = float(
        (value_roundtrip_cov - value_controls_cov).abs().mean()
    )
    metrics["boundary/macro_entropy"] = float(
        -(inferred_motor_macro_probs * inferred_motor_macro_probs.clamp(min=1e-8).log())
        .sum(dim=-1)
        .mean()
    )
    metrics["boundary/macro_active"] = float(
        (inferred_motor_macro_probs.mean(dim=0) > 0.05).float().sum()
    )
    metrics["boundary/motor_nuisance_norm_mean"] = float(
        inferred_motor_nuisance.norm(dim=-1).mean()
    )
    inferred_compliance_diag = inferred_motor_compliance.diagonal(dim1=-2, dim2=-1)
    metrics["boundary/motor_compliance_mean"] = float(inferred_compliance_diag.mean())
    metrics["boundary/motor_compliance_max"] = float(inferred_compliance_diag.max())
    metrics["boundary/motor_compliance_frob_mean"] = float(
        torch.linalg.norm(inferred_motor_compliance, dim=(-2, -1)).mean()
    )
    metrics["boundary/grad_norm"] = float(boundary_grad)
    metrics["time/boundary"] = time.perf_counter() - t_section

    with torch.no_grad():
        inferred_boundary_det = action_encoder(z_prev_flat, actions_flat, rw_prev_flat)
        inferred_controls_tan_det = inferred_boundary_det["control_tan"]
        inferred_controls_cov_det = lower_control(
            z_prev_flat,
            inferred_controls_tan_det,
            metric=world_model.metric,
        )
    control_mask = replay_control_valid.unsqueeze(-1)
    controls_model_cov = torch.where(
        control_mask.bool(),
        replay_controls_cov,
        inferred_controls_cov_det.reshape(B, T, config.latent_dim),
    )
    controls_model_tan = torch.where(
        control_mask.bool(),
        replay_controls_tan,
        inferred_controls_tan_det.reshape(B, T, config.latent_dim),
    )

    # =====================================================================
    # 3. World model + reward training (DETACHED encoder outputs)
    # =====================================================================
    replay_dones = batch["dones"][:, :-1]
    replay_valid = _transition_valid_mask(replay_dones)

    t_section = time.perf_counter()
    optimizer_wm.zero_grad()

    wm_out = world_model(z_0, controls_model_cov[:, :H_wm], rw_0)
    z_pred = wm_out["z_trajectory"]  # [B, T_wm, D]

    T_wm = min(z_pred.shape[1], H_wm)
    z_tgt_wm = z_targets[:, :T_wm]

    L_geo = compute_dynamics_geodesic_loss(z_pred, z_tgt_wm)
    metrics["wm/L_geodesic"] = float(L_geo)

    target_charts = K_all.detach()[:, 1:T_wm + 1]
    L_chart = compute_dynamics_chart_loss(wm_out["chart_logits"][:, :T_wm], target_charts)
    metrics["wm/L_chart"] = float(L_chart)
    chart_probs = F.softmax(wm_out["chart_logits"][:, :T_wm], dim=-1)
    metrics["wm/chart_acc"] = float(
        (wm_out["chart_logits"][:, :T_wm].argmax(dim=-1) == target_charts).float().mean()
    )
    metrics["wm/chart_entropy"] = float(
        -(chart_probs * chart_probs.clamp(min=1e-8).log()).sum(dim=-1).mean()
    )
    metrics["wm/chart_confidence"] = float(chart_probs.max(dim=-1).values.mean())

    controls_model_tan_flat = controls_model_tan.reshape(-1, config.latent_dim)
    reward_info = reward_head.decompose(
        z_prev_flat,
        replay_action_means_flat,
        rw_prev_flat,
        control=controls_model_tan_flat,
    )
    r_pred = reward_info["reward_total"]
    r_cons = reward_info["reward_conservative"]
    r_noncons = reward_info["reward_nonconservative"]
    r_curl = reward_info["reward_curl"]
    L_reward = (r_pred.squeeze(-1) - rewards.reshape(-1)).pow(2).mean()
    L_reward_nonconservative = r_noncons.pow(2).mean()
    metrics["wm/L_reward"] = float(L_reward)
    metrics["wm/L_reward_nonconservative"] = float(L_reward_nonconservative)
    metrics["wm/reward_control_norm_mean"] = float(controls_model_tan_flat.norm(dim=-1).mean())
    metrics["wm/reward_conservative_mean"] = float(r_cons.mean())
    metrics["wm/reward_nonconservative_mean"] = float(r_noncons.mean())
    metrics["wm/reward_nonconservative_frac"] = float(
        r_noncons.abs().mean() / (r_pred.abs().mean() + 1e-8)
    )
    metrics["wm/reward_form_norm_mean"] = float(
        reward_info["reward_form_cov"].norm(dim=-1).mean()
    )
    metrics["wm/reward_curl_norm_mean"] = float(
        torch.linalg.norm(r_curl, dim=(-2, -1)).mean()
    )

    L_momentum = compute_momentum_regularization(wm_out["momenta"], wm_out["z_trajectory"])
    metrics["wm/L_momentum"] = float(L_momentum)

    if "energy_var" in wm_out:
        L_energy = wm_out["energy_var"]
    else:
        L_energy = compute_energy_conservation_loss(
            wm_out["phi_eff"],
            wm_out["momenta"],
            wm_out["z_trajectory"],
        )
    metrics["wm/L_energy"] = float(L_energy)

    L_hodge = compute_hodge_consistency_loss(wm_out["hodge_harmonic_forces"])
    metrics["wm/L_hodge"] = float(L_hodge)

    L_wm_core = (
        config.w_dynamics * L_geo
        + config.w_dynamics * L_chart
        + config.w_reward * L_reward
        + config.w_reward_nonconservative * L_reward_nonconservative
        + config.w_momentum_reg * L_momentum
        + config.w_energy_conservation * L_energy
        + config.w_hodge * L_hodge
    )

    critic_t0 = time.perf_counter()
    replay_values = critic_value(critic, z_prev_flat, rw_prev_flat).reshape(B, T)
    replay_rtg = _discounted_return_to_go(rewards, replay_dones, config.gamma)
    replay_gap = replay_values - replay_rtg
    L_value = _masked_mean(replay_gap.pow(2), replay_valid)
    L_poisson = compute_screened_poisson_loss(
        critic,
        z_prev.detach(),
        None,
        rw_prev.detach(),
        reward_density=r_cons.detach().reshape(B, T, 1),
        kappa=config.screened_poisson_kappa,
    )
    L_critic = L_poisson + config.w_critic * L_value
    metrics["time/critic"] = time.perf_counter() - critic_t0
    metrics["critic/L_critic"] = float(L_critic)
    metrics["critic/L_value"] = float(L_value)
    metrics["critic/L_anchor"] = float(L_value)
    metrics["critic/L_poisson"] = float(L_poisson)

    if shared_critic:
        L_wm_total = L_wm_core + L_critic
        L_wm_total.backward()
        critic_grad = _parameter_grad_norm(list(critic.parameters()))
        wm_grad = nn.utils.clip_grad_norm_(_optimizer_parameters(optimizer_wm), config.grad_clip)
        optimizer_wm.step()
    else:
        L_wm_core.backward()
        wm_grad = nn.utils.clip_grad_norm_(_optimizer_parameters(optimizer_wm), config.grad_clip)
        optimizer_wm.step()

        optimizer_critic.zero_grad()
        L_critic.backward()
        critic_grad = nn.utils.clip_grad_norm_(critic.parameters(), config.grad_clip)
        optimizer_critic.step()

    metrics["wm/grad_norm"] = float(wm_grad)
    metrics["critic/grad_norm"] = float(critic_grad)
    metrics["time/world_model"] = time.perf_counter() - t_section

    # Imagined control-field diagnostics
    imagination = _imagine(
        world_model,
        reward_head,
        critic,
        action_decoder,
        z_0.detach(),
        rw_0.detach(),
        config.imagination_horizon,
    )
    imag_z_states = imagination["z_states"]      # [B, H, D]
    imag_rw_states = imagination["rw_states"]    # [B, H, K]
    imag_controls_tan = imagination["controls_tan"]      # [B, H, D]
    imag_controls_cov = imagination["controls_cov"]      # [B, H, D]
    imag_motor_macro_probs = imagination["motor_macro_probs"]  # [B, H, M]
    imag_motor_nuisance = imagination["motor_nuisance"]  # [B, H, D]
    imag_motor_compliance = imagination["motor_compliance"]  # [B, H, A, A]
    imag_rewards = imagination["rewards"]        # [B, H]
    imag_reward_cons = imagination["reward_conservative"]  # [B, H]
    imag_reward_noncons = imagination["reward_nonconservative"]  # [B, H]
    imag_reward_curl = imagination["reward_curl_norm"]  # [B, H]
    imag_log_std = imagination["action_log_std"] # [B, H, A]
    imag_z = imagination["z_traj"]               # [B, H, D]
    imag_rw = imagination["rw_traj"]             # [B, H, K]
    imag_actions = imagination["actions"]        # [B, H, A]
    discounted_rewards = _discounted_sum(imag_rewards, config.gamma)
    terminal_value = critic_value(critic, imag_z[:, -1], imag_rw[:, -1]).squeeze(-1)
    boundary_value = (config.gamma ** config.imagination_horizon) * terminal_value
    control_objective = discounted_rewards + boundary_value

    # =====================================================================
    # Diagnostic metrics
    # =====================================================================
    if compute_diagnostics:
        t_section = time.perf_counter()
        with torch.no_grad():
            replay_values_all = critic_value(
                critic,
                z_all.reshape(-1, config.latent_dim),
                rw_all.reshape(-1, config.num_charts),
            ).reshape(B, T + 1)
            replay_values_diag = replay_values_all[:, :-1]
            replay_next_values = replay_values_all[:, 1:]
            replay_delta = (
                rewards
                + config.gamma * (1.0 - replay_dones) * replay_next_values
                - replay_values_diag
            )
            replay_gap = replay_values_diag - replay_rtg
            cal_err, cal_max, cal_bins = _value_calibration_error(
                replay_values_diag,
                replay_rtg,
                replay_valid,
            )
            imag_rewards_det = imag_rewards.detach()
            imag_controls_tan_det = imag_controls_tan.detach()
            imag_controls_cov_det = imag_controls_cov.detach()
            imag_motor_macro_probs_det = imag_motor_macro_probs.detach()
            imag_motor_nuisance_det = imag_motor_nuisance.detach()
            imag_motor_compliance_det = imag_motor_compliance.detach()
            imag_reward_cons_det = imag_reward_cons.detach()
            imag_reward_noncons_det = imag_reward_noncons.detach()
            imag_reward_curl_det = imag_reward_curl.detach()
            imag_log_std_det = imag_log_std.detach()
            imag_rw_states_det = imag_rw_states.detach()
            imag_rw_det = imag_rw.detach()
            imag_actions_det = imag_actions.detach()
            discounted_rewards_det = discounted_rewards.detach()
            boundary_value_det = boundary_value.detach()
            control_objective_det = control_objective.detach()

            metrics["policy/control_norm_mean"] = float(imag_controls_tan_det.norm(dim=-1).mean())
            metrics["policy/control_norm_max"] = float(imag_controls_tan_det.norm(dim=-1).max())
            metrics["policy/control_cov_norm_mean"] = float(
                imag_controls_cov_det.norm(dim=-1).mean()
            )
            metrics["policy/control_cov_norm_max"] = float(
                imag_controls_cov_det.norm(dim=-1).max()
            )
            metrics["policy/action_abs_mean"] = float(imag_actions_det.abs().mean())
            metrics["policy/action_sat_frac"] = float(
                (imag_actions_det.abs() > 0.95).float().mean()
            )
            metrics["boundary/texture_std_mean"] = float(imag_log_std_det.exp().mean())
            metrics["boundary/texture_std_max"] = float(imag_log_std_det.exp().max())
            metrics["policy/motor_macro_entropy"] = float(
                -(imag_motor_macro_probs_det * imag_motor_macro_probs_det.clamp(min=1e-8).log())
                .sum(dim=-1)
                .mean()
            )
            metrics["policy/motor_macro_active"] = float(
                (imag_motor_macro_probs_det.mean(dim=(0, 1)) > 0.05).float().sum()
            )
            metrics["policy/motor_nuisance_norm_mean"] = float(
                imag_motor_nuisance_det.norm(dim=-1).mean()
            )
            metrics["policy/motor_compliance_mean"] = float(
                imag_motor_compliance_det.diagonal(dim1=-2, dim2=-1).mean()
            )
            metrics["policy/motor_compliance_max"] = float(
                imag_motor_compliance_det.diagonal(dim1=-2, dim2=-1).max()
            )
            metrics["policy/motor_compliance_frob_mean"] = float(
                torch.linalg.norm(imag_motor_compliance_det, dim=(-2, -1)).mean()
            )
            metrics["imagination/reward_mean"] = float(imag_rewards_det.mean())
            metrics["imagination/reward_std"] = float(imag_rewards_det.std())
            metrics["imagination/reward_conservative_mean"] = float(imag_reward_cons_det.mean())
            metrics["imagination/reward_nonconservative_mean"] = float(
                imag_reward_noncons_det.mean()
            )
            metrics["imagination/reward_nonconservative_frac"] = float(
                imag_reward_noncons_det.abs().mean() / (imag_rewards_det.abs().mean() + 1e-8)
            )
            metrics["imagination/reward_curl_norm_mean"] = float(imag_reward_curl_det.mean())
            metrics["imagination/reward_sum_mean"] = float(imag_rewards_det.sum(dim=1).mean())
            metrics["imagination/reward_only_return_mean"] = float(discounted_rewards_det.mean())
            metrics["imagination/discounted_reward_mean"] = float(discounted_rewards_det.mean())
            metrics["imagination/terminal_value_mean"] = float(terminal_value.detach().mean())
            metrics["imagination/boundary_value_mean"] = float(boundary_value_det.mean())
            metrics["imagination/return_mean"] = float(control_objective_det.mean())
            metrics["imagination/return_std"] = float(control_objective_det.std(unbiased=False))
            metrics["imagination/bootstrap0_mean"] = float(boundary_value_det.mean())
            metrics["imagination/bootstrap_ratio"] = float(
                boundary_value_det.abs().mean()
                / (discounted_rewards_det.abs().mean() + 1e-8)
            )
            metrics["imagination/bootstrap_share"] = float(
                (boundary_value_det.abs() / (control_objective_det.abs() + 1e-8)).mean()
            )
            metrics["imagination/boundary_ratio"] = float(
                boundary_value_det.abs().mean()
                / (discounted_rewards_det.abs().mean() + 1e-8)
            )
            metrics["imagination/router_entropy"] = float(
                -(imag_rw_states_det * imag_rw_states_det.clamp(min=1e-8).log())
                .sum(dim=-1)
                .mean()
            )
            metrics["imagination/router_drift"] = float(
                (imag_rw_det - imag_rw_states_det).abs().mean()
            )
            metrics["critic/value_bias"] = float(_masked_mean(replay_gap, replay_valid))
            metrics["critic/value_abs_err"] = float(_masked_mean(replay_gap.abs(), replay_valid))
            metrics["critic/value_mean"] = float(_masked_mean(replay_values_diag, replay_valid))
            replay_phi_eff = world_model.potential_net.effective_potential(
                z_prev_flat,
                rw_prev_flat,
            ).reshape(B, T)
            metrics["critic/phi_eff_mean"] = float(_masked_mean(replay_phi_eff, replay_valid))
            metrics["critic/value_struct_gap"] = float(
                _masked_mean(replay_phi_eff - replay_values_diag, replay_valid)
            )
            metrics["critic/replay_bellman_mean"] = float(_masked_mean(replay_delta, replay_valid))
            metrics["critic/replay_bellman_abs"] = float(
                _masked_mean(replay_delta.abs(), replay_valid)
            )
            bellman_centered = replay_delta - _masked_mean(replay_delta, replay_valid)
            metrics["critic/replay_bellman_std"] = float(
                (_masked_mean(bellman_centered.pow(2), replay_valid) + 1e-8).sqrt()
            )
            metrics["critic/replay_rtg_mean"] = float(_masked_mean(replay_rtg, replay_valid))
            metrics["critic/replay_rtg_bias"] = float(_masked_mean(replay_gap, replay_valid))
            metrics["critic/replay_rtg_abs_err"] = float(
                _masked_mean(replay_gap.abs(), replay_valid)
            )
            metrics["critic/replay_calibration_err"] = float(cal_err)
            metrics["critic/replay_calibration_max"] = float(cal_max)
            metrics["critic/replay_calibration_bins"] = cal_bins
            metrics["train/replay_horizon"] = float(T)
            metrics["train/wm_horizon"] = float(T_wm)
            metrics["geometric/z_norm_mean"] = float(z_all.norm(dim=-1).mean())
            metrics["geometric/z_norm_max"] = float(z_all.norm(dim=-1).max())
            metrics["geometric/jump_frac"] = float(wm_out["jumped"].float().mean())
            metrics["geometric/hodge_conservative"] = float(
                wm_out["hodge_conservative_ratio"].mean()
            )
            metrics["geometric/hodge_solenoidal"] = float(
                wm_out["hodge_solenoidal_ratio"].mean()
            )
            metrics["geometric/hodge_harmonic"] = float(wm_out["hodge_harmonic_ratio"].mean())
            metrics["geometric/energy_var"] = float(wm_out["energy_var"])
            enc_centers = _project_to_ball(model.encoder.chart_centers.detach())
            wm_centers = _project_to_ball(world_model.potential_net.chart_tok.chart_centers.detach())
            action_encoder_centers = _project_to_ball(
                action_encoder.visual_chart_tok.chart_centers.detach(),
            )
            action_decoder_centers = _project_to_ball(
                action_decoder.visual_chart_tok.chart_centers.detach(),
            )
            critic_centers = _project_to_ball(critic.chart_tok.chart_centers.detach())
            metrics["chart/wm_center_drift"] = float((enc_centers - wm_centers).norm(dim=-1).mean())
            metrics["chart/action_encoder_center_drift"] = float(
                (enc_centers - action_encoder_centers).norm(dim=-1).mean()
            )
            metrics["chart/action_decoder_center_drift"] = float(
                (enc_centers - action_decoder_centers).norm(dim=-1).mean()
            )
            metrics["chart/critic_center_drift"] = float(
                (enc_centers - critic_centers).norm(dim=-1).mean()
            )
        metrics["time/diagnostics"] = time.perf_counter() - t_section

    metrics["time/step"] = time.perf_counter() - step_t0

    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(config: DreamerConfig) -> None:
    """Run Geometric Dreamer training."""
    device = torch.device(config.device)
    print(f"Device: {device}")
    torch.manual_seed(config.seed)
    obs_normalizer: ObservationNormalizer | None = None

    # --- MLflow ---
    mlflow_enabled = False
    if config.mlflow and MLFLOW_AVAILABLE and mlflow is not None:
        if config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        mlflow.start_run()
        try:
            safe = {k: str(v) for k, v in asdict(config).items()}
            mlflow.log_params(safe)
        except Exception:
            pass
        mlflow_enabled = True

    # --- Environment ---
    env = _make_env(config.domain, config.task)
    time_step = env.reset()
    actual_obs_dim = len(_flatten_obs(time_step))
    if actual_obs_dim != config.obs_dim:
        print(f"Overriding obs_dim: {config.obs_dim} -> {actual_obs_dim}")
        config.obs_dim = actual_obs_dim

    print(f"Environment: {config.domain}-{config.task}  "
          f"obs_dim={config.obs_dim}  action_dim={config.action_dim}")

    # --- Models ---
    model = SharedDynTopoEncoder(
        input_dim=config.obs_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_charts=config.num_charts,
        codes_per_chart=config.codes_per_chart,
        covariant_attn=True,
        covariant_attn_tensorization="full",
        soft_equiv_metric=True,
        conv_backbone=False,
        film_conditioning=True,
        commitment_beta=config.commitment_beta,
        codebook_loss_weight=config.codebook_loss_weight,
    ).to(device)
    jump_op = FactorizedJumpOperator(
        num_charts=config.num_charts,
        latent_dim=config.latent_dim,
    ).to(device)
    dyn_trans_model = DynamicsTransitionModel(
        chart_dim=config.latent_dim,
        action_dim=config.action_dim,
        num_charts=config.num_charts,
        codes_per_chart=config.codes_per_chart,
        hidden_dim=config.dyn_transition_hidden_dim,
    ).to(device)

    world_model = GeometricWorldModel(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        control_dim=config.latent_dim,
        num_charts=config.num_charts,
        d_model=config.d_model,
        hidden_dim=config.hidden_dim,
        dt=config.wm_dt,
        gamma_friction=config.wm_gamma_friction,
        T_c=config.wm_T_c,
        alpha_potential=config.wm_alpha_potential,
        beta_curl=config.wm_beta_curl,
        gamma_risk=config.wm_gamma_risk,
        use_boris=config.wm_use_boris,
        use_jump=config.wm_use_jump,
        n_refine_steps=config.wm_n_refine_steps,
        jump_beta=config.wm_jump_beta,
        min_length=config.wm_min_length,
        risk_metric_alpha=config.wm_risk_metric_alpha,
    ).to(device)

    action_encoder = GeometricActionEncoder(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        num_charts=config.num_charts,
        num_action_charts=config.num_action_charts,
        num_action_macros=config.num_action_macros,
        d_model=config.d_model,
    ).to(device)
    action_decoder = GeometricActionBoundaryDecoder(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        num_charts=config.num_charts,
        num_action_charts=config.num_action_charts,
        num_action_macros=config.num_action_macros,
        d_model=config.d_model,
        sigma_motor=config.sigma_motor,
        metric=world_model.metric,
    ).to(device)
    critic = world_model.potential_net

    reward_head = RewardHead(
        potential_net=world_model.potential_net,
        action_dim=config.action_dim,
        d_model=config.d_model,
        metric=world_model.metric,
    ).to(device)

    dyn_trans_model = _maybe_compile_module(
        dyn_trans_model,
        enabled=config.torch_compile,
        mode=config.torch_compile_mode,
    )
    action_encoder = _maybe_compile_module(
        action_encoder,
        enabled=config.torch_compile,
        mode=config.torch_compile_mode,
    )
    action_decoder = _maybe_compile_module(
        action_decoder,
        enabled=config.torch_compile,
        mode=config.torch_compile_mode,
    )
    reward_head = _maybe_compile_module(
        reward_head,
        enabled=config.torch_compile,
        mode=config.torch_compile_mode,
    )
    shared_critic = _shared_value_field(world_model, critic)

    # --- Parameter counts ---
    def _count(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    print(f"Encoder:     {_count(model.encoder):,} params")
    print(f"Decoder:     {_count(model.decoder):,} params")
    print(f"Jump op:     {_count(jump_op):,} params")
    print(f"Dyn trans:   {_count(dyn_trans_model):,} params")
    print(f"World model: {_count(world_model):,} params")
    print(f"Act enc:     {_count(action_encoder):,} params")
    print(f"Act dec:     {_count(action_decoder):,} params")
    if shared_critic:
        print(f"Value field: shared in world model ({_count(critic):,} params)")
    else:
        print(f"Critic:      {_count(critic):,} params")
    print(f"Reward head: {_count(reward_head):,} params  (chart_tok/z_embed shared)")

    # --- Optimizers ---
    encoder_groups = build_encoder_param_groups(
        model,
        jump_op,
        base_lr=config.lr_encoder,
        lr_chart_centers_scale=config.lr_chart_centers_scale,
        lr_codebook_scale=config.lr_codebook_scale,
    )
    encoder_groups.append(
        {
            "params": list(dyn_trans_model.parameters()),
            "lr": config.lr_dyn_transition,
        },
    )
    optimizer_enc = torch.optim.Adam(encoder_groups)

    # WM optimizer: world_model + reward_head (minus shared params)
    reward_own_params = [p for n, p in reward_head.named_parameters()
                         if "chart_tok" not in n and "z_embed" not in n]
    optimizer_wm = torch.optim.Adam([
        {"params": world_model.parameters(), "lr": config.lr_wm},
        {"params": reward_own_params, "lr": config.lr_wm},
    ])

    optimizer_boundary = torch.optim.Adam(
        list(action_encoder.parameters()) + list(action_decoder.parameters()),
        lr=config.lr_actor,
    )
    scheduler_enc = _make_cosine_scheduler(optimizer_enc, config.total_epochs, config.lr_min)
    scheduler_wm = _make_cosine_scheduler(optimizer_wm, config.total_epochs, config.lr_min)
    scheduler_boundary = _make_cosine_scheduler(
        optimizer_boundary,
        config.total_epochs,
        config.lr_min,
    )
    optimizer_critic = None
    scheduler_critic = None
    if not shared_critic:
        optimizer_critic = torch.optim.Adam(critic.parameters(), lr=config.lr_wm)
        scheduler_critic = _make_cosine_scheduler(
            optimizer_critic,
            config.total_epochs,
            config.lr_min,
        )

    # --- Phase 1 config for encoder losses ---
    phase1_state = init_phase1_adaptive_state(config)
    phase1_cfg = _phase1_config(config, phase1_state)

    # --- Load checkpoint ---
    start_epoch = 0
    if config.load_checkpoint and os.path.exists(config.load_checkpoint):
        ckpt = torch.load(config.load_checkpoint, map_location=device, weights_only=False)
        if config.normalize_observations and ckpt.get("obs_normalizer") is not None:
            obs_normalizer = ObservationNormalizer.from_state_dict(
                ckpt["obs_normalizer"],
                device=device,
            )
        if "encoder" in ckpt:
            model.encoder.load_state_dict(ckpt["encoder"], strict=False)
        if "decoder" in ckpt:
            model.decoder.load_state_dict(ckpt["decoder"], strict=False)
        if "jump_op" in ckpt:
            jump_op.load_state_dict(ckpt["jump_op"], strict=False)
        if "dyn_trans_model" in ckpt:
            _unwrap_compiled_module(dyn_trans_model).load_state_dict(
                ckpt["dyn_trans_model"],
                strict=False,
            )
        if "world_model" in ckpt:
            _unwrap_compiled_module(world_model).load_state_dict(ckpt["world_model"], strict=False)
        if "action_encoder" in ckpt:
            _unwrap_compiled_module(action_encoder).load_state_dict(
                ckpt["action_encoder"],
                strict=False,
            )
        if "action_decoder" in ckpt:
            _unwrap_compiled_module(action_decoder).load_state_dict(
                ckpt["action_decoder"],
                strict=False,
            )
        if "critic" in ckpt and not shared_critic:
            _unwrap_compiled_module(critic).load_state_dict(ckpt["critic"], strict=False)
        if "reward_head" in ckpt:
            _unwrap_compiled_module(reward_head).load_state_dict(
                ckpt["reward_head"],
                strict=False,
            )
        if "optimizer_enc" in ckpt:
            optimizer_enc.load_state_dict(ckpt["optimizer_enc"])
        if "optimizer_wm" in ckpt:
            optimizer_wm.load_state_dict(ckpt["optimizer_wm"])
        if "optimizer_boundary" in ckpt:
            optimizer_boundary.load_state_dict(ckpt["optimizer_boundary"])
        if optimizer_critic is not None and "optimizer_critic" in ckpt:
            optimizer_critic.load_state_dict(ckpt["optimizer_critic"])
        if "scheduler_enc" in ckpt:
            scheduler_enc.load_state_dict(ckpt["scheduler_enc"])
        if "scheduler_wm" in ckpt:
            scheduler_wm.load_state_dict(ckpt["scheduler_wm"])
        if "scheduler_boundary" in ckpt:
            scheduler_boundary.load_state_dict(ckpt["scheduler_boundary"])
        if scheduler_critic is not None and "scheduler_critic" in ckpt:
            scheduler_critic.load_state_dict(ckpt["scheduler_critic"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Loaded checkpoint from {config.load_checkpoint} (epoch {start_epoch})")

    if config.freeze_encoder:
        model.encoder.eval()
        model.decoder.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        jump_op.eval()
        dyn_trans_model.eval()
        for p in jump_op.parameters():
            p.requires_grad_(False)
        for p in dyn_trans_model.parameters():
            p.requires_grad_(False)
        print("Encoder + decoder frozen.")

    _sync_rl_atlas(model, world_model, critic, action_encoder, action_decoder)
    print("Bound RL chart centers to encoder atlas.")

    # --- Replay buffer ---
    buffer = SequenceReplayBuffer(
        capacity=config.buffer_capacity,
        seq_len=config.seq_len,
    )

    # --- Seed episodes (random policy) ---
    print(f"Collecting {config.seed_episodes} seed episodes...")
    seed_episodes_data: list[dict[str, np.ndarray]] = []
    for i in range(config.seed_episodes):
        ep = _collect_episode(
            env,
            None,
            None,
            model,
            device,
            config.latent_dim,
            config.num_action_macros,
            obs_normalizer=obs_normalizer,
            action_repeat=config.action_repeat,
            max_steps=config.max_episode_steps,
            hard_routing=config.hard_routing,
            hard_routing_tau=config.hard_routing_tau,
            use_motor_texture=config.use_motor_texture,
        )
        buffer.add_episode(ep)
        seed_episodes_data.append(ep)
        ep_r = ep["rewards"].sum()
        print(f"  Seed {i + 1}/{config.seed_episodes}: reward={ep_r:.1f}  len={len(ep['obs'])}")

    if config.normalize_observations and obs_normalizer is None:
        if not seed_episodes_data:
            msg = "Observation normalization requires at least one seed episode or checkpoint stats."
            raise ValueError(msg)
        obs_normalizer = ObservationNormalizer.from_episodes(
            seed_episodes_data,
            device=device,
            min_std=config.obs_norm_min_std,
        )
        std_cpu = obs_normalizer.std.detach().cpu()
        print(
            "Observation normalization: "
            f"min_std={std_cpu.min().item():.4f}  "
            f"mean_std={std_cpu.mean().item():.4f}  "
            f"max_std={std_cpu.max().item():.4f}",
        )

    # --- Main training loop ---
    total_env_steps = buffer.total_steps
    episode_rewards: list[float] = []
    best_eval_reward = -float("inf")

    _tpb = config.batch_size * max(config.seq_len, 1)
    _est_upd = config.updates_per_epoch if config.updates_per_epoch > 0 else max(1, buffer.total_steps // _tpb)
    print(f"\nStarting training for {config.total_epochs} epochs "
          f"(~{_est_upd} updates/epoch, buffer={buffer.total_steps} steps, "
          f"capacity={config.buffer_capacity})")
    print("=" * 80)

    for epoch in range(start_epoch, config.total_epochs):
        t0 = time.perf_counter()
        collect_time = 0.0

        # --- Data collection ---
        if config.use_gas:
            if epoch % config.gas_collect_every == 0:
                collect_t0 = time.perf_counter()
                episodes, gas_info = _collect_gas_episodes(
                    critic,
                    action_decoder,
                    model,
                    device,
                    config,
                    obs_normalizer=obs_normalizer,
                )
                for ep in episodes:
                    buffer.add_episode(ep)
                total_env_steps += config.gas_walkers * config.gas_steps * config.action_repeat
                episode_rewards.append(gas_info["gas/max_reward"])
                collect_time += time.perf_counter() - collect_t0
                print(f"  GAS  episodes={gas_info['gas/n_episodes']}  "
                      f"transitions={gas_info['gas/transitions']}  "
                      f"max_rew={gas_info['gas/max_reward']:.2f}  "
                      f"alive={gas_info['gas/alive_frac']:.2f}  "
                      f"clones={gas_info['gas/total_clones']:.0f}")
        else:
            if epoch % config.collect_every == 0:
                collect_t0 = time.perf_counter()
                ep = _collect_episode(
                    env,
                    critic,
                    action_decoder,
                    model,
                    device,
                    config.latent_dim,
                    config.num_action_macros,
                    obs_normalizer=obs_normalizer,
                    action_repeat=config.action_repeat,
                    max_steps=config.max_episode_steps,
                    hard_routing=config.hard_routing,
                    hard_routing_tau=config.hard_routing_tau,
                    use_motor_texture=config.use_motor_texture,
                )
                buffer.add_episode(ep)
                ep_reward = ep["rewards"].sum()
                episode_rewards.append(ep_reward)
                total_env_steps += len(ep["obs"])
                collect_time += time.perf_counter() - collect_t0

        # --- Training steps (multiple updates per epoch) ---
        if not config.freeze_encoder:
            model.train()
            jump_op.train()
            dyn_trans_model.train()
        world_model.train()
        action_encoder.train()
        action_decoder.train()
        critic.train()
        reward_head.train()

        tokens_per_batch = config.batch_size * max(config.seq_len, 1)
        if config.updates_per_epoch > 0:
            n_updates = config.updates_per_epoch
        else:
            n_updates = max(1, buffer.total_steps // tokens_per_batch)

        current_hard_routing = _use_hard_routing(config, epoch)
        current_tau = _get_hard_routing_tau(config, epoch, config.total_epochs)
        phase1_cfg = _phase1_config(config, phase1_state)

        metrics_accum: dict[str, list[float]] = {}
        sample_time_total = 0.0
        update_time_total = 0.0
        for _u in range(n_updates):
            sample_t0 = time.perf_counter()
            batch = buffer.sample(config.batch_size, device=device)
            sample_time_total += time.perf_counter() - sample_t0
            step_t0 = time.perf_counter()
            compute_diagnostics = (
                config.diagnostics_every_updates <= 1
                or ((_u + 1) % config.diagnostics_every_updates == 0)
                or (_u == n_updates - 1)
            )

            step_metrics = _train_step(
                model,
                jump_op,
                dyn_trans_model,
                world_model,
                reward_head,
                critic,
                action_encoder,
                action_decoder,
                optimizer_enc,
                optimizer_wm,
                optimizer_critic,
                optimizer_boundary,
                batch, config, phase1_cfg, epoch, current_hard_routing, current_tau,
                obs_normalizer=obs_normalizer,
                compute_diagnostics=compute_diagnostics,
            )
            update_time_total += time.perf_counter() - step_t0
            for k, v in step_metrics.items():
                metrics_accum.setdefault(k, []).append(v)

        # Average metrics across updates
        metrics = {k: float(np.mean(v)) for k, v in metrics_accum.items()}
        metrics["train/updates"] = float(n_updates)
        metrics["time/sample"] = sample_time_total / max(n_updates, 1)
        metrics["time/sample_total"] = sample_time_total
        metrics["time/update_total"] = update_time_total
        metrics["time/collection"] = collect_time
        metrics["time/diagnostics_interval"] = float(max(1, config.diagnostics_every_updates))
        controller_metrics = _phase1_controller_metrics(metrics, config)
        update_phase1_adaptive_state(
            phase1_state,
            config,
            metrics,
            controller_metrics,
            epoch,
        )
        current_scales = phase1_effective_weight_scales(config, phase1_state)
        metrics["phase1/entropy_scale"] = current_scales["entropy_scale"]
        metrics["phase1/chart_usage_scale"] = current_scales["chart_usage_scale"]
        metrics["phase1/chart_ot_scale"] = current_scales["chart_ot_scale"]
        metrics["phase1/code_usage_scale"] = current_scales["code_usage_scale"]

        dt = time.perf_counter() - t0

        # --- Logging ---
        if epoch % config.log_every == 0:
            metrics["env/total_steps"] = total_env_steps
            metrics["env/buffer_episodes"] = buffer.num_episodes
            metrics["env/buffer_steps"] = buffer.total_steps
            metrics["train/wall_time"] = dt
            metrics["train/lr_encoder"] = _current_lr(optimizer_enc)
            metrics["train/lr_wm"] = _current_lr(optimizer_wm)
            metrics["train/lr_boundary"] = _current_lr(optimizer_boundary)
            metrics["train/lr_critic"] = (
                _current_lr(optimizer_critic)
                if optimizer_critic is not None
                else _current_lr(optimizer_wm)
            )
            if episode_rewards:
                metrics["env/last_episode_reward"] = episode_rewards[-1]
                metrics["env/mean_episode_reward_20"] = float(
                    np.mean(episode_rewards[-20:])
                )

            # Console output
            hdr = f"E{epoch:04d} [{n_updates}upd]"
            # Line 1: env + dynamics losses
            line1_keys = [
                ("ep_rew", "env/last_episode_reward"),
                ("rew_20", "env/mean_episode_reward_20"),
                ("L_geo", "wm/L_geodesic"),
                ("L_rew", "wm/L_reward"),
                ("L_chart", "wm/L_chart"),
                ("L_crit", "critic/L_critic"),
                ("L_bnd", "boundary/L_total"),
                ("lr", "train/lr_boundary"),
            ]
            # Line 2: encoder losses
            line2_keys = [
                ("recon", "enc/recon"),
                ("vq", "enc/vq"),
                ("code_H", "enc/H_code_usage"),
                ("code_px", "enc/code_usage_perplexity"),
                ("ch_usage", "enc/chart_usage"),
                ("rtr_mrg", "enc/router_margin"),
                ("enc_gn", "enc/grad_norm"),
            ]
            # Line 3: actor/imagination
            line3_keys = [
                ("ctrl", "policy/control_norm_mean"),
                ("tex", "boundary/texture_std_mean"),
                ("im_rew", "imagination/reward_mean"),
                ("im_ret", "imagination/return_mean"),
                ("value", "critic/value_mean"),
                ("wm_gn", "wm/grad_norm"),
            ]
            # Line 4: geometric + chart diagnostics
            line4_keys = [
                ("z_norm", "geometric/z_norm_mean"),
                ("z_max", "geometric/z_norm_max"),
                ("jump", "geometric/jump_frac"),
                ("cons", "geometric/hodge_conservative"),
                ("sol", "geometric/hodge_solenoidal"),
                ("e_var", "geometric/energy_var"),
                ("ch_ent", "chart/usage_entropy"),
                ("ch_act", "chart/active_charts"),
                ("rtr_conf", "chart/router_confidence"),
            ]
            # Line 5: RL signal quality
            line5_keys = [
                ("obj", "imagination/return_mean"),
                ("dret", "imagination/discounted_reward_mean"),
                ("term", "imagination/terminal_value_mean"),
                ("bnd", "imagination/boundary_ratio"),
                ("chart_acc", "wm/chart_acc"),
                ("chart_ent", "wm/chart_entropy"),
                ("rw_drift", "imagination/router_drift"),
            ]
            line6_keys = [
                ("v_err", "critic/value_abs_err"),
                ("a_sat", "policy/action_sat_frac"),
                ("wm_ctr", "chart/wm_center_drift"),
                ("enc_ctr", "chart/action_encoder_center_drift"),
                ("dec_ctr", "chart/action_decoder_center_drift"),
                ("crt_ctr", "chart/critic_center_drift"),
                ("u_cos", "boundary/value_intent_cos"),
            ]
            line7_keys = [
                ("bnd_x", "imagination/bootstrap_share"),
                ("bell", "critic/replay_bellman_abs"),
                ("bell_s", "critic/replay_bellman_std"),
                ("rtg_e", "critic/replay_rtg_abs_err"),
                ("rtg_b", "critic/replay_rtg_bias"),
                ("cal_e", "critic/replay_calibration_err"),
                ("u_l2", "boundary/value_intent_l2"),
                ("cov_n", "policy/control_cov_norm_mean"),
            ]
            line8_keys = [
                ("col", "time/collection"),
                ("smp", "time/sample"),
                ("enc_t", "time/encoder"),
                ("bnd_t", "time/boundary"),
                ("wm_t", "time/world_model"),
                ("crt_t", "time/critic"),
                ("diag_t", "time/diagnostics"),
            ]

            def _fmt(pairs):
                return "  ".join(
                    f"{label}={metrics[key]:.4f}"
                    for label, key in pairs if key in metrics
                )

            print(f"{hdr}  {_fmt(line1_keys)}  dt={dt:.2f}s")
            print(f"  {'':4s}  {_fmt(line2_keys)}")
            print(f"  {'':4s}  {_fmt(line3_keys)}")
            print(f"  {'':4s}  {_fmt(line4_keys)}")
            print(f"  {'':4s}  {_fmt(line5_keys)}")
            print(f"  {'':4s}  {_fmt(line6_keys)}")
            print(f"  {'':4s}  {_fmt(line7_keys)}")
            print(f"  {'':4s}  {_fmt(line8_keys)}")

            # Per-chart usage line
            usage_parts = []
            for k in range(config.num_charts):
                key = f"chart/{k}/usage"
                if key in metrics:
                    usage_parts.append(f"c{k}={metrics[key]:.2f}")
            if usage_parts:
                active_charts = int(round(metrics.get("chart/active_charts", 0.0)))
                print(
                    f"  {'':4s}  charts: {active_charts}/{config.num_charts} active  "
                    f"{' '.join(usage_parts)}",
                )

            symbol_parts = []
            for k in range(config.num_charts):
                active_key = f"chart/{k}/active_codes"
                entropy_key = f"chart/{k}/code_entropy"
                if active_key in metrics and entropy_key in metrics:
                    active_codes = int(round(metrics[active_key]))
                    symbol_parts.append(
                        f"c{k}={active_codes}/{config.codes_per_chart}(H={metrics[entropy_key]:.2f})",
                    )
            if symbol_parts:
                active_symbols = int(round(metrics.get("chart/active_symbols", 0.0)))
                total_symbols = config.num_charts * config.codes_per_chart
                print(
                    f"  {'':4s}  symbols: {active_symbols}/{total_symbols} active  "
                    f"{' '.join(symbol_parts)}",
                )

            # MLflow
            if mlflow_enabled and log_mlflow_metrics is not None:
                prefixed = {f"phase4/{k}": v for k, v in metrics.items()}
                log_mlflow_metrics(prefixed, step=epoch, enabled=True)

        # --- Evaluation ---
        if epoch % config.eval_every == 0 and epoch > 0:
            model.encoder.eval()
            action_decoder.eval()
            critic.eval()
            eval_metrics = _eval_policy(
                env,
                critic,
                action_decoder,
                model,
                device,
                obs_normalizer=obs_normalizer,
                action_repeat=config.action_repeat,
                num_episodes=config.eval_episodes,
                max_steps=config.max_episode_steps,
                hard_routing=config.hard_routing,
                hard_routing_tau=config.hard_routing_tau,
            )
            print(f"  EVAL  reward={eval_metrics['eval/reward_mean']:.1f} "
                  f"+/- {eval_metrics['eval/reward_std']:.1f}  "
                  f"len={eval_metrics['eval/length_mean']:.0f}")
            if mlflow_enabled and log_mlflow_metrics is not None:
                log_mlflow_metrics(
                    {f"phase4/{k}": v for k, v in eval_metrics.items()},
                    step=epoch, enabled=True,
                )
            if eval_metrics["eval/reward_mean"] > best_eval_reward:
                best_eval_reward = eval_metrics["eval/reward_mean"]
                _save_checkpoint(
                    os.path.join(config.checkpoint_dir, "best.pt"),
                    model,
                    jump_op,
                    dyn_trans_model,
                    world_model,
                    action_encoder,
                    action_decoder,
                    critic,
                    reward_head,
                    optimizer_enc,
                    optimizer_wm,
                    optimizer_boundary,
                    optimizer_critic,
                    scheduler_enc,
                    scheduler_wm,
                    scheduler_boundary,
                    scheduler_critic,
                    epoch, config, eval_metrics, obs_normalizer=obs_normalizer,
                )
                print(f"  New best: {best_eval_reward:.1f}")

        # --- Checkpoint ---
        if epoch % config.checkpoint_every == 0 and epoch > 0:
            _save_checkpoint(
                os.path.join(config.checkpoint_dir, f"epoch_{epoch:05d}.pt"),
                model,
                jump_op,
                dyn_trans_model,
                world_model,
                action_encoder,
                action_decoder,
                critic,
                reward_head,
                optimizer_enc,
                optimizer_wm,
                optimizer_boundary,
                optimizer_critic,
                scheduler_enc,
                scheduler_wm,
                scheduler_boundary,
                scheduler_critic,
                epoch, config, metrics, obs_normalizer=obs_normalizer,
            )

        if not config.freeze_encoder:
            scheduler_enc.step()
        scheduler_wm.step()
        scheduler_boundary.step()
        if scheduler_critic is not None:
            scheduler_critic.step()

    print("=" * 80)
    print("Training complete.")
    if mlflow_enabled and end_mlflow_run is not None:
        end_mlflow_run(enabled=True)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def _save_checkpoint(
    path: str,
    model: TopoEncoderPrimitives,
    jump_op: nn.Module,
    dyn_trans_model: nn.Module,
    world_model: nn.Module,
    action_encoder: nn.Module,
    action_decoder: nn.Module,
    critic: nn.Module,
    reward_head: nn.Module,
    optimizer_enc: torch.optim.Optimizer,
    optimizer_wm: torch.optim.Optimizer,
    optimizer_boundary: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer | None,
    scheduler_enc: torch.optim.lr_scheduler.LRScheduler,
    scheduler_wm: torch.optim.lr_scheduler.LRScheduler,
    scheduler_boundary: torch.optim.lr_scheduler.LRScheduler,
    scheduler_critic: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    config: DreamerConfig,
    metrics: dict | None = None,
    obs_normalizer: ObservationNormalizer | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "config": asdict(config),
        "encoder": {k: v.cpu() for k, v in model.encoder.state_dict().items()},
        "decoder": {k: v.cpu() for k, v in model.decoder.state_dict().items()},
        "jump_op": {k: v.cpu() for k, v in jump_op.state_dict().items()},
        "dyn_trans_model": {
            k: v.cpu() for k, v in _unwrap_compiled_module(dyn_trans_model).state_dict().items()
        },
        "world_model": {
            k: v.cpu() for k, v in _unwrap_compiled_module(world_model).state_dict().items()
        },
        "action_encoder": {
            k: v.cpu() for k, v in _unwrap_compiled_module(action_encoder).state_dict().items()
        },
        "action_decoder": {
            k: v.cpu() for k, v in _unwrap_compiled_module(action_decoder).state_dict().items()
        },
        "critic": {
            k: v.cpu() for k, v in _unwrap_compiled_module(critic).state_dict().items()
        },
        "reward_head": {
            k: v.cpu() for k, v in _unwrap_compiled_module(reward_head).state_dict().items()
        },
        "optimizer_enc": optimizer_enc.state_dict(),
        "optimizer_wm": optimizer_wm.state_dict(),
        "optimizer_boundary": optimizer_boundary.state_dict(),
        "scheduler_enc": scheduler_enc.state_dict(),
        "scheduler_wm": scheduler_wm.state_dict(),
        "scheduler_boundary": scheduler_boundary.state_dict(),
        "metrics": metrics or {},
        "obs_normalizer": None if obs_normalizer is None else obs_normalizer.state_dict(),
    }
    if optimizer_critic is not None:
        state["optimizer_critic"] = optimizer_critic.state_dict()
    if scheduler_critic is not None:
        state["scheduler_critic"] = scheduler_critic.state_dict()
    torch.save(state, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> DreamerConfig:
    import dataclasses

    parser = argparse.ArgumentParser(description="Geometric Dreamer training")
    for f in fields(DreamerConfig):
        # Resolve default (handle default_factory for fields like `device`)
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        else:
            continue  # skip fields with no default
        if f.type == "bool":
            parser.add_argument(f"--{f.name}", action="store_true", default=default)
            parser.add_argument(f"--no-{f.name}", dest=f.name, action="store_false")
        else:
            parser.add_argument(f"--{f.name}", type=type(default), default=default)
    args = parser.parse_args()
    return DreamerConfig(**{f.name: getattr(args, f.name) for f in fields(DreamerConfig)})


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
