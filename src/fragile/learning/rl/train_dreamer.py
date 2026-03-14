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
import contextlib
from dataclasses import asdict, dataclass, fields
import os
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fragile.fractalai.robots.death_conditions import walker_ground_death
from fragile.fractalai.robots.dm_control_env import VectorizedDMControlEnv
from fragile.fractalai.robots.robotic_gas import RoboticFractalGas
from fragile.learning.core.layers import FactorizedJumpOperator
from fragile.learning.core.layers.atlas import _project_to_ball, TopoEncoderPrimitives
from fragile.learning.vla.config import VLAConfig
from fragile.learning.vla.covariant_world_model import GeometricWorldModel
from fragile.learning.vla.losses import (
    compute_dynamics_chart_loss,
    compute_dynamics_geodesic_loss,
    compute_energy_conservation_loss,
    compute_hodge_consistency_loss,
    compute_momentum_regularization,
    compute_screened_poisson_loss,
    zeno_loss,
)
from fragile.learning.vla.optim import build_encoder_param_groups
from fragile.learning.vla.phase1_control import (
    init_phase1_adaptive_state,
    phase1_effective_weight_scales,
    update_phase1_adaptive_state,
)
from fragile.learning.vla.shared_dyn.encoder import SharedDynTopoEncoder
from fragile.learning.vla.train_joint import (
    _compute_encoder_losses,
    _get_hard_routing_tau,
    _phase1_grad_breakdown,
    _use_hard_routing,
)

from .action_manifold import CovariantObsActionClosureModel, symbolize_latent_with_atlas
from .actor import GeometricActor
from .boundary import (
    critic_value,
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


def _conservative_reward_from_value(
    critic: nn.Module,
    z: torch.Tensor,
    rw: torch.Tensor,
    z_next: torch.Tensor,
    rw_next: torch.Tensor,
    gamma: float,
    continuation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the exact discounted reward increment from the value field."""
    v_curr = critic_value(critic, z, rw)
    v_next = critic_value(critic, z_next, rw_next)
    continuation_scale = gamma if continuation is None else gamma * continuation
    reward_conservative = v_curr - continuation_scale * v_next
    return reward_conservative, v_curr, v_next


def _value_covector_from_critic(
    critic: nn.Module,
    z: torch.Tensor,
    rw: torch.Tensor,
    *,
    create_graph: bool = False,
    detach: bool = True,
) -> torch.Tensor:
    """Return the exact state covector field ``dV`` induced by the critic."""
    z_req = z if z.requires_grad else z.detach().requires_grad_(True)
    value = critic_value(critic, z_req, rw.detach()).sum()
    value_covector = torch.autograd.grad(
        value,
        z_req,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    return value_covector.detach() if detach else value_covector


def _parameter_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    """Compute an unclipped gradient norm for diagnostics."""
    grads = [p.grad.detach().norm() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack(grads), p=2))


@contextlib.contextmanager
def _temporary_requires_grad(
    modules: list[tuple[nn.Module, bool]],
) -> None:
    """Temporarily override ``requires_grad`` for a list of modules."""
    saved_states: list[tuple[nn.Parameter, bool]] = []
    for module, requires_grad in modules:
        for param in module.parameters():
            saved_states.append((param, param.requires_grad))
            param.requires_grad_(requires_grad)
    try:
        yield
    finally:
        for param, requires_grad in saved_states:
            param.requires_grad_(requires_grad)


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


def _mark_compile_step_begin() -> None:
    """Hint TorchInductor/CUDAGraphs that a new training step is starting."""
    compiler = getattr(torch, "compiler", None)
    mark_step = getattr(compiler, "cudagraph_mark_step_begin", None)
    if callable(mark_step):
        mark_step()


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


def _policy_action(
    actor: GeometricActor,
    action_model: SharedDynTopoEncoder,
    closure_model: CovariantObsActionClosureModel,
    z: torch.Tensor,
    rw: torch.Tensor,
    z_q: torch.Tensor,
    *,
    use_motor_texture: bool,
    hard_routing: bool = True,
    hard_routing_tau: float = -1.0,
) -> dict[str, torch.Tensor]:
    """Predict an action-manifold ``z_geo`` and decode it to an environment action."""
    if use_motor_texture:
        action_latent, action_latent_mean, log_std = actor.sample_latent(z, rw)
    else:
        action_latent_mean, log_std = actor.forward(z, rw)
        action_latent = action_latent_mean
    action_info = symbolize_latent_with_atlas(
        action_model,
        action_latent,
        hard_routing=hard_routing,
        hard_routing_tau=hard_routing_tau,
    )
    action_info_mean = symbolize_latent_with_atlas(
        action_model,
        action_latent_mean,
        hard_routing=hard_routing,
        hard_routing_tau=hard_routing_tau,
    )
    with torch.inference_mode():
        action, _, _ = action_model.decoder(
            action_info["z_geo"].detach(),
            None,
            router_weights=action_info["router_weights"].detach(),
            hard_routing=hard_routing,
            hard_routing_tau=hard_routing_tau,
        )
        action_mean, _, _ = action_model.decoder(
            action_info_mean["z_geo"].detach(),
            None,
            router_weights=action_info_mean["router_weights"].detach(),
            hard_routing=hard_routing,
            hard_routing_tau=hard_routing_tau,
        )
        closure_out = closure_model(
            z.detach(),
            rw.detach(),
            z_q.detach(),
            action_info["z_geo"].detach(),
            action_info["router_weights"].detach(),
            action_info["z_q"].detach(),
        )
    return {
        "action": action.detach(),
        "action_mean": action_mean.detach(),
        "action_latent": action_info["z_geo"].detach(),
        "action_latent_mean": action_info_mean["z_geo"].detach(),
        "action_router_weights": action_info["router_weights"].detach(),
        "action_chart_idx": action_info["chart_idx"].detach(),
        "action_code_idx": action_info["code_idx"].detach(),
        "action_code_latent": action_info["z_q"].detach(),
        "control_tan": closure_out["control_tan"].detach(),
        "control_cov": closure_out["control_cov"].detach(),
        "log_std": log_std.detach(),
    }


def _build_episode_dict(
    obs_list: list[np.ndarray],
    act_list: list[np.ndarray],
    rew_list: list[np.float32],
    done_list: list[np.float32],
    action_mean_list: list[np.ndarray],
    control_tan_list: list[np.ndarray],
    control_cov_list: list[np.ndarray],
    control_valid_list: list[np.float32],
    action_latent_list: list[np.ndarray],
    action_router_weight_list: list[np.ndarray],
    action_chart_idx_list: list[np.int64],
    action_code_idx_list: list[np.int64],
    action_code_latent_list: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Pack per-step episode traces into the replay-buffer episode format."""
    return {
        "obs": np.stack(obs_list),
        "actions": np.stack([*act_list, np.zeros_like(act_list[0])]),
        "action_means": np.stack([*action_mean_list, np.zeros_like(action_mean_list[0])]),
        "controls": np.stack([*control_cov_list, np.zeros_like(control_cov_list[0])]),
        "controls_tan": np.stack([*control_tan_list, np.zeros_like(control_tan_list[0])]),
        "controls_cov": np.stack([*control_cov_list, np.zeros_like(control_cov_list[0])]),
        "control_valid": np.array([*control_valid_list, 0.0], dtype=np.float32),
        "action_latents": np.stack(
            [*action_latent_list, np.zeros_like(action_latent_list[0])],
        ),
        "action_router_weights": np.stack(
            [*action_router_weight_list, np.zeros_like(action_router_weight_list[0])],
        ),
        "action_charts": np.array([*action_chart_idx_list, 0], dtype=np.int64),
        "action_codes": np.array([*action_code_idx_list, 0], dtype=np.int64),
        "action_code_latents": np.stack(
            [*action_code_latent_list, np.zeros_like(action_code_latent_list[0])],
        ),
        "rewards": np.array([*rew_list, 0.0], dtype=np.float32),
        "dones": np.array([*done_list, 1.0], dtype=np.float32),
    }


def _collect_episode(
    env,
    actor: GeometricActor | None,
    action_model: SharedDynTopoEncoder | None,
    closure_model: CovariantObsActionClosureModel | None,
    encoder: nn.Module,
    device: torch.device,
    control_dim: int,
    num_action_charts: int,
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
    action_latent_list, action_router_weight_list = [], []
    action_chart_idx_list, action_code_idx_list, action_code_latent_list = [], [], []
    time_step = env.reset()
    action_spec = env.action_spec()
    step = 0
    routing_tau = _rollout_routing_tau(hard_routing, hard_routing_tau)

    while not time_step.last() and step < max_steps:
        obs = _flatten_obs(time_step)
        obs_list.append(obs)

        if actor is None or action_model is None or closure_model is None:
            action = np.random.uniform(
                action_spec.minimum, action_spec.maximum,
                size=action_spec.shape,
            ).astype(np.float32)
            action_mean = action.copy()
            control_tan = np.zeros(control_dim, dtype=np.float32)
            control_cov = np.zeros(control_dim, dtype=np.float32)
            control_valid = 0.0
            action_latent = np.zeros(control_dim, dtype=np.float32)
            action_router_weights = np.zeros(num_action_charts, dtype=np.float32)
            action_chart_idx = np.int64(0)
            action_code_idx = np.int64(0)
            action_code_latent = np.zeros(control_dim, dtype=np.float32)
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
                z_q = enc_out[11]  # [1, D] code latent
            action_out = _policy_action(
                actor,
                action_model,
                closure_model,
                z,
                rw,
                z_q,
                use_motor_texture=use_motor_texture,
                hard_routing=hard_routing,
                hard_routing_tau=routing_tau,
            )
            action = action_out["action"].squeeze(0).cpu().numpy()
            action_mean = action_out["action_mean"].squeeze(0).cpu().numpy()
            control_tan = action_out["control_tan"].squeeze(0).cpu().numpy()
            control_cov = action_out["control_cov"].squeeze(0).cpu().numpy()
            control_valid = 1.0
            action_latent = action_out["action_latent"].squeeze(0).cpu().numpy()
            action_router_weights = action_out["action_router_weights"].squeeze(0).cpu().numpy()
            action_chart_idx = np.int64(action_out["action_chart_idx"].item())
            action_code_idx = np.int64(action_out["action_code_idx"].item())
            action_code_latent = action_out["action_code_latent"].squeeze(0).cpu().numpy()
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
        action_latent_list.append(action_latent)
        action_router_weight_list.append(action_router_weights)
        action_chart_idx_list.append(action_chart_idx)
        action_code_idx_list.append(action_code_idx)
        action_code_latent_list.append(action_code_latent)
        rew_list.append(np.float32(total_reward))
        done_list.append(np.float32(time_step.last()))
        step += 1

    # Append final observation
    obs_list.append(_flatten_obs(time_step))

    return _build_episode_dict(
        obs_list,
        act_list,
        rew_list,
        done_list,
        action_mean_list,
        control_tan_list,
        control_cov_list,
        control_valid_list,
        action_latent_list,
        action_router_weight_list,
        action_chart_idx_list,
        action_code_idx_list,
        action_code_latent_list,
    )


def _collect_parallel_episodes(
    env: VectorizedDMControlEnv,
    actor: GeometricActor | None,
    action_model: SharedDynTopoEncoder | None,
    closure_model: CovariantObsActionClosureModel | None,
    encoder: nn.Module,
    device: torch.device,
    control_dim: int,
    num_action_charts: int,
    *,
    num_episodes: int,
    obs_normalizer: ObservationNormalizer | None = None,
    action_repeat: int = 1,
    max_steps: int = 1000,
    hard_routing: bool = True,
    hard_routing_tau: float = 1.0,
    use_motor_texture: bool = True,
) -> list[dict[str, np.ndarray]]:
    """Collect multiple no-gas episodes in parallel with batched policy inference."""
    if num_episodes < 1:
        return []
    if num_episodes > env.n_workers:
        msg = f"num_episodes={num_episodes} exceeds available workers={env.n_workers}"
        raise ValueError(msg)

    action_spec = env.action_space
    action_min = np.asarray(action_spec.minimum, dtype=np.float32)
    action_max = np.asarray(action_spec.maximum, dtype=np.float32)
    action_shape = tuple(action_spec.shape)
    routing_tau = _rollout_routing_tau(hard_routing, hard_routing_tau)
    env_indices = np.arange(num_episodes, dtype=int)
    observations = env.reset_batch(env_indices=env_indices).astype(np.float32, copy=False)

    obs_lists = [[observations[i].copy()] for i in range(num_episodes)]
    act_lists: list[list[np.ndarray]] = [[] for _ in range(num_episodes)]
    rew_lists: list[list[np.float32]] = [[] for _ in range(num_episodes)]
    done_lists: list[list[np.float32]] = [[] for _ in range(num_episodes)]
    action_mean_lists: list[list[np.ndarray]] = [[] for _ in range(num_episodes)]
    control_tan_lists: list[list[np.ndarray]] = [[] for _ in range(num_episodes)]
    control_cov_lists: list[list[np.ndarray]] = [[] for _ in range(num_episodes)]
    control_valid_lists: list[list[np.float32]] = [[] for _ in range(num_episodes)]
    action_latent_lists: list[list[np.ndarray]] = [[] for _ in range(num_episodes)]
    action_router_weight_lists: list[list[np.ndarray]] = [[] for _ in range(num_episodes)]
    action_chart_lists: list[list[np.int64]] = [[] for _ in range(num_episodes)]
    action_code_lists: list[list[np.int64]] = [[] for _ in range(num_episodes)]
    action_code_latent_lists: list[list[np.ndarray]] = [[] for _ in range(num_episodes)]

    active = np.ones(num_episodes, dtype=bool)
    step_counts = np.zeros(num_episodes, dtype=np.int32)

    while active.any():
        active_indices = np.flatnonzero(active)
        active_obs = observations[active_indices]

        if actor is None or action_model is None or closure_model is None:
            actions = np.random.uniform(
                action_min,
                action_max,
                size=(len(active_indices), *action_shape),
            ).astype(np.float32)
            action_means = actions.copy()
            control_tan = np.zeros((len(active_indices), control_dim), dtype=np.float32)
            control_cov = np.zeros((len(active_indices), control_dim), dtype=np.float32)
            control_valid = np.zeros(len(active_indices), dtype=np.float32)
            action_latents = np.zeros((len(active_indices), control_dim), dtype=np.float32)
            action_router_weights = np.zeros(
                (len(active_indices), num_action_charts),
                dtype=np.float32,
            )
            action_chart_idx = np.zeros(len(active_indices), dtype=np.int64)
            action_code_idx = np.zeros(len(active_indices), dtype=np.int64)
            action_code_latents = np.zeros((len(active_indices), control_dim), dtype=np.float32)
        else:
            obs_t = torch.from_numpy(active_obs).to(device)
            if obs_normalizer is not None:
                obs_t = obs_normalizer.normalize_tensor(obs_t)
            with torch.no_grad():
                enc_out = encoder.encoder(
                    obs_t,
                    hard_routing=hard_routing,
                    hard_routing_tau=routing_tau,
                )
                rw = enc_out[4]
                z = enc_out[5]
                z_q = enc_out[11]
            action_out = _policy_action(
                actor,
                action_model,
                closure_model,
                z,
                rw,
                z_q,
                use_motor_texture=use_motor_texture,
                hard_routing=hard_routing,
                hard_routing_tau=routing_tau,
            )
            actions = action_out["action"].cpu().numpy()
            action_means = action_out["action_mean"].cpu().numpy()
            control_tan = action_out["control_tan"].cpu().numpy()
            control_cov = action_out["control_cov"].cpu().numpy()
            control_valid = np.ones(len(active_indices), dtype=np.float32)
            action_latents = action_out["action_latent"].cpu().numpy()
            action_router_weights = action_out["action_router_weights"].cpu().numpy()
            action_chart_idx = action_out["action_chart_idx"].cpu().numpy().astype(np.int64, copy=False)
            action_code_idx = action_out["action_code_idx"].cpu().numpy().astype(np.int64, copy=False)
            action_code_latents = action_out["action_code_latent"].cpu().numpy()
            actions = np.clip(actions, action_min, action_max)
            action_means = np.clip(action_means, action_min, action_max)

        next_obs, rewards, dones, _trunc, _infos = env.step_actions_batch(
            actions.astype(np.float64, copy=False),
            dt=np.full(len(active_indices), action_repeat, dtype=int),
            env_indices=active_indices,
        )
        next_obs = next_obs.astype(np.float32, copy=False)

        for row, episode_idx in enumerate(active_indices):
            act_lists[episode_idx].append(actions[row].astype(np.float32, copy=False))
            action_mean_lists[episode_idx].append(
                action_means[row].astype(np.float32, copy=False),
            )
            control_tan_lists[episode_idx].append(control_tan[row].astype(np.float32, copy=False))
            control_cov_lists[episode_idx].append(control_cov[row].astype(np.float32, copy=False))
            control_valid_lists[episode_idx].append(np.float32(control_valid[row]))
            action_latent_lists[episode_idx].append(
                action_latents[row].astype(np.float32, copy=False),
            )
            action_router_weight_lists[episode_idx].append(
                action_router_weights[row].astype(np.float32, copy=False),
            )
            action_chart_lists[episode_idx].append(np.int64(action_chart_idx[row]))
            action_code_lists[episode_idx].append(np.int64(action_code_idx[row]))
            action_code_latent_lists[episode_idx].append(
                action_code_latents[row].astype(np.float32, copy=False),
            )
            rew_lists[episode_idx].append(np.float32(rewards[row]))
            done_lists[episode_idx].append(np.float32(dones[row]))
            step_counts[episode_idx] += 1
            observations[episode_idx] = next_obs[row]
            obs_lists[episode_idx].append(next_obs[row].copy())
            if dones[row] or step_counts[episode_idx] >= max_steps:
                active[episode_idx] = False

    return [
        _build_episode_dict(
            obs_lists[i],
            act_lists[i],
            rew_lists[i],
            done_lists[i],
            action_mean_lists[i],
            control_tan_lists[i],
            control_cov_lists[i],
            control_valid_lists[i],
            action_latent_lists[i],
            action_router_weight_lists[i],
            action_chart_lists[i],
            action_code_lists[i],
            action_code_latent_lists[i],
        )
        for i in range(num_episodes)
    ]


def _eval_policy(
    env,
    actor: GeometricActor,
    action_model: SharedDynTopoEncoder,
    closure_model: CovariantObsActionClosureModel,
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
                rw, z, z_q = enc_out[4], enc_out[5], enc_out[11]
            action_out = _policy_action(
                actor,
                action_model,
                closure_model,
                z,
                rw,
                z_q,
                use_motor_texture=False,
                hard_routing=hard_routing,
                hard_routing_tau=routing_tau,
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
    actor: GeometricActor | None,
    action_model: SharedDynTopoEncoder | None,
    closure_model: CovariantObsActionClosureModel | None,
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
    all_action_latents = np.zeros((N, steps + 1, config.latent_dim), dtype=np.float32)
    all_action_router_weights = np.zeros(
        (N, steps + 1, config.num_action_charts),
        dtype=np.float32,
    )
    all_action_charts = np.zeros((N, steps + 1), dtype=np.int64)
    all_action_codes = np.zeros((N, steps + 1), dtype=np.int64)
    all_action_code_latents = np.zeros((N, steps + 1, config.latent_dim), dtype=np.float32)
    all_rewards = np.zeros((N, steps + 1), dtype=np.float32)
    all_dones = np.zeros((N, steps + 1), dtype=np.float32)

    all_obs[:, 0] = state.observations.cpu().numpy()
    all_dones[:, 0] = state.dones.cpu().float().numpy()
    routing_tau = _rollout_routing_tau(config.hard_routing, config.hard_routing_tau)

    for t in range(steps):
        # Compute actions from policy (or None for random)
        actions_np = None
        if actor is not None and action_model is not None and closure_model is not None:
            obs_t = state.observations.to(device)
            if obs_normalizer is not None:
                obs_t = obs_normalizer.normalize_tensor(obs_t)
            chunk_size = 1024
            action_chunks = []
            action_mean_chunks = []
            control_tan_chunks = []
            control_cov_chunks = []
            action_latent_chunks = []
            action_router_chunks = []
            action_chart_chunks = []
            action_code_chunks = []
            action_code_latent_chunks = []
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
                    z_q = enc_out[11]
                action_out = _policy_action(
                    actor,
                    action_model,
                    closure_model,
                    z,
                    rw,
                    z_q,
                    use_motor_texture=config.use_motor_texture,
                    hard_routing=config.hard_routing,
                    hard_routing_tau=routing_tau,
                )
                action_chunks.append(action_out["action"].cpu().numpy())
                action_mean_chunks.append(action_out["action_mean"].cpu().numpy())
                control_tan_chunks.append(action_out["control_tan"].cpu().numpy())
                control_cov_chunks.append(action_out["control_cov"].cpu().numpy())
                action_latent_chunks.append(action_out["action_latent"].cpu().numpy())
                action_router_chunks.append(action_out["action_router_weights"].cpu().numpy())
                action_chart_chunks.append(action_out["action_chart_idx"].cpu().numpy())
                action_code_chunks.append(action_out["action_code_idx"].cpu().numpy())
                action_code_latent_chunks.append(action_out["action_code_latent"].cpu().numpy())
            actions_np = np.concatenate(action_chunks, axis=0)
            all_action_means[:, t] = np.concatenate(action_mean_chunks, axis=0)
            all_controls_tan[:, t] = np.concatenate(control_tan_chunks, axis=0)
            all_controls_cov[:, t] = np.concatenate(control_cov_chunks, axis=0)
            all_action_latents[:, t] = np.concatenate(action_latent_chunks, axis=0)
            all_action_router_weights[:, t] = np.concatenate(action_router_chunks, axis=0)
            all_action_charts[:, t] = np.concatenate(action_chart_chunks, axis=0)
            all_action_codes[:, t] = np.concatenate(action_code_chunks, axis=0)
            all_action_code_latents[:, t] = np.concatenate(action_code_latent_chunks, axis=0)
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
        _clone_prefix_inplace(all_action_latents, companions, will_clone, t)
        _clone_prefix_inplace(all_action_router_weights, companions, will_clone, t)
        _clone_prefix_inplace(all_action_charts, companions, will_clone, t)
        _clone_prefix_inplace(all_action_codes, companions, will_clone, t)
        _clone_prefix_inplace(all_action_code_latents, companions, will_clone, t)
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
            "action_latents": all_action_latents[i],
            "action_router_weights": all_action_router_weights[i],
            "action_charts": all_action_charts[i],
            "action_codes": all_action_codes[i],
            "action_code_latents": all_action_code_latents[i],
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
    obs_model: TopoEncoderPrimitives,
    world_model: GeometricWorldModel,
    reward_head: RewardHead,
    critic: nn.Module,
    actor: GeometricActor,
    action_model: SharedDynTopoEncoder,
    closure_model: CovariantObsActionClosureModel,
    z_0: torch.Tensor,
    rw_0: torch.Tensor,
    horizon: int,
    gamma: float,
    reward_curl_batch_limit: int | None = None,
) -> dict[str, torch.Tensor]:
    """Roll out the critic-induced control field in latent space.

    Returns:
        z_states: [B, H, D] policy states before each action
        rw_states: [B, H, K] router weights before each action
        z_traj: [B, H, D]
        rw_traj: [B, H, K]
        controls_tan: [B, H, D]
        controls_cov: [B, H, D]
        action_latents: [B, H, D]
        action_router_weights: [B, H, K_a]
        actions: [B, H, A]
        action_log_std: [B, H, A]
        rewards: [B, H]
        reward_conservative: [B, H]
        reward_nonconservative: [B, H]
        reward_curl_norm: [B, H]
        reward_curl_valid: [B, H]
        phi_eff: [B, H, 1]
    """
    z, rw = z_0, rw_0
    p = world_model.momentum_init(z_0)  # [B, D]

    z_state_list, rw_state_list = [], []
    z_list, rw_list, control_tan_list, control_cov_list, action_list = [], [], [], [], []
    action_latent_list, action_router_list = [], []
    r_list, r_cons_list, r_noncons_list, r_curl_list, r_curl_valid_list = [], [], [], [], []
    texture_list, phi_list = [], []

    for _t in range(horizon):
        obs_info = symbolize_latent_with_atlas(
            obs_model,
            z,
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        action_out = _policy_action(
            actor,
            action_model,
            closure_model,
            z,
            rw,
            obs_info["z_q"],
            use_motor_texture=False,
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        z_state_list.append(z.detach())
        rw_state_list.append(rw.detach())
        control_tan_list.append(action_out["control_tan"])
        control_cov_list.append(action_out["control_cov"])
        action_latent_list.append(action_out["action_latent"])
        action_router_list.append(action_out["action_router_weights"])
        action_list.append(action_out["action_mean"])
        z_curr = z
        rw_curr = rw

        with torch.no_grad():
            step_out = world_model._rollout_transition(
                z,
                p,
                action_out["control_cov"],
                rw,
                track_energy=False,
            )
            z_next = step_out["z"]
            p_next = step_out["p"]
            rw_next = step_out["rw"]
            phi_eff = step_out["phi_eff"]
            reward_conservative, _v_curr, _v_next = _conservative_reward_from_value(
                critic,
                z_curr,
                rw_curr,
                z_next,
                rw_next,
                gamma,
            )
            z = z_next
            p = p_next
            rw = rw_next
        exact_covector = _value_covector_from_critic(critic, z_curr, rw_curr)
        rw_curr_detached = rw_curr.detach()

        def exact_covector_fn(z_req: torch.Tensor) -> torch.Tensor:
            return _value_covector_from_critic(
                critic,
                z_req,
                rw_curr_detached[: z_req.shape[0]],
                create_graph=True,
                detach=False,
            )

        with torch.no_grad():
            reward_info = reward_head.decompose(
                z_curr,
                rw_curr,
                action_out["action_latent"],
                action_out["action_router_weights"],
                action_out["action_code_latent"],
                control=action_out["control_tan"],
                exact_covector=exact_covector,
                compute_curl=False,
            )
            r_noncons = reward_info["reward_nonconservative"]
            r_hat = reward_conservative + r_noncons  # [B, 1]

        reward_curl = reward_head.reward_curl(
            z_curr,
            rw_curr,
            action_out["action_latent"],
            action_out["action_router_weights"],
            action_out["action_code_latent"],
            exact_covector=exact_covector,
            exact_covector_fn=exact_covector_fn,
            max_batch=reward_curl_batch_limit,
        )
        reward_curl_norm = z_curr.new_zeros(z_curr.shape[0])
        reward_curl_valid = torch.zeros(z_curr.shape[0], dtype=torch.bool, device=z_curr.device)
        if reward_curl.numel() > 0:
            eval_batch = reward_curl.shape[0]
            reward_curl_norm[:eval_batch] = torch.linalg.norm(reward_curl, dim=(-2, -1))
            reward_curl_valid[:eval_batch] = True

        z_list.append(z.detach())
        rw_list.append(rw.detach())
        r_list.append(r_hat.squeeze(-1))
        r_cons_list.append(reward_conservative.squeeze(-1))
        r_noncons_list.append(r_noncons.squeeze(-1))
        r_curl_list.append(reward_curl_norm)
        r_curl_valid_list.append(reward_curl_valid)
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
        "action_latents": torch.stack(action_latent_list, dim=1),  # [B, H, D]
        "action_router_weights": torch.stack(action_router_list, dim=1),  # [B, H, K_a]
        "actions": torch.stack(action_list, dim=1),  # [B, H, A]
        "rewards": torch.stack(r_list, dim=1),   # [B, H]
        "reward_conservative": torch.stack(r_cons_list, dim=1),  # [B, H]
        "reward_nonconservative": torch.stack(r_noncons_list, dim=1),  # [B, H]
        "reward_curl_norm": torch.stack(r_curl_list, dim=1),  # [B, H]
        "reward_curl_valid": torch.stack(r_curl_valid_list, dim=1),  # [B, H]
        "action_log_std": torch.stack(texture_list, dim=1),  # [B, H, A]
        "phi_eff": torch.stack(phi_list, dim=1),    # [B, H, 1]
    }


# ---------------------------------------------------------------------------
# VLAConfig bridge — create Phase 1 config from DreamerConfig
# ---------------------------------------------------------------------------


def _phase1_config(
    config: DreamerConfig,
    phase1_state=None,
    *,
    input_dim: int | None = None,
    num_charts: int | None = None,
    codes_per_chart: int | None = None,
) -> VLAConfig:
    """Build a VLAConfig with Phase 1 weights matching DreamerConfig."""
    scales = phase1_effective_weight_scales(config, phase1_state)
    return VLAConfig(
        input_dim=config.obs_dim if input_dim is None else input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_charts=config.num_charts if num_charts is None else num_charts,
        codes_per_chart=config.codes_per_chart if codes_per_chart is None else codes_per_chart,
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
def _bind_chart_tokenizer_centers(chart_tok: nn.Module, centers: torch.Tensor) -> None:
    """Copy projected chart centers into a chart tokenizer and freeze them."""
    safe_centers = _project_to_ball(centers.detach())
    chart_tok.chart_centers.copy_(
        safe_centers.to(
            device=chart_tok.chart_centers.device,
            dtype=chart_tok.chart_centers.dtype,
        ),
    )
    chart_tok.chart_centers.requires_grad_(False)


@torch.no_grad()
def _sync_rl_atlas(
    model: TopoEncoderPrimitives,
    action_model: TopoEncoderPrimitives,
    world_model: GeometricWorldModel,
    critic: nn.Module,
    actor: GeometricActor,
    closure_model: CovariantObsActionClosureModel,
    reward_head: RewardHead,
) -> None:
    """Keep RL consumers bound to the observation or action atlas they read from."""
    obs_centers = getattr(model.encoder, "chart_centers", None)
    action_centers = getattr(action_model.encoder, "chart_centers", None)
    if obs_centers is None or action_centers is None:
        return

    obs_centers = _project_to_ball(obs_centers.detach())
    action_centers = _project_to_ball(action_centers.detach())
    world_model_mod = _unwrap_compiled_module(world_model)
    critic_mod = _unwrap_compiled_module(critic)
    reward_head_mod = _unwrap_compiled_module(reward_head)

    world_model_mod.bind_chart_centers(obs_centers, freeze=True)
    _bind_chart_tokenizer_centers(actor.chart_tok, obs_centers)
    _bind_chart_tokenizer_centers(closure_model.obs_chart_tok, obs_centers)
    if hasattr(critic_mod, "chart_tok"):
        _bind_chart_tokenizer_centers(critic_mod.chart_tok, obs_centers)
    _bind_chart_tokenizer_centers(closure_model.action_chart_tok, action_centers)
    _bind_chart_tokenizer_centers(reward_head_mod.action_chart_tok, action_centers)


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
    *,
    enc_prefix: str,
    chart_prefix: str,
    num_charts: int,
) -> dict[str, float]:
    """Map per-manifold Dreamer diagnostics to Phase 1 controller inputs."""

    def _active_code_entropies(prefix: str, num_charts: int) -> list[float]:
        entropies: list[float] = []
        for chart_idx in range(num_charts):
            active_codes = metrics.get(f"{prefix}/{chart_idx}/active_codes", 0.0)
            if active_codes > 0.0:
                entropies.append(metrics.get(f"{prefix}/{chart_idx}/code_entropy", 0.0))
        return entropies

    return {
        "soft_top1_prob_mean": float(metrics.get(f"{enc_prefix}/top1_prob_mean", 0.0)),
        "soft_I_XK": float(metrics.get(f"{enc_prefix}/I_XK", 0.0)),
        "hard_entropy": float(metrics.get(f"{chart_prefix}/usage_entropy", 0.0)),
        "code_entropy_mean_active": (
            float(np.mean(_active_code_entropies(chart_prefix, num_charts)))
            if num_charts > 0
            else 0.0
        ),
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


def _should_run_actor_update(
    config: DreamerConfig,
    *,
    epoch: int,
    update_idx: int,
) -> bool:
    """Return whether the imagined-return actor step should run."""
    if config.w_actor_return <= 0.0:
        return False
    if config.actor_return_update_every <= 0:
        return False
    if config.actor_return_horizon <= 0:
        return False
    if epoch < config.actor_return_warmup_epochs:
        return False
    return update_idx % config.actor_return_update_every == 0


def _imagine_actor_return(
    obs_model: TopoEncoderPrimitives,
    world_model: GeometricWorldModel,
    reward_head: RewardHead,
    critic: nn.Module,
    actor: GeometricActor,
    action_model: SharedDynTopoEncoder,
    closure_model: CovariantObsActionClosureModel,
    z_0: torch.Tensor,
    rw_0: torch.Tensor,
    horizon: int,
    gamma: float,
) -> dict[str, torch.Tensor]:
    """Differentiate the actor objective through latent-action imagination.

    The exact reward sector is represented by the critic value field, so the
    actor should optimize only the non-conservative work term. The exact sector
    is a boundary contribution fixed by the start state and should not steer the
    policy.
    """
    z = z_0
    rw = rw_0
    p = world_model.momentum_init(z_0)

    reward_noncons_list: list[torch.Tensor] = []
    action_list: list[torch.Tensor] = []
    control_tan_list: list[torch.Tensor] = []

    for _ in range(horizon):
        obs_info = symbolize_latent_with_atlas(
            obs_model,
            z,
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        action_latent, _action_latent_mean, _log_std = actor.sample_latent(z, rw)
        action_info = symbolize_latent_with_atlas(
            action_model,
            action_latent,
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        action_mean, _, _ = action_model.decoder(
            action_info["z_geo"],
            None,
            router_weights=action_info["router_weights"],
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        closure_out = closure_model(
            z,
            rw,
            obs_info["z_q"],
            action_info["z_geo"],
            action_info["router_weights"],
            action_info["z_q"],
        )
        control_cov = closure_out["control_cov"]
        control_tan = closure_out["control_tan"]
        step_out = world_model._rollout_transition(
            z,
            p,
            control_cov,
            rw,
            track_energy=False,
        )
        z_next = step_out["z"]
        rw_next = step_out["rw"]
        exact_covector = _value_covector_from_critic(critic, z, rw)
        reward_info = reward_head.decompose(
            z,
            rw,
            action_info["z_geo"],
            action_info["router_weights"],
            action_info["z_q"],
            control=control_tan,
            exact_covector=exact_covector,
        )
        reward_nonconservative = reward_info["reward_nonconservative"].squeeze(-1)
        reward_noncons_list.append(reward_nonconservative)
        action_list.append(action_mean)
        control_tan_list.append(control_tan)
        z = z_next
        p = step_out["p"]
        rw = rw_next

    reward_nonconservative = torch.stack(reward_noncons_list, dim=1)
    objective = _discounted_sum(reward_nonconservative, gamma)
    return {
        "reward_nonconservative": reward_nonconservative,
        "objective": objective,
        "actions": torch.stack(action_list, dim=1),
        "controls_tan": torch.stack(control_tan_list, dim=1),
    }


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
    action_model: TopoEncoderPrimitives,
    action_jump_op: FactorizedJumpOperator,
    closure_model: CovariantObsActionClosureModel,
    world_model: GeometricWorldModel,
    reward_head: RewardHead,
    critic: nn.Module,
    actor: GeometricActor,
    optimizer_enc: torch.optim.Optimizer,
    optimizer_wm: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer | None,
    optimizer_boundary: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    config: DreamerConfig,
    phase1_cfg: VLAConfig,
    action_phase1_cfg: VLAConfig,
    epoch: int,
    current_hard_routing: bool,
    current_tau: float,
    update_idx: int = 0,
    obs_normalizer: ObservationNormalizer | None = None,
    compute_diagnostics: bool = True,
) -> dict[str, float]:
    """One training iteration for the two-manifold Dreamer."""
    metrics: dict[str, float] = {}
    step_t0 = time.perf_counter()
    _mark_compile_step_begin()

    obs_raw = batch["obs"]
    obs = obs_normalizer.normalize_tensor(obs_raw) if obs_normalizer is not None else obs_raw
    action_seq = batch["actions"]
    actions = action_seq[:, :-1]
    rewards = batch["rewards"][:, :-1]
    replay_dones = batch["dones"][:, :-1]
    B, T, _A = actions.shape
    H_wm = min(T, max(1, int(config.wm_prediction_horizon), int(config.imagination_horizon)))
    shared_critic = _shared_value_field(world_model, critic)
    replay_valid = _transition_valid_mask(replay_dones)

    t_section = time.perf_counter()
    flat_obs = obs.reshape(B * (T + 1), -1)
    flat_actions = action_seq.reshape(B * (T + 1), -1)

    def _encode_sequence(
        x: torch.Tensor,
        topo_model: TopoEncoderPrimitives,
        topo_jump_op: FactorizedJumpOperator,
        phase_cfg: VLAConfig,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, float],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            base_loss,
            zn_reg_loss,
            enc_metrics,
            z_geo_flat,
            enc_w_flat,
            K_ch_flat,
            _z_n_flat,
            _z_tex_flat,
            c_bar_flat,
            K_code_flat,
            z_q_flat,
            v_local_flat,
        ) = _compute_encoder_losses(
            x,
            topo_model,
            topo_jump_op,
            config,
            epoch,
            hard_routing=current_hard_routing,
            hard_routing_tau=current_tau,
            phase1_config=phase_cfg,
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
            z_q_flat.reshape(B, T + 1, -1),
            v_local_flat.reshape(B, T + 1, -1),
        )

    optimizer_enc.zero_grad()

    encode_context = torch.no_grad() if config.freeze_encoder else contextlib.nullcontext()
    with encode_context:
        (
            base_loss_obs,
            zn_reg_loss_obs,
            enc_metrics_obs,
            z_all,
            rw_all,
            K_all,
            K_code_all,
            _c_bar_all,
            z_q_all,
            _v_local_all,
        ) = _encode_sequence(flat_obs, model, jump_op, phase1_cfg)
        (
            base_loss_action,
            zn_reg_loss_action,
            enc_metrics_action,
            action_z_all,
            action_rw_all,
            action_K_all,
            action_K_code_all,
            _action_c_bar_all,
            action_z_q_all,
            _action_v_local_all,
        ) = _encode_sequence(flat_actions, action_model, action_jump_op, action_phase1_cfg)

    z_prev = z_all[:, :-1]
    rw_prev = rw_all[:, :-1]
    z_q_prev = z_q_all[:, :-1]
    action_z_prev = action_z_all[:, :-1]
    action_rw_prev = action_rw_all[:, :-1]
    action_z_q_prev = action_z_q_all[:, :-1]

    closure_out = closure_model(
        z_prev.reshape(-1, config.latent_dim),
        rw_prev.reshape(-1, config.num_charts),
        z_q_prev.reshape(-1, config.latent_dim),
        action_z_prev.reshape(-1, config.latent_dim),
        action_rw_prev.reshape(-1, config.num_action_charts),
        action_z_q_prev.reshape(-1, config.latent_dim),
    )

    replay_valid_flat = replay_valid.reshape(-1)
    action_next_valid = (replay_valid * (1.0 - replay_dones)).reshape(-1)
    obs_targets_flat = (
        K_all[:, 1:].long() * config.codes_per_chart + K_code_all[:, 1:].long()
    ).reshape(-1)
    action_targets_flat = (
        action_K_all[:, 1:].long() * config.action_codes_per_chart
        + action_K_code_all[:, 1:].long()
    ).reshape(-1)

    obs_log_probs = F.log_softmax(closure_out["obs_state_logits"], dim=-1)
    action_log_probs = F.log_softmax(closure_out["action_state_logits"], dim=-1)
    obs_ce = -obs_log_probs.gather(1, obs_targets_flat.unsqueeze(1)).squeeze(1)
    action_ce = -action_log_probs.gather(1, action_targets_flat.unsqueeze(1)).squeeze(1)
    L_closure_obs = _masked_mean(obs_ce, replay_valid_flat)
    L_closure_action = _masked_mean(action_ce, action_next_valid)
    L_obs_zeno = zeno_loss(
        rw_all[:, 1:].reshape(-1, config.num_charts),
        rw_all[:, :-1].reshape(-1, config.num_charts),
        mode=config.zeno_mode,
    )
    L_action_zeno = zeno_loss(
        action_rw_all[:, 1:].reshape(-1, config.num_action_charts),
        action_rw_all[:, :-1].reshape(-1, config.num_action_charts),
        mode=config.zeno_mode,
    )

    replay_controls_tan = batch.get("controls_tan")
    replay_controls_cov = batch.get("controls_cov")
    replay_control_valid = batch.get("control_valid")
    if replay_controls_tan is None:
        replay_controls_tan = torch.zeros(B, T + 1, config.latent_dim, device=actions.device)
    if replay_controls_cov is None:
        replay_controls_cov = torch.zeros(B, T + 1, config.latent_dim, device=actions.device)
    if replay_control_valid is None:
        replay_control_valid = torch.zeros(B, T + 1, device=actions.device)
    replay_controls_tan = replay_controls_tan[:, :-1]
    replay_controls_cov = replay_controls_cov[:, :-1]
    replay_control_valid = replay_control_valid[:, :-1]
    replay_control_valid_flat = replay_control_valid.reshape(-1)
    control_supervise_err = F.smooth_l1_loss(
        closure_out["control_tan"],
        replay_controls_tan.reshape(-1, config.latent_dim),
        reduction="none",
    ).mean(dim=-1)
    L_control_supervise = _masked_mean(control_supervise_err, replay_control_valid_flat)
    L_closure = (
        config.w_dyn_transition * (L_closure_obs + L_closure_action)
        + config.w_zeno * (L_obs_zeno + L_action_zeno)
        + config.w_control_supervise * L_control_supervise
    )

    L_obs_total = base_loss_obs + zn_reg_loss_obs
    L_action_total = base_loss_action + zn_reg_loss_action
    if config.freeze_encoder:
        L_enc_total = config.encoder_loss_scale * L_closure
    else:
        L_enc_total = config.encoder_loss_scale * (L_obs_total + L_action_total + L_closure)
    L_enc_total.backward()
    enc_grad = nn.utils.clip_grad_norm_(_optimizer_parameters(optimizer_enc), config.grad_clip)
    optimizer_enc.step()
    metrics["enc/grad_norm"] = float(enc_grad)
    metrics["closure/grad_norm"] = float(
        _parameter_grad_norm(list(closure_model.parameters())),
    )
    if not config.freeze_encoder:
        for key, value in _phase1_grad_breakdown(model).items():
            metrics[f"enc/{key}"] = value
        for key, value in _phase1_grad_breakdown(action_model).items():
            metrics[f"enc_action/{key}"] = value

    for key, value in enc_metrics_obs.items():
        metrics[f"enc/{key}"] = value
    for key, value in enc_metrics_action.items():
        metrics[f"enc_action/{key}"] = value
    metrics["enc/L_total"] = float(config.encoder_loss_scale * L_obs_total)
    metrics["enc_action/L_total"] = float(config.encoder_loss_scale * L_action_total)
    metrics["closure/L_total"] = float(L_closure)
    metrics["closure/L_obs_state"] = float(L_closure_obs)
    metrics["closure/L_action_state"] = float(L_closure_action)
    metrics["closure/L_control_supervise"] = float(L_control_supervise)
    metrics["closure/L_obs_zeno"] = float(L_obs_zeno)
    metrics["closure/L_action_zeno"] = float(L_action_zeno)
    metrics["closure/obs_state_acc"] = float(
        ((closure_out["obs_state_logits"].argmax(dim=-1) == obs_targets_flat).float() * replay_valid_flat)
        .sum()
        / replay_valid_flat.sum().clamp(min=1.0)
    )
    metrics["closure/action_state_acc"] = float(
        (
            (closure_out["action_state_logits"].argmax(dim=-1) == action_targets_flat).float()
            * action_next_valid
        ).sum()
        / action_next_valid.sum().clamp(min=1.0)
    )
    metrics["time/encoder"] = time.perf_counter() - t_section

    t_section = time.perf_counter()
    _sync_rl_atlas(model, action_model, world_model, critic, actor, closure_model, reward_head)
    metrics["time/atlas_sync"] = time.perf_counter() - t_section

    if compute_diagnostics:
        t_section = time.perf_counter()
        metrics.update(
            _per_chart_diagnostics(
                K_all.reshape(-1),
                K_code_all.reshape(-1),
                rw_all.reshape(-1, config.num_charts),
                z_all.reshape(-1, config.latent_dim),
                config.num_charts,
                config.codes_per_chart,
            ),
        )
        action_chart_diag = _per_chart_diagnostics(
            action_K_all.reshape(-1),
            action_K_code_all.reshape(-1),
            action_rw_all.reshape(-1, config.num_action_charts),
            action_z_all.reshape(-1, config.latent_dim),
            config.num_action_charts,
            config.action_codes_per_chart,
        )
        metrics.update({f"action_chart/{k}": v for k, v in action_chart_diag.items()})
        metrics["time/chart_diag"] = time.perf_counter() - t_section

    z_0 = z_all[:, 0].detach()
    rw_0 = rw_all[:, 0].detach()
    z_targets = z_all[:, 1:].detach()
    z_prev = z_all[:, :-1].detach()
    rw_prev = rw_all[:, :-1].detach()
    action_z_prev = action_z_all[:, :-1].detach()
    action_rw_prev = action_rw_all[:, :-1].detach()
    action_z_q_prev = action_z_q_all[:, :-1].detach()
    predicted_controls_tan = closure_out["control_tan"].reshape(B, T, config.latent_dim).detach()
    predicted_controls_cov = closure_out["control_cov"].reshape(B, T, config.latent_dim).detach()
    control_mask = replay_control_valid.unsqueeze(-1).bool()
    controls_model_tan = torch.where(control_mask, replay_controls_tan, predicted_controls_tan)
    controls_model_cov = torch.where(control_mask, replay_controls_cov, predicted_controls_cov)

    t_section = time.perf_counter()
    optimizer_wm.zero_grad()
    wm_out = world_model(z_0, controls_model_cov[:, :H_wm], rw_0)
    z_pred = wm_out["z_trajectory"]
    T_wm = min(z_pred.shape[1], H_wm)
    z_tgt_wm = z_targets[:, :T_wm]
    L_geo = compute_dynamics_geodesic_loss(z_pred, z_tgt_wm)
    target_charts = K_all.detach()[:, 1 : T_wm + 1]
    L_chart = compute_dynamics_chart_loss(wm_out["chart_logits"][:, :T_wm], target_charts)
    chart_probs = F.softmax(wm_out["chart_logits"][:, :T_wm], dim=-1)
    metrics["wm/L_geodesic"] = float(L_geo)
    metrics["wm/L_chart"] = float(L_chart)
    metrics["wm/chart_acc"] = float(
        (wm_out["chart_logits"][:, :T_wm].argmax(dim=-1) == target_charts).float().mean()
    )
    metrics["wm/chart_entropy"] = float(
        -(chart_probs * chart_probs.clamp(min=1e-8).log()).sum(dim=-1).mean()
    )

    z_prev_flat = z_prev.reshape(-1, config.latent_dim)
    rw_prev_flat = rw_prev.reshape(-1, config.num_charts)
    z_next_flat = z_targets.reshape(-1, config.latent_dim)
    rw_next_flat = rw_all[:, 1:].detach().reshape(-1, config.num_charts)
    action_z_prev_flat = action_z_prev.reshape(-1, config.latent_dim)
    action_rw_prev_flat = action_rw_prev.reshape(-1, config.num_action_charts)
    action_z_q_prev_flat = action_z_q_prev.reshape(-1, config.latent_dim)
    controls_model_tan_flat = controls_model_tan.reshape(-1, config.latent_dim)
    continuation_flat = (1.0 - replay_dones).reshape(-1, 1)
    reward_conservative_flat, value_prev_flat, _value_next_flat = _conservative_reward_from_value(
        critic,
        z_prev_flat,
        rw_prev_flat,
        z_next_flat,
        rw_next_flat,
        config.gamma,
        continuation=continuation_flat,
    )
    exact_covector = _value_covector_from_critic(critic, z_prev_flat, rw_prev_flat)
    exact_covector_fn = None
    if compute_diagnostics:
        rw_prev_flat_detached = rw_prev_flat.detach()

        def exact_covector_fn(z_req: torch.Tensor) -> torch.Tensor:
            return _value_covector_from_critic(
                critic,
                z_req,
                rw_prev_flat_detached[: z_req.shape[0]],
                create_graph=True,
                detach=False,
            )

    reward_info = reward_head.decompose(
        z_prev_flat,
        rw_prev_flat,
        action_z_prev_flat,
        action_rw_prev_flat,
        action_z_q_prev_flat,
        control=controls_model_tan_flat,
        exact_covector=exact_covector,
        exact_covector_fn=exact_covector_fn,
        compute_curl=compute_diagnostics,
        curl_batch_limit=config.reward_curl_batch_limit,
    )
    r_noncons_flat = reward_info["reward_nonconservative"]
    r_pred = (reward_conservative_flat + r_noncons_flat).reshape(B, T)
    r_cons = reward_conservative_flat.reshape(B, T, 1)
    r_noncons = r_noncons_flat.reshape(B, T)
    reward_residual_target = rewards - r_cons.detach().squeeze(-1)
    rho_r = reward_info["reward_density"].reshape(B, T, 1)
    reward_form_cov = reward_info["reward_form_cov"]
    reward_form_cov_raw = reward_info["reward_form_cov_raw"]
    reward_form_exact_component = reward_info["reward_form_exact_component"]
    L_reward = _masked_mean((r_pred - rewards).pow(2), replay_valid)
    L_reward_nonconservative = _masked_mean(
        (r_noncons - reward_residual_target).pow(2),
        replay_valid,
    )
    reward_form_dot_exact = (reward_form_cov_raw * exact_covector).sum(dim=-1)
    reward_form_norm_sq = reward_form_cov_raw.pow(2).sum(dim=-1)
    exact_covector_norm_sq = exact_covector.pow(2).sum(dim=-1)
    # The latent Poincare metric is conformal, so the scalar G^{ij}(z) factor cancels in
    # this pointwise cosine between covectors and the Euclidean formula is equivalent.
    reward_exact_cos2 = reward_form_dot_exact.pow(2) / (
        reward_form_norm_sq * exact_covector_norm_sq + 1e-8
    )
    L_reward_exact_orth = _masked_mean(reward_exact_cos2, replay_valid.reshape(-1))
    metrics["wm/L_reward"] = float(L_reward)
    metrics["wm/L_reward_nonconservative"] = float(L_reward_nonconservative)
    metrics["wm/L_reward_exact_orth"] = float(L_reward_exact_orth)
    metrics["wm/reward_control_norm_mean"] = float(controls_model_tan_flat.norm(dim=-1).mean())
    metrics["wm/reward_conservative_mean"] = float(r_cons.mean())
    metrics["wm/reward_nonconservative_mean"] = float(r_noncons.mean())
    metrics["wm/reward_residual_target_mean"] = float(reward_residual_target.mean())
    metrics["wm/reward_exact_cos2_mean"] = float(_masked_mean(reward_exact_cos2, replay_valid.reshape(-1)))
    metrics["wm/reward_nonconservative_frac"] = float(
        r_noncons.abs().mean() / (r_pred.abs().mean() + 1e-8)
    )
    metrics["wm/reward_density_mean"] = float(rho_r.mean())
    metrics["wm/reward_form_norm_mean"] = float(reward_form_cov.norm(dim=-1).mean())
    metrics["wm/reward_form_raw_norm_mean"] = float(reward_form_cov_raw.norm(dim=-1).mean())
    metrics["wm/reward_form_exact_leakage_mean"] = float(
        reward_form_exact_component.norm(dim=-1).mean()
    )
    metrics["wm/reward_exact_covector_norm_mean"] = float(exact_covector.norm(dim=-1).mean())
    if reward_info["reward_curl"].numel() > 0:
        metrics["wm/reward_curl_norm_mean"] = float(
            torch.linalg.norm(reward_info["reward_curl"], dim=(-2, -1)).mean()
        )
    else:
        metrics["wm/reward_curl_norm_mean"] = 0.0

    L_momentum = compute_momentum_regularization(wm_out["momenta"], wm_out["z_trajectory"])
    L_energy = (
        wm_out["energy_var"]
        if "energy_var" in wm_out
        else compute_energy_conservation_loss(
            wm_out["phi_eff"],
            wm_out["momenta"],
            wm_out["z_trajectory"],
        )
    )
    L_hodge = compute_hodge_consistency_loss(wm_out["hodge_harmonic_forces"])
    metrics["wm/L_momentum"] = float(L_momentum)
    metrics["wm/L_energy"] = float(L_energy)
    metrics["wm/L_hodge"] = float(L_hodge)
    L_wm_core = (
        config.w_dynamics * (L_geo + L_chart)
        + config.w_reward * L_reward
        + config.w_reward_nonconservative * L_reward_nonconservative
        + config.w_reward_exact_orth * L_reward_exact_orth
        + config.w_momentum_reg * L_momentum
        + config.w_energy_conservation * L_energy
        + config.w_hodge * L_hodge
    )

    critic_t0 = time.perf_counter()
    replay_values = value_prev_flat.reshape(B, T)
    replay_cons_target = rewards - r_noncons.detach()
    replay_rtg = _discounted_return_to_go(replay_cons_target, replay_dones, config.gamma)
    replay_gap = replay_values - replay_rtg
    L_value = _masked_mean(replay_gap.pow(2), replay_valid)
    L_poisson = compute_screened_poisson_loss(
        critic,
        z_prev,
        None,
        rw_prev,
        reward_density=rho_r,
        kappa=config.screened_poisson_kappa,
    )
    L_critic = config.w_screened_poisson * L_poisson + config.w_critic * L_value
    metrics["time/critic"] = time.perf_counter() - critic_t0
    metrics["critic/L_critic"] = float(L_critic)
    metrics["critic/L_value"] = float(L_value)
    metrics["critic/L_poisson"] = float(L_poisson)

    if shared_critic:
        (L_wm_core + L_critic).backward()
        critic_grad = _parameter_grad_norm(list(critic.parameters()))
        wm_grad = nn.utils.clip_grad_norm_(_optimizer_parameters(optimizer_wm), config.grad_clip)
        optimizer_wm.step()
    else:
        L_wm_core.backward()
        wm_grad = nn.utils.clip_grad_norm_(_optimizer_parameters(optimizer_wm), config.grad_clip)
        optimizer_wm.step()
        if optimizer_critic is not None:
            optimizer_critic.zero_grad()
            L_critic.backward()
            critic_grad = nn.utils.clip_grad_norm_(critic.parameters(), config.grad_clip)
            optimizer_critic.step()
        else:
            critic_grad = 0.0
    metrics["wm/grad_norm"] = float(wm_grad)
    metrics["critic/grad_norm"] = float(critic_grad)
    metrics["time/world_model"] = time.perf_counter() - t_section

    t_section = time.perf_counter()
    optimizer_boundary.zero_grad()
    actor_modules = [
        (_unwrap_compiled_module(world_model), False),
        (_unwrap_compiled_module(reward_head), False),
        (_unwrap_compiled_module(critic), False),
        (action_model, False),
        (closure_model, False),
        (actor, True),
    ]
    with _temporary_requires_grad(actor_modules):
        actor_mean_flat, _actor_log_std_flat = actor(z_prev_flat.detach(), rw_prev_flat.detach())
        L_actor_supervise = F.smooth_l1_loss(actor_mean_flat, action_z_prev_flat.detach())

        actor_update_due = _should_run_actor_update(config, epoch=epoch, update_idx=update_idx)
        L_actor_return = actor_mean_flat.new_zeros(())
        actor_return = actor_mean_flat.new_zeros(())
        actor_reward_mean = 0.0
        actor_control_norm_mean = 0.0
        actor_action_abs_mean = 0.0
        actor_batch_size = 0
        if actor_update_due:
            actor_batch_size = min(int(config.actor_return_batch_size), B)
            if actor_batch_size > 0:
                if actor_batch_size < B:
                    actor_idx = torch.randperm(B, device=z_0.device)[:actor_batch_size]
                    z_actor = z_0[actor_idx]
                    rw_actor = rw_0[actor_idx]
                else:
                    z_actor = z_0
                    rw_actor = rw_0
                actor_rollout = _imagine_actor_return(
                    model,
                    _unwrap_compiled_module(world_model),
                    _unwrap_compiled_module(reward_head),
                    critic,
                    actor,
                    action_model,
                    closure_model,
                    z_actor,
                    rw_actor,
                    horizon=config.actor_return_horizon,
                    gamma=config.gamma,
                )
                actor_return = actor_rollout["objective"].mean()
                L_actor_return = -config.w_actor_return * actor_return
                actor_reward_mean = float(actor_rollout["reward_nonconservative"].detach().mean())
                actor_control_norm_mean = float(
                    actor_rollout["controls_tan"].detach().norm(dim=-1).mean()
                )
                actor_action_abs_mean = float(actor_rollout["actions"].detach().abs().mean())
            else:
                actor_update_due = False
        L_actor_total = config.w_action_latent_supervise * L_actor_supervise + L_actor_return
        L_actor_total.backward()
    actor_grad = nn.utils.clip_grad_norm_(actor.parameters(), config.grad_clip)
    optimizer_boundary.step()
    metrics["actor/L_total"] = float(L_actor_total.detach())
    metrics["actor/L_supervise"] = float(L_actor_supervise.detach())
    metrics["actor/L_return"] = float(L_actor_return.detach())
    metrics["actor/return_mean"] = float(actor_return.detach())
    metrics["actor/reward_mean"] = actor_reward_mean
    metrics["actor/control_norm_mean"] = actor_control_norm_mean
    metrics["actor/action_abs_mean"] = actor_action_abs_mean
    metrics["actor/grad_norm"] = float(actor_grad)
    metrics["actor/update_applied"] = 1.0 if actor_update_due else 0.0
    metrics["actor/horizon"] = float(config.actor_return_horizon if actor_update_due else 0.0)
    metrics["actor/batch_size"] = float(actor_batch_size)
    metrics["time/actor"] = time.perf_counter() - t_section

    if compute_diagnostics:
        t_section = time.perf_counter()
        imagination = _imagine(
            model,
            _unwrap_compiled_module(world_model),
            _unwrap_compiled_module(reward_head),
            critic,
            actor,
            action_model,
            closure_model,
            z_0,
            rw_0,
            config.imagination_horizon,
            config.gamma,
            reward_curl_batch_limit=config.reward_curl_batch_limit,
        )
        imag_rw_states = imagination["rw_states"]
        imag_controls_tan = imagination["controls_tan"]
        imag_controls_cov = imagination["controls_cov"]
        imag_action_latents = imagination["action_latents"]
        imag_action_router = imagination["action_router_weights"]
        imag_rewards = imagination["rewards"]
        imag_reward_cons = imagination["reward_conservative"]
        imag_reward_noncons = imagination["reward_nonconservative"]
        imag_reward_curl = imagination["reward_curl_norm"]
        imag_reward_curl_valid = imagination["reward_curl_valid"]
        imag_log_std = imagination["action_log_std"]
        imag_z = imagination["z_traj"]
        imag_rw = imagination["rw_traj"]
        imag_actions = imagination["actions"]
        discounted_rewards = _discounted_sum(imag_reward_noncons, config.gamma)
        discounted_exact_boundary = _discounted_sum(imag_reward_cons, config.gamma)
        terminal_value = critic_value(critic, imag_z[:, -1], imag_rw[:, -1]).squeeze(-1)
        control_objective = discounted_rewards
        full_return = discounted_exact_boundary + discounted_rewards
        replay_values_all = critic_value(
            critic,
            z_all.reshape(-1, config.latent_dim),
            rw_all.reshape(-1, config.num_charts),
        ).reshape(B, T + 1)
        replay_values_diag = replay_values_all[:, :-1]
        replay_next_values = replay_values_all[:, 1:]
        replay_delta = (
            replay_cons_target
            + config.gamma * (1.0 - replay_dones) * replay_next_values
            - replay_values_diag
        )
        cal_err, cal_max, cal_bins = _value_calibration_error(
            replay_values_diag,
            replay_rtg,
            replay_valid,
        )
        imag_action_router_entropy = -(
            imag_action_router * imag_action_router.clamp(min=1e-8).log()
        ).sum(dim=-1)
        metrics["policy/control_norm_mean"] = float(imag_controls_tan.norm(dim=-1).mean())
        metrics["policy/control_cov_norm_mean"] = float(imag_controls_cov.norm(dim=-1).mean())
        metrics["policy/action_abs_mean"] = float(imag_actions.abs().mean())
        metrics["policy/action_latent_norm_mean"] = float(imag_action_latents.norm(dim=-1).mean())
        metrics["policy/action_router_entropy"] = float(imag_action_router_entropy.mean())
        metrics["boundary/texture_std_mean"] = float(imag_log_std.exp().mean())
        metrics["imagination/reward_mean"] = float(imag_rewards.mean())
        metrics["imagination/reward_std"] = float(imag_rewards.std())
        metrics["imagination/reward_conservative_mean"] = float(imag_reward_cons.mean())
        metrics["imagination/reward_nonconservative_mean"] = float(imag_reward_noncons.mean())
        if imag_reward_curl_valid.any():
            metrics["imagination/reward_curl_norm_mean"] = float(
                _masked_mean(imag_reward_curl, imag_reward_curl_valid.float())
            )
        else:
            metrics["imagination/reward_curl_norm_mean"] = 0.0
        metrics["imagination/reward_curl_eval_frac"] = float(imag_reward_curl_valid.float().mean())
        metrics["imagination/discounted_reward_mean"] = float(discounted_rewards.mean())
        metrics["imagination/exact_boundary_mean"] = float(discounted_exact_boundary.mean())
        metrics["imagination/terminal_value_mean"] = float(terminal_value.mean())
        metrics["imagination/return_mean"] = float(control_objective.mean())
        metrics["imagination/full_return_mean"] = float(full_return.mean())
        metrics["imagination/router_entropy"] = float(
            -(imag_rw_states * imag_rw_states.clamp(min=1e-8).log()).sum(dim=-1).mean()
        )
        metrics["imagination/router_drift"] = float((imag_rw - imag_rw_states).abs().mean())
        metrics["critic/value_bias"] = float(_masked_mean(replay_gap, replay_valid))
        metrics["critic/value_abs_err"] = float(_masked_mean(replay_gap.abs(), replay_valid))
        metrics["critic/value_mean"] = float(_masked_mean(replay_values_diag, replay_valid))
        metrics["critic/replay_bellman_abs"] = float(_masked_mean(replay_delta.abs(), replay_valid))
        bellman_centered = replay_delta - _masked_mean(replay_delta, replay_valid)
        metrics["critic/replay_bellman_std"] = float(
            (_masked_mean(bellman_centered.pow(2), replay_valid) + 1e-8).sqrt()
        )
        metrics["critic/replay_rtg_abs_err"] = float(_masked_mean(replay_gap.abs(), replay_valid))
        metrics["critic/replay_calibration_err"] = float(cal_err)
        metrics["critic/replay_calibration_max"] = float(cal_max)
        metrics["critic/replay_calibration_bins"] = cal_bins
        metrics["train/replay_horizon"] = float(T)
        metrics["train/wm_horizon"] = float(T_wm)
        metrics["geometric/z_norm_mean"] = float(z_all.norm(dim=-1).mean())
        metrics["geometric/z_norm_max"] = float(z_all.norm(dim=-1).max())
        metrics["geometric/action_z_norm_mean"] = float(action_z_all.norm(dim=-1).mean())
        metrics["geometric/jump_frac"] = float(wm_out["jumped"].float().mean())
        metrics["geometric/hodge_conservative"] = float(
            wm_out["hodge_conservative_ratio"].mean()
        )
        metrics["geometric/hodge_solenoidal"] = float(
            wm_out["hodge_solenoidal_ratio"].mean()
        )
        metrics["geometric/hodge_harmonic"] = float(wm_out["hodge_harmonic_ratio"].mean())
        metrics["geometric/energy_var"] = float(wm_out["energy_var"])

        obs_centers = _project_to_ball(model.encoder.chart_centers.detach())
        action_centers = _project_to_ball(action_model.encoder.chart_centers.detach())
        world_model_mod = _unwrap_compiled_module(world_model)
        reward_head_mod = _unwrap_compiled_module(reward_head)
        metrics["chart/wm_center_drift"] = float(
            (
                obs_centers
                - _project_to_ball(world_model_mod.potential_net.chart_tok.chart_centers.detach())
            ).norm(dim=-1).mean()
        )
        metrics["chart/actor_center_drift"] = float(
            (obs_centers - _project_to_ball(actor.chart_tok.chart_centers.detach())).norm(dim=-1).mean()
        )
        metrics["chart/closure_obs_center_drift"] = float(
            (obs_centers - _project_to_ball(closure_model.obs_chart_tok.chart_centers.detach()))
            .norm(dim=-1)
            .mean()
        )
        metrics["action_chart/closure_center_drift"] = float(
            (
                action_centers
                - _project_to_ball(closure_model.action_chart_tok.chart_centers.detach())
            ).norm(dim=-1).mean()
        )
        metrics["action_chart/reward_center_drift"] = float(
            (
                action_centers
                - _project_to_ball(reward_head_mod.action_chart_tok.chart_centers.detach())
            ).norm(dim=-1).mean()
        )
        metrics["time/diagnostics"] = time.perf_counter() - t_section
    else:
        metrics["time/diagnostics"] = 0.0

    metrics["time/step"] = time.perf_counter() - step_t0
    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(config: DreamerConfig) -> None:
    """Run Geometric Dreamer training."""
    device = torch.device(config.device)
    print(f"Device: {device}")
    if config.matmul_precision:
        torch.set_float32_matmul_precision(config.matmul_precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
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
    collect_env = None
    if not config.use_gas and config.collect_n_env_workers > 1:
        collect_env = VectorizedDMControlEnv(
            f"{config.domain}-{config.task}",
            n_workers=config.collect_n_env_workers,
            include_rgb=False,
        )
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
    action_model = SharedDynTopoEncoder(
        input_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_charts=config.num_action_charts,
        codes_per_chart=config.action_codes_per_chart,
        covariant_attn=True,
        covariant_attn_tensorization="full",
        soft_equiv_metric=True,
        conv_backbone=False,
        film_conditioning=True,
        commitment_beta=config.commitment_beta,
        codebook_loss_weight=config.codebook_loss_weight,
    ).to(device)
    action_jump_op = FactorizedJumpOperator(
        num_charts=config.num_action_charts,
        latent_dim=config.latent_dim,
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

    actor = GeometricActor(
        latent_dim=config.latent_dim,
        action_latent_dim=config.latent_dim,
        num_charts=config.num_charts,
        d_model=config.d_model,
    ).to(device)
    closure_model = CovariantObsActionClosureModel(
        latent_dim=config.latent_dim,
        num_action_charts=config.num_action_charts,
        num_obs_charts=config.num_charts,
        obs_codes_per_chart=config.codes_per_chart,
        action_codes_per_chart=config.action_codes_per_chart,
        d_model=config.d_model,
        metric=world_model.metric,
    ).to(device)
    critic = world_model.potential_net

    reward_head = RewardHead(
        potential_net=world_model.potential_net,
        num_action_charts=config.num_action_charts,
        d_model=config.d_model,
    ).to(device)

    world_model = _maybe_compile_module(
        world_model,
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
    print(f"Act enc:     {_count(action_model.encoder):,} params")
    print(f"Act dec:     {_count(action_model.decoder):,} params")
    print(f"Act jump:    {_count(action_jump_op):,} params")
    print(f"World model: {_count(world_model):,} params")
    print(f"Closure:     {_count(closure_model):,} params")
    print(f"Actor:       {_count(actor):,} params")
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
    encoder_groups.extend(
        build_encoder_param_groups(
            action_model,
            action_jump_op,
            base_lr=config.lr_encoder,
            lr_chart_centers_scale=config.lr_chart_centers_scale,
            lr_codebook_scale=config.lr_codebook_scale,
        ),
    )
    encoder_groups.append(
        {
            "params": list(closure_model.parameters()),
            "lr": config.lr_dyn_transition,
        },
    )
    optimizer_enc = torch.optim.Adam(encoder_groups)

    # WM optimizer: world_model + reward_head (minus shared params)
    reward_head_mod = _unwrap_compiled_module(reward_head)
    reward_shared_ids = {
        *(id(p) for p in reward_head_mod.chart_tok.parameters()),
        *(id(p) for p in reward_head_mod.z_embed.parameters()),
    }
    reward_own_params = [
        p for p in reward_head_mod.parameters()
        if id(p) not in reward_shared_ids
    ]
    optimizer_wm = torch.optim.Adam(
        [
            {"params": world_model.parameters(), "lr": config.lr_wm},
            {"params": reward_own_params, "lr": config.lr_wm},
        ],
    )

    optimizer_boundary = torch.optim.Adam(
        actor.parameters(),
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
    phase1_state_obs = init_phase1_adaptive_state(config)
    phase1_state_action = init_phase1_adaptive_state(config)
    phase1_cfg = _phase1_config(config, phase1_state_obs)
    action_phase1_cfg = _phase1_config(
        config,
        phase1_state_action,
        input_dim=config.action_dim,
        num_charts=config.num_action_charts,
        codes_per_chart=config.action_codes_per_chart,
    )

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
        if "action_model" in ckpt:
            action_model.load_state_dict(ckpt["action_model"], strict=False)
        if "action_jump_op" in ckpt:
            action_jump_op.load_state_dict(ckpt["action_jump_op"], strict=False)
        if "closure_model" in ckpt:
            closure_model.load_state_dict(ckpt["closure_model"], strict=False)
        if "world_model" in ckpt:
            _unwrap_compiled_module(world_model).load_state_dict(ckpt["world_model"], strict=False)
        if "actor" in ckpt:
            actor.load_state_dict(ckpt["actor"], strict=False)
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
        action_model.encoder.eval()
        action_model.decoder.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        for p in action_model.parameters():
            p.requires_grad_(False)
        jump_op.eval()
        action_jump_op.eval()
        for p in jump_op.parameters():
            p.requires_grad_(False)
        for p in action_jump_op.parameters():
            p.requires_grad_(False)
        print("Observation and action topoencoders frozen.")

    _sync_rl_atlas(model, action_model, world_model, critic, actor, closure_model, reward_head)
    print("Bound RL chart centers to encoder atlas.")

    # --- Replay buffer ---
    buffer = SequenceReplayBuffer(
        capacity=config.buffer_capacity,
        seq_len=config.seq_len,
    )

    # --- Seed episodes (random policy) ---
    print(f"Collecting {config.seed_episodes} seed episodes...")
    seed_episodes_data: list[dict[str, np.ndarray]] = []
    seed_count = 0
    while seed_count < config.seed_episodes:
        batch_episodes = min(
            config.seed_episodes - seed_count,
            config.collect_n_env_workers if collect_env is not None else 1,
        )
        if collect_env is not None:
            episodes = _collect_parallel_episodes(
                collect_env,
                None,
                None,
                None,
                model,
                device,
                config.latent_dim,
                config.num_action_charts,
                num_episodes=batch_episodes,
                obs_normalizer=obs_normalizer,
                action_repeat=config.action_repeat,
                max_steps=config.max_episode_steps,
                hard_routing=config.hard_routing,
                hard_routing_tau=config.hard_routing_tau,
                use_motor_texture=config.use_motor_texture,
            )
        else:
            episodes = [
                _collect_episode(
                    env,
                    None,
                    None,
                    None,
                    model,
                    device,
                    config.latent_dim,
                    config.num_action_charts,
                    obs_normalizer=obs_normalizer,
                    action_repeat=config.action_repeat,
                    max_steps=config.max_episode_steps,
                    hard_routing=config.hard_routing,
                    hard_routing_tau=config.hard_routing_tau,
                    use_motor_texture=config.use_motor_texture,
                ),
            ]
        for ep in episodes:
            buffer.add_episode(ep)
            seed_episodes_data.append(ep)
            seed_count += 1
            ep_r = ep["rewards"].sum()
            print(f"  Seed {seed_count}/{config.seed_episodes}: reward={ep_r:.1f}  len={len(ep['obs'])}")

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
                    actor,
                    action_model,
                    closure_model,
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
        elif epoch % config.collect_every == 0:
            collect_t0 = time.perf_counter()
            if collect_env is not None:
                episodes = _collect_parallel_episodes(
                    collect_env,
                    actor,
                    action_model,
                    closure_model,
                    model,
                    device,
                    config.latent_dim,
                    config.num_action_charts,
                    num_episodes=config.collect_n_env_workers,
                    obs_normalizer=obs_normalizer,
                    action_repeat=config.action_repeat,
                    max_steps=config.max_episode_steps,
                    hard_routing=config.hard_routing,
                    hard_routing_tau=config.hard_routing_tau,
                    use_motor_texture=config.use_motor_texture,
                )
            else:
                episodes = [
                    _collect_episode(
                        env,
                        actor,
                        action_model,
                        closure_model,
                        model,
                        device,
                        config.latent_dim,
                        config.num_action_charts,
                        obs_normalizer=obs_normalizer,
                        action_repeat=config.action_repeat,
                        max_steps=config.max_episode_steps,
                        hard_routing=config.hard_routing,
                        hard_routing_tau=config.hard_routing_tau,
                        use_motor_texture=config.use_motor_texture,
                    ),
                ]
            for ep in episodes:
                buffer.add_episode(ep)
                ep_reward = ep["rewards"].sum()
                episode_rewards.append(ep_reward)
                total_env_steps += len(ep["obs"])
            collect_time += time.perf_counter() - collect_t0

        # --- Training steps (multiple updates per epoch) ---
        if not config.freeze_encoder:
            model.train()
            jump_op.train()
            action_model.train()
            action_jump_op.train()
        closure_model.train()
        world_model.train()
        actor.train()
        critic.train()
        reward_head.train()

        tokens_per_batch = config.batch_size * max(config.seq_len, 1)
        if config.updates_per_epoch > 0:
            n_updates = config.updates_per_epoch
        else:
            n_updates = max(1, buffer.total_steps // tokens_per_batch)

        current_hard_routing = _use_hard_routing(config, epoch)
        current_tau = _get_hard_routing_tau(config, epoch, config.total_epochs)
        phase1_cfg = _phase1_config(config, phase1_state_obs)
        action_phase1_cfg = _phase1_config(
            config,
            phase1_state_action,
            input_dim=config.action_dim,
            num_charts=config.num_action_charts,
            codes_per_chart=config.action_codes_per_chart,
        )

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
                action_model,
                action_jump_op,
                closure_model,
                world_model,
                reward_head,
                critic,
                actor,
                optimizer_enc,
                optimizer_wm,
                optimizer_critic,
                optimizer_boundary,
                batch,
                config,
                phase1_cfg,
                action_phase1_cfg,
                epoch,
                current_hard_routing,
                current_tau,
                update_idx=epoch * max(n_updates, 1) + _u,
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
        controller_metrics_obs = _phase1_controller_metrics(
            metrics,
            enc_prefix="enc",
            chart_prefix="chart",
            num_charts=config.num_charts,
        )
        controller_metrics_action = _phase1_controller_metrics(
            metrics,
            enc_prefix="enc_action",
            chart_prefix="action_chart",
            num_charts=config.num_action_charts,
        )
        update_phase1_adaptive_state(
            phase1_state_obs,
            config,
            metrics,
            controller_metrics_obs,
            epoch,
        )
        update_phase1_adaptive_state(
            phase1_state_action,
            config,
            metrics,
            controller_metrics_action,
            epoch,
        )
        current_scales_obs = phase1_effective_weight_scales(config, phase1_state_obs)
        current_scales_action = phase1_effective_weight_scales(config, phase1_state_action)
        for name, value in current_scales_obs.items():
            metrics[f"phase1_obs/{name}"] = value
        for name, value in current_scales_action.items():
            metrics[f"phase1_action/{name}"] = value
        metrics["phase1/entropy_scale"] = 0.5 * (
            current_scales_obs["entropy_scale"] + current_scales_action["entropy_scale"]
        )
        metrics["phase1/chart_usage_scale"] = 0.5 * (
            current_scales_obs["chart_usage_scale"] + current_scales_action["chart_usage_scale"]
        )
        metrics["phase1/chart_ot_scale"] = 0.5 * (
            current_scales_obs["chart_ot_scale"] + current_scales_action["chart_ot_scale"]
        )
        metrics["phase1/code_usage_scale"] = 0.5 * (
            current_scales_obs["code_usage_scale"] + current_scales_action["code_usage_scale"]
        )

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
            line1_keys = [
                ("ep_rew", "env/last_episode_reward"),
                ("rew_20", "env/mean_episode_reward_20"),
                ("enc_o", "enc/L_total"),
                ("enc_a", "enc_action/L_total"),
                ("close", "closure/L_total"),
                ("L_geo", "wm/L_geodesic"),
                ("L_rew", "wm/L_reward"),
                ("L_crit", "critic/L_critic"),
                ("L_act", "actor/L_total"),
                ("lr_a", "train/lr_boundary"),
            ]
            line2_keys = [
                ("recon", "enc/recon"),
                ("vq", "enc/vq"),
                ("a_recon", "enc_action/recon"),
                ("a_vq", "enc_action/vq"),
                ("cl_obs", "closure/L_obs_state"),
                ("cl_act", "closure/L_action_state"),
                ("code_H", "enc/H_code_usage"),
                ("act_H", "action_chart/usage_entropy"),
                ("enc_gn", "enc/grad_norm"),
                ("act_gn", "actor/grad_norm"),
            ]
            line3_keys = [
                ("ctrl", "policy/control_norm_mean"),
                ("a_lat", "policy/action_latent_norm_mean"),
                ("a_ent", "policy/action_router_entropy"),
                ("tex", "boundary/texture_std_mean"),
                ("im_rew", "imagination/reward_mean"),
                ("im_ret", "imagination/return_mean"),
                ("act_ret", "actor/return_mean"),
                ("value", "critic/value_mean"),
                ("wm_gn", "wm/grad_norm"),
            ]
            line4_keys = [
                ("z_norm", "geometric/z_norm_mean"),
                ("az_norm", "geometric/action_z_norm_mean"),
                ("jump", "geometric/jump_frac"),
                ("cons", "geometric/hodge_conservative"),
                ("sol", "geometric/hodge_solenoidal"),
                ("e_var", "geometric/energy_var"),
                ("ch_ent", "chart/usage_entropy"),
                ("ach_ent", "action_chart/usage_entropy"),
                ("wm_ctr", "chart/wm_center_drift"),
                ("cl_ctr", "chart/closure_obs_center_drift"),
                ("acl_ctr", "action_chart/closure_center_drift"),
            ]
            line5_keys = [
                ("obj", "imagination/return_mean"),
                ("dret", "imagination/discounted_reward_mean"),
                ("term", "imagination/terminal_value_mean"),
                ("exbd", "imagination/exact_boundary_mean"),
                ("chart_acc", "wm/chart_acc"),
                ("rw_drift", "imagination/router_drift"),
                ("v_err", "critic/value_abs_err"),
                ("cov_n", "policy/control_cov_norm_mean"),
            ]
            line6_keys = [
                ("col", "time/collection"),
                ("smp", "time/sample"),
                ("enc_t", "time/encoder"),
                ("wm_t", "time/world_model"),
                ("crt_t", "time/critic"),
                ("act_t", "time/actor"),
                ("diag_t", "time/diagnostics"),
            ]
            line7_keys = [
                ("wm_ctr", "chart/wm_center_drift"),
                ("bell", "critic/replay_bellman_abs"),
                ("bell_s", "critic/replay_bellman_std"),
                ("rtg_e", "critic/replay_rtg_abs_err"),
                ("cal_e", "critic/replay_calibration_err"),
                ("c_sup", "closure/L_control_supervise"),
                ("a_sup", "actor/L_supervise"),
                ("upd", "actor/update_applied"),
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

            action_usage_parts = []
            for k in range(config.num_action_charts):
                key = f"action_chart/{k}/usage"
                if key in metrics:
                    action_usage_parts.append(f"a{k}={metrics[key]:.2f}")
            if action_usage_parts:
                active_action_charts = int(round(metrics.get("action_chart/active_charts", 0.0)))
                print(
                    f"  {'':4s}  action charts: {active_action_charts}/{config.num_action_charts} active  "
                    f"{' '.join(action_usage_parts)}",
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

            action_symbol_parts = []
            for k in range(config.num_action_charts):
                active_key = f"action_chart/{k}/active_codes"
                entropy_key = f"action_chart/{k}/code_entropy"
                if active_key in metrics and entropy_key in metrics:
                    active_codes = int(round(metrics[active_key]))
                    action_symbol_parts.append(
                        f"a{k}={active_codes}/{config.action_codes_per_chart}(H={metrics[entropy_key]:.2f})",
                    )
            if action_symbol_parts:
                active_action_symbols = int(round(metrics.get("action_chart/active_symbols", 0.0)))
                total_action_symbols = config.num_action_charts * config.action_codes_per_chart
                print(
                    f"  {'':4s}  action symbols: {active_action_symbols}/{total_action_symbols} active  "
                    f"{' '.join(action_symbol_parts)}",
                )

            # MLflow
            if mlflow_enabled and log_mlflow_metrics is not None:
                prefixed = {f"phase4/{k}": v for k, v in metrics.items()}
                log_mlflow_metrics(prefixed, step=epoch, enabled=True)

        # --- Evaluation ---
        if epoch % config.eval_every == 0 and epoch > 0:
            model.encoder.eval()
            action_model.eval()
            closure_model.eval()
            actor.eval()
            critic.eval()
            eval_metrics = _eval_policy(
                env,
                actor,
                action_model,
                closure_model,
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
                    action_model,
                    action_jump_op,
                    closure_model,
                    world_model,
                    actor,
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
                action_model,
                action_jump_op,
                closure_model,
                world_model,
                actor,
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
    if collect_env is not None:
        collect_env.close()
    if mlflow_enabled and end_mlflow_run is not None:
        end_mlflow_run(enabled=True)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def _save_checkpoint(
    path: str,
    model: TopoEncoderPrimitives,
    jump_op: nn.Module,
    action_model: nn.Module,
    action_jump_op: nn.Module,
    closure_model: nn.Module,
    world_model: nn.Module,
    actor: nn.Module,
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
        "action_model": {
            k: v.cpu() for k, v in _unwrap_compiled_module(action_model).state_dict().items()
        },
        "action_jump_op": {
            k: v.cpu() for k, v in _unwrap_compiled_module(action_jump_op).state_dict().items()
        },
        "closure_model": {
            k: v.cpu() for k, v in _unwrap_compiled_module(closure_model).state_dict().items()
        },
        "world_model": {
            k: v.cpu() for k, v in _unwrap_compiled_module(world_model).state_dict().items()
        },
        "actor": {
            k: v.cpu() for k, v in _unwrap_compiled_module(actor).state_dict().items()
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
