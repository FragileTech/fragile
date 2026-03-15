"""Current-path tests for the theory-aligned Dreamer RL stack."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


B = 4
D = 8
A = 6
K = 4
CODES_PER_CHART = 4
D_MODEL = 32
OBS_DIM = 24
H_IMAGINATION = 3
H_WM = 2


def _random_action_form_inputs(
    z: torch.Tensor,
    *,
    num_action_charts: int = K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action_z = torch.randn_like(z) * 0.2
    action_rw = torch.softmax(
        torch.randn(z.shape[0], num_action_charts, device=z.device, dtype=z.dtype),
        dim=-1,
    )
    action_code_z = torch.randn_like(z) * 0.2
    return action_z, action_rw, action_code_z


def _make_replay_episode(length: int) -> dict[str, np.ndarray]:
    router = np.random.rand(length, K).astype(np.float32)
    router /= router.sum(axis=-1, keepdims=True)
    return {
        "obs": np.random.randn(length, OBS_DIM).astype(np.float32),
        "actions": np.random.randn(length, A).astype(np.float32),
        "action_means": np.random.randn(length, A).astype(np.float32),
        "controls": np.random.randn(length, D).astype(np.float32),
        "controls_tan": np.random.randn(length, D).astype(np.float32),
        "controls_cov": np.random.randn(length, D).astype(np.float32),
        "control_valid": np.ones(length, dtype=np.float32),
        "action_latents": np.random.randn(length, D).astype(np.float32),
        "action_router_weights": router,
        "action_charts": np.random.randint(0, K, size=length, dtype=np.int64),
        "action_codes": np.random.randint(0, CODES_PER_CHART, size=length, dtype=np.int64),
        "action_code_latents": np.random.randn(length, D).astype(np.float32),
        "rewards": np.random.randn(length).astype(np.float32),
        "dones": np.zeros(length, dtype=np.float32),
    }


def _make_training_batch(device: torch.device) -> dict[str, torch.Tensor]:
    time_dim = H_IMAGINATION + 3
    controls_tan = torch.randn(B, time_dim, D, device=device) * 0.05
    controls_cov = torch.randn(B, time_dim, D, device=device) * 0.05
    control_valid = torch.ones(B, time_dim, device=device)
    control_valid[:, -1] = 0.0
    rewards = torch.randn(B, time_dim, device=device) * 0.1
    dones = torch.zeros(B, time_dim, device=device)
    return {
        "obs": torch.randn(B, time_dim, OBS_DIM, device=device),
        "actions": torch.randn(B, time_dim, A, device=device) * 0.2,
        "action_means": torch.randn(B, time_dim, A, device=device) * 0.2,
        "controls": controls_cov.clone(),
        "controls_tan": controls_tan,
        "controls_cov": controls_cov,
        "control_valid": control_valid,
        "rewards": rewards,
        "dones": dones,
    }


def _make_optimizers(
    obs_model: nn.Module,
    jump_op: nn.Module,
    action_model: nn.Module,
    action_jump_op: nn.Module,
    closure_model: nn.Module,
    world_model: nn.Module,
    reward_head: nn.Module,
    actor: nn.Module,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer]:
    reward_shared_ids = {
        *(id(param) for param in reward_head.chart_tok.parameters()),
        *(id(param) for param in reward_head.z_embed.parameters()),
    }
    reward_own_params = [
        param for param in reward_head.parameters() if id(param) not in reward_shared_ids
    ]
    optimizer_enc = torch.optim.Adam(
        list(obs_model.parameters())
        + list(jump_op.parameters())
        + list(action_model.parameters())
        + list(action_jump_op.parameters())
        + list(closure_model.parameters()),
        lr=1e-3,
    )
    optimizer_wm = torch.optim.Adam(
        list(world_model.parameters()) + reward_own_params,
        lr=1e-3,
    )
    optimizer_boundary = torch.optim.Adam(actor.parameters(), lr=1e-3)
    return optimizer_enc, optimizer_wm, optimizer_boundary


def _make_constant_policy_output(
    z_in: torch.Tensor,
    *,
    action_value: float = 0.25,
    code_idx: int = 1,
) -> dict[str, torch.Tensor]:
    batch = z_in.shape[0]
    action_router = torch.zeros(batch, K, device=z_in.device, dtype=z_in.dtype)
    action_router[:, 0] = 1.0
    return {
        "action": torch.full((batch, A), action_value, device=z_in.device, dtype=z_in.dtype),
        "action_mean": torch.full((batch, A), action_value, device=z_in.device, dtype=z_in.dtype),
        "action_latent": torch.full((batch, D), 0.15, device=z_in.device, dtype=z_in.dtype),
        "action_latent_mean": torch.full((batch, D), 0.15, device=z_in.device, dtype=z_in.dtype),
        "action_router_weights": action_router,
        "action_chart_idx": torch.zeros(batch, device=z_in.device, dtype=torch.long),
        "action_code_idx": torch.full((batch,), code_idx, device=z_in.device, dtype=torch.long),
        "action_code_latent": torch.full((batch, D), 0.05, device=z_in.device, dtype=z_in.dtype),
        "control_tan": torch.full((batch, D), 0.3, device=z_in.device, dtype=z_in.dtype),
        "control_cov": torch.full((batch, D), 0.6, device=z_in.device, dtype=z_in.dtype),
    }


class FakeTimeStep:
    def __init__(self, obs, reward: float = 0.0, last: bool = False):
        self.observation = {"obs": np.asarray(obs, dtype=np.float32)}
        self.reward = reward
        self._last = last

    def last(self) -> bool:
        return self._last


class FakeActionSpec:
    def __init__(self, action_dim: int = A):
        self.minimum = -np.ones(action_dim, dtype=np.float32)
        self.maximum = np.ones(action_dim, dtype=np.float32)
        self.shape = (action_dim,)


class SingleEpisodeEnv:
    def __init__(
        self,
        *,
        start_obs: list[float] | np.ndarray,
        next_obs: list[float] | np.ndarray,
        reward: float = 1.0,
        action_dim: int = A,
    ):
        self.start_obs = np.asarray(start_obs, dtype=np.float32)
        self.next_obs = np.asarray(next_obs, dtype=np.float32)
        self.reward = reward
        self._step = 0
        self._action_spec = FakeActionSpec(action_dim)

    def reset(self):
        self._step = 0
        return FakeTimeStep(self.start_obs, last=False)

    def action_spec(self):
        return self._action_spec

    def step(self, action):
        del action
        self._step += 1
        return FakeTimeStep(self.next_obs, reward=self.reward, last=self._step >= 1)


class ParallelEnv:
    def __init__(self):
        self.n_workers = 2
        self.action_space = FakeActionSpec(action_dim=A)
        self.steps = np.zeros(2, dtype=np.int32)
        self.batch_calls: list[list[int]] = []

    def reset_batch(self, env_indices=None, seeds=None):
        del seeds
        indices = np.asarray(env_indices, dtype=int)
        self.steps[indices] = 0
        return np.stack(
            [np.full(OBS_DIM, float(idx) + 0.1, dtype=np.float32) for idx in indices],
        )

    def step_actions_batch(self, actions, dt=None, env_indices=None):
        del actions, dt
        indices = np.asarray(env_indices, dtype=int)
        self.batch_calls.append(indices.tolist())
        obs = []
        rewards = np.zeros(len(indices), dtype=np.float32)
        dones = np.zeros(len(indices), dtype=bool)
        truncated = np.zeros(len(indices), dtype=bool)
        infos = [{} for _ in range(len(indices))]
        for row, idx in enumerate(indices):
            self.steps[idx] += 1
            rewards[row] = float(idx + 1)
            done_after = 1 if idx == 0 else 2
            dones[row] = self.steps[idx] >= done_after
            obs.append(np.full(OBS_DIM, float(idx + self.steps[idx]), dtype=np.float32))
        return np.stack(obs), rewards, dones, truncated, infos


class RecordingEncoder:
    def __init__(self, *, latent_dim: int = D, num_charts: int = K):
        self.latent_dim = latent_dim
        self.num_charts = num_charts
        self.calls: list[tuple[bool, float]] = []
        self.obs_seen: list[torch.Tensor] = []

    def __call__(self, obs_t, *, hard_routing, hard_routing_tau):
        self.calls.append((hard_routing, hard_routing_tau))
        self.obs_seen.append(obs_t.clone())
        batch = obs_t.shape[0]
        rw = torch.zeros(batch, self.num_charts, device=obs_t.device, dtype=obs_t.dtype)
        rw[:, 0] = 1.0
        z = torch.full((batch, self.latent_dim), 0.25, device=obs_t.device, dtype=obs_t.dtype)
        z_q = torch.full_like(z, 0.05)
        out = [None] * 12
        out[4] = rw
        out[5] = z
        out[11] = z_q
        return tuple(out)


class FakeEncoderModel:
    def __init__(self, *, latent_dim: int = D, num_charts: int = K):
        self.encoder = RecordingEncoder(latent_dim=latent_dim, num_charts=num_charts)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def z(device):
    return torch.randn(B, D, device=device) * 0.2


@pytest.fixture
def rw(device):
    return torch.softmax(torch.randn(B, K, device=device), dim=-1)


@pytest.fixture
def actor():
    from fragile.learning.rl.actor import GeometricActor

    return GeometricActor(
        latent_dim=D,
        num_obs_charts=K,
        obs_codes_per_chart=CODES_PER_CHART,
        num_action_charts=K,
        action_codes_per_chart=CODES_PER_CHART,
        d_model=D_MODEL,
    )


@pytest.fixture
def standalone_critic():
    from fragile.learning.rl.critic import GeometricCritic

    return GeometricCritic(D, K, D_MODEL)


@pytest.fixture
def obs_model():
    from fragile.learning.vla.shared_dyn.encoder import SharedDynTopoEncoder

    return SharedDynTopoEncoder(
        input_dim=OBS_DIM,
        hidden_dim=D_MODEL,
        latent_dim=D,
        num_charts=K,
        codes_per_chart=CODES_PER_CHART,
    )


@pytest.fixture
def action_model():
    from fragile.learning.vla.shared_dyn.encoder import SharedDynTopoEncoder

    return SharedDynTopoEncoder(
        input_dim=A,
        hidden_dim=D_MODEL,
        latent_dim=D,
        num_charts=K,
        codes_per_chart=CODES_PER_CHART,
    )


@pytest.fixture
def jump_op():
    from fragile.learning.core.layers import FactorizedJumpOperator

    return FactorizedJumpOperator(num_charts=K, latent_dim=D)


@pytest.fixture
def action_jump_op():
    from fragile.learning.core.layers import FactorizedJumpOperator

    return FactorizedJumpOperator(num_charts=K, latent_dim=D)


@pytest.fixture
def world_model():
    from fragile.learning.vla.covariant_world_model import GeometricWorldModel

    return GeometricWorldModel(
        latent_dim=D,
        action_dim=A,
        control_dim=D,
        num_charts=K,
        d_model=D_MODEL,
        hidden_dim=D_MODEL,
        n_refine_steps=2,
        use_boris=False,
        use_jump=False,
    )


@pytest.fixture
def critic(world_model):
    return world_model.potential_net


@pytest.fixture
def closure_model(world_model):
    from fragile.learning.rl.action_manifold import CovariantObsActionClosureModel

    return CovariantObsActionClosureModel(
        latent_dim=D,
        num_obs_charts=K,
        num_action_charts=K,
        obs_codes_per_chart=CODES_PER_CHART,
        action_codes_per_chart=CODES_PER_CHART,
        d_model=D_MODEL,
        metric=world_model.metric,
    )


@pytest.fixture
def reward_head(world_model):
    from fragile.learning.rl.reward_head import RewardHead

    return RewardHead(world_model.potential_net, K, D_MODEL)


@pytest.fixture
def config(tmp_path):
    from fragile.learning.rl.config import DreamerConfig

    return DreamerConfig(
        device="cpu",
        obs_dim=OBS_DIM,
        action_dim=A,
        latent_dim=D,
        num_charts=K,
        num_action_charts=K,
        d_model=D_MODEL,
        hidden_dim=D_MODEL,
        codes_per_chart=CODES_PER_CHART,
        action_codes_per_chart=CODES_PER_CHART,
        wm_prediction_horizon=H_WM,
        imagination_horizon=H_IMAGINATION,
        batch_size=B,
        seq_len=6,
        total_epochs=2,
        checkpoint_dir=str(tmp_path / "ckpt"),
        use_gas=False,
        normalize_observations=False,
        actor_return_horizon=2,
        actor_return_batch_size=2,
        actor_return_update_every=1,
        actor_return_warmup_epochs=0,
        reward_curl_batch_limit=2,
    )


@pytest.fixture
def phase1_cfg(config):
    from fragile.learning.rl.train_dreamer import _phase1_config

    return _phase1_config(config)


@pytest.fixture
def action_phase1_cfg(config):
    from fragile.learning.rl.train_dreamer import _phase1_config

    return _phase1_config(
        config,
        input_dim=config.action_dim,
        num_charts=config.num_action_charts,
        codes_per_chart=config.action_codes_per_chart,
    )


class TestGeometricActor:
    def test_forward_outputs_structured_action_state(
        self,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        closure_model,
        reward_head,
        z,
    ):
        from fragile.learning.rl.action_manifold import symbolize_latent_with_atlas
        from fragile.learning.rl.train_dreamer import _sync_rl_atlas

        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, closure_model, reward_head)
        obs_info = symbolize_latent_with_atlas(obs_model, z, hard_routing=False, hard_routing_tau=1.0)
        out = actor(
            obs_info["chart_idx"],
            obs_info["code_idx"],
            obs_info["z_n"],
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        assert out["action_chart_logits"].shape == (B, K)
        assert out["action_code_logits"].shape == (B, K, CODES_PER_CHART)
        assert out["action_z_n"].shape == (B, D)
        assert out["action_z_geo"].shape == (B, D)
        assert (out["action_z_geo"].norm(dim=-1) < 1.0).all()

    def test_sample_and_mode_latents_are_deterministic(
        self,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        closure_model,
        reward_head,
        z,
    ):
        from fragile.learning.rl.action_manifold import symbolize_latent_with_atlas
        from fragile.learning.rl.train_dreamer import _sync_rl_atlas

        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, closure_model, reward_head)
        actor.eval()
        obs_info = symbolize_latent_with_atlas(obs_model, z, hard_routing=False, hard_routing_tau=1.0)
        sample = actor.sample_latent(
            obs_info["chart_idx"],
            obs_info["code_idx"],
            obs_info["z_n"],
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        mode = actor.mode_latent(
            obs_info["chart_idx"],
            obs_info["code_idx"],
            obs_info["z_n"],
        )
        torch.testing.assert_close(sample["action_z_geo"], mode["action_z_geo"], atol=1e-6, rtol=0)
        torch.testing.assert_close(
            mode["action_z_geo"],
            actor.mode_latent(obs_info["chart_idx"], obs_info["code_idx"], obs_info["z_n"])["action_z_geo"],
            atol=1e-6,
            rtol=0,
        )

    def test_gradients_flow_through_actor(
        self,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        closure_model,
        reward_head,
        z,
    ):
        from fragile.learning.rl.action_manifold import symbolize_latent_with_atlas
        from fragile.learning.rl.train_dreamer import _sync_rl_atlas

        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, closure_model, reward_head)
        obs_info = symbolize_latent_with_atlas(obs_model, z, hard_routing=False, hard_routing_tau=1.0)
        action_out = actor(
            obs_info["chart_idx"],
            obs_info["code_idx"],
            obs_info["z_n"],
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        action_out["action_z_geo"].mean().backward()
        grad_norms = [param.grad.norm().item() for param in actor.parameters() if param.grad is not None]
        assert grad_norms
        assert any(norm > 0.0 for norm in grad_norms)


class TestGeometricCritic:
    def test_forward_shape_and_finiteness(self, standalone_critic, z, rw):
        value = standalone_critic(z, rw)
        assert value.shape == (B, 1)
        assert torch.isfinite(value).all()

    def test_gradient_flows_to_state(self, standalone_critic, z, rw):
        z_req = z.clone().requires_grad_(True)
        standalone_critic(z_req, rw).sum().backward()
        assert z_req.grad is not None
        assert torch.isfinite(z_req.grad).all()


class TestBoundaryGeometry:
    def test_raise_lower_roundtrip(self, z):
        from fragile.learning.rl.boundary import lower_control, raise_control

        control_tan = torch.randn_like(z)
        control_cov = lower_control(z, control_tan)
        control_tan_rt = raise_control(z, control_cov)
        torch.testing.assert_close(control_tan_rt, control_tan, atol=1e-5, rtol=1e-5)

    def test_critic_control_field_matches_metric_raise(self, standalone_critic, z, rw):
        from fragile.learning.rl.boundary import critic_control_field, lower_control

        control_cov, control_tan, value = critic_control_field(standalone_critic, z, rw)
        assert control_cov.shape == (B, D)
        assert control_tan.shape == (B, D)
        assert value.shape == (B, 1)
        torch.testing.assert_close(lower_control(z, control_tan), control_cov, atol=1e-5, rtol=1e-5)


class TestRewardHead:
    def test_decompose_uses_current_action_manifold_inputs(self, reward_head, z, rw):
        action_z, action_rw, action_code_z = _random_action_form_inputs(z)
        control = torch.randn_like(z)
        info = reward_head.decompose(
            z,
            rw,
            action_z,
            action_rw,
            action_code_z,
            control=control,
            exact_covector=torch.randn_like(z),
        )
        assert info["reward_nonconservative"].shape == (B, 1)
        assert info["reward_density"].shape == (B, 1)
        assert info["reward_form_cov"].shape == (B, D)
        assert info["reward_form_cov_raw"].shape == (B, D)
        assert info["reward_form_exact_component"].shape == (B, D)

    def test_chart_and_embedding_are_shared_with_potential_net(self, reward_head, world_model):
        assert reward_head.chart_tok is world_model.potential_net.chart_tok
        assert reward_head.z_embed is world_model.potential_net.z_embed


class TestPolicyAction:
    def test_outputs_action_manifold_and_boundary_fields(
        self,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        closure_model,
        reward_head,
        z,
        rw,
    ):
        from fragile.learning.rl.action_manifold import symbolize_latent_with_atlas
        from fragile.learning.rl.train_dreamer import _policy_action, _sync_rl_atlas

        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, closure_model, reward_head)
        obs_info = symbolize_latent_with_atlas(
            obs_model,
            z,
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        out = _policy_action(
            actor,
            action_model,
            closure_model,
            obs_info,
            use_motor_texture=True,
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        assert out["action"].shape == (B, A)
        assert out["action_mean"].shape == (B, A)
        assert out["action_latent"].shape == (B, D)
        assert out["action_latent_mean"].shape == (B, D)
        assert out["action_router_weights"].shape == (B, K)
        assert out["action_chart_idx"].shape == (B,)
        assert out["action_code_idx"].shape == (B,)
        assert out["action_code_latent"].shape == (B, D)
        assert out["control_tan"].shape == (B, D)
        assert out["control_cov"].shape == (B, D)
        torch.testing.assert_close(out["action"], out["action_mean"])
        torch.testing.assert_close(out["action_latent"], out["action_latent_mean"])
        torch.testing.assert_close(
            out["action_router_weights"].sum(dim=-1),
            torch.ones(B, device=out["action_router_weights"].device),
        )

    def test_preserves_actor_structured_action_state(
        self,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        closure_model,
        reward_head,
        z,
    ):
        from fragile.learning.rl.action_manifold import symbolize_latent_with_atlas
        from fragile.learning.rl.train_dreamer import _policy_action, _sync_rl_atlas

        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, closure_model, reward_head)
        actor.eval()
        obs_info = symbolize_latent_with_atlas(
            obs_model,
            z,
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        actor_out = actor(
            obs_info["chart_idx"],
            obs_info["code_idx"],
            obs_info["z_n"],
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        out = _policy_action(
            actor,
            action_model,
            closure_model,
            obs_info,
            use_motor_texture=False,
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        torch.testing.assert_close(out["action_latent"], actor_out["action_z_geo"])
        torch.testing.assert_close(out["action_router_weights"], actor_out["action_router_weights"])
        torch.testing.assert_close(out["action_code_latent"], actor_out["action_z_q"])
        torch.testing.assert_close(out["action_chart_idx"], actor_out["action_chart_idx"])
        torch.testing.assert_close(out["action_code_idx"], actor_out["action_code_idx"])


class TestReplayBuffer:
    def test_samples_current_action_manifold_schema(self):
        from fragile.learning.rl.replay_buffer import SequenceReplayBuffer

        buffer = SequenceReplayBuffer(capacity=10_000, seq_len=3)
        buffer.add_episode(_make_replay_episode(10))
        batch = buffer.sample(2, device="cpu")
        assert batch["obs"].shape == (2, 4, OBS_DIM)
        assert batch["actions"].shape == (2, 4, A)
        assert batch["controls"].shape == (2, 4, D)
        assert batch["controls_tan"].shape == (2, 4, D)
        assert batch["controls_cov"].shape == (2, 4, D)
        assert batch["control_valid"].shape == (2, 4)
        assert batch["action_latents"].shape == (2, 4, D)
        assert batch["action_router_weights"].shape == (2, 4, K)
        assert batch["action_charts"].shape == (2, 4)
        assert batch["action_codes"].shape == (2, 4)
        assert batch["action_code_latents"].shape == (2, 4, D)

    def test_short_episode_fallback_keeps_schema(self):
        from fragile.learning.rl.replay_buffer import SequenceReplayBuffer

        buffer = SequenceReplayBuffer(capacity=10_000, seq_len=100)
        buffer.add_episode(_make_replay_episode(5))
        batch = buffer.sample(2, device="cpu")
        assert batch["obs"].shape[0] == 2
        assert batch["action_router_weights"].shape[1] == 5


class TestDreamerHelpers:
    def test_build_episode_dict_appends_zero_tail(self):
        from fragile.learning.rl.train_dreamer import _build_episode_dict

        obs = [np.ones(OBS_DIM, dtype=np.float32), np.full(OBS_DIM, 2.0, dtype=np.float32)]
        actions = [np.ones(A, dtype=np.float32)]
        rewards = [np.float32(1.0)]
        dones = [np.float32(0.0)]
        action_means = [np.full(A, 0.5, dtype=np.float32)]
        control_tan = [np.ones(D, dtype=np.float32)]
        control_cov = [np.full(D, 2.0, dtype=np.float32)]
        control_valid = [np.float32(1.0)]
        action_latents = [np.full(D, 0.1, dtype=np.float32)]
        action_router = [np.eye(K, dtype=np.float32)[0]]
        action_charts = [np.int64(2)]
        action_codes = [np.int64(3)]
        action_code_latents = [np.full(D, 0.2, dtype=np.float32)]

        episode = _build_episode_dict(
            obs,
            actions,
            rewards,
            dones,
            action_means,
            control_tan,
            control_cov,
            control_valid,
            action_latents,
            action_router,
            action_charts,
            action_codes,
            action_code_latents,
        )

        assert episode["obs"].shape == (2, OBS_DIM)
        assert episode["actions"].shape == (2, A)
        assert episode["controls_cov"].shape == (2, D)
        assert episode["action_router_weights"].shape == (2, K)
        assert episode["actions"][-1].sum() == 0.0
        assert episode["control_valid"][-1] == 0.0
        assert episode["rewards"][-1] == 0.0
        assert episode["dones"][-1] == 1.0

    def test_optimizer_parameters_deduplicate_shared_tensors(self):
        from fragile.learning.rl.train_dreamer import _optimizer_parameters

        param = nn.Parameter(torch.randn(3))
        optimizer = SimpleNamespace(param_groups=[{"params": [param, param]}])
        params = _optimizer_parameters(optimizer)
        assert params == [param]

    def test_rollout_routing_tau_is_deterministic_for_hard_routing(self):
        from fragile.learning.rl.train_dreamer import _rollout_routing_tau

        assert _rollout_routing_tau(True, 0.7) == -1.0
        assert _rollout_routing_tau(False, 0.7) == 0.7


class TestObservationNormalizer:
    def test_round_trip_tensor(self, device):
        from fragile.learning.rl.train_dreamer import ObservationNormalizer

        normalizer = ObservationNormalizer(
            mean=torch.arange(OBS_DIM, device=device, dtype=torch.float32),
            std=torch.full((OBS_DIM,), 2.0, device=device),
        )
        obs = torch.randn(3, OBS_DIM, device=device)
        denorm = normalizer.denormalize_tensor(normalizer.normalize_tensor(obs))
        torch.testing.assert_close(denorm, obs)

    def test_from_episodes_clamps_small_std(self, device):
        from fragile.learning.rl.train_dreamer import ObservationNormalizer

        episodes = [
            {"obs": np.ones((3, OBS_DIM), dtype=np.float32)},
            {"obs": np.ones((2, OBS_DIM), dtype=np.float32)},
        ]
        normalizer = ObservationNormalizer.from_episodes(episodes, device=device, min_std=0.25)
        assert float(normalizer.std.min()) == pytest.approx(0.25)


class TestConfigAndParseArgs:
    def test_config_canonicalizes_action_atlas_sizes(self):
        from fragile.learning.rl.config import DreamerConfig

        cfg = DreamerConfig(num_charts=5, num_action_charts=0, action_codes_per_chart=0, codes_per_chart=7)
        assert cfg.num_action_charts == 5
        assert cfg.num_action_macros == 5
        assert cfg.action_codes_per_chart == 7

    def test_parse_args_uses_current_cli_schema(self, monkeypatch):
        from fragile.learning.rl.train_dreamer import _parse_args

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--action_dim",
                "3",
                "--latent_dim",
                "5",
                "--num_charts",
                "6",
                "--no-use_gas",
            ],
        )
        cfg = _parse_args()
        assert cfg.action_dim == 3
        assert cfg.latent_dim == 5
        assert cfg.num_charts == 6
        assert not cfg.use_gas


class TestAtlasSync:
    def test_sync_rl_atlas_binds_obs_and_action_atlases(
        self,
        obs_model,
        action_model,
        world_model,
        standalone_critic,
        actor,
        closure_model,
        reward_head,
    ):
        from fragile.learning.core.layers.atlas import _project_to_ball
        from fragile.learning.rl.train_dreamer import _sync_rl_atlas

        with torch.no_grad():
            obs_model.encoder.chart_centers.copy_(torch.randn_like(obs_model.encoder.chart_centers) * 0.4)
            action_model.encoder.chart_centers.copy_(
                torch.randn_like(action_model.encoder.chart_centers) * 0.4
            )
            world_model.potential_net.chart_tok.chart_centers.zero_()
            standalone_critic.chart_tok.chart_centers.zero_()
            actor.action_chart_centers.zero_()
            closure_model.obs_chart_centers.zero_()
            reward_head.action_chart_tok.chart_centers.zero_()

        _sync_rl_atlas(
            obs_model,
            action_model,
            world_model,
            standalone_critic,
            actor,
            closure_model,
            reward_head,
        )

        expected_obs = _project_to_ball(obs_model.encoder.chart_centers.detach())
        expected_action = _project_to_ball(action_model.encoder.chart_centers.detach())
        torch.testing.assert_close(
            _project_to_ball(world_model.potential_net.chart_tok.chart_centers.detach()),
            expected_obs,
        )
        torch.testing.assert_close(
            _project_to_ball(standalone_critic.chart_tok.chart_centers.detach()),
            expected_obs,
        )
        torch.testing.assert_close(
            _project_to_ball(actor.action_chart_centers.detach()),
            expected_action,
        )
        torch.testing.assert_close(
            _project_to_ball(closure_model.obs_chart_centers.detach()),
            expected_obs,
        )
        torch.testing.assert_close(
            _project_to_ball(reward_head.action_chart_tok.chart_centers.detach()),
            expected_action,
        )
        assert not actor.action_chart_centers.requires_grad
        assert not closure_model.obs_chart_centers.requires_grad
        assert not reward_head.action_chart_tok.chart_centers.requires_grad


class TestRolloutCollection:
    def test_collect_episode_without_policy_marks_controls_invalid(self):
        from fragile.learning.rl.train_dreamer import _collect_episode

        env = SingleEpisodeEnv(start_obs=[0.1] * OBS_DIM, next_obs=[0.2] * OBS_DIM)
        episode = _collect_episode(
            env,
            None,
            None,
            None,
            SimpleNamespace(),
            torch.device("cpu"),
            control_dim=D,
            num_action_charts=K,
            action_repeat=1,
            max_steps=1,
        )
        assert episode["obs"].shape == (2, OBS_DIM)
        assert episode["controls_cov"].shape == (2, D)
        assert episode["action_latents"].shape == (2, D)
        assert episode["action_router_weights"].shape == (2, K)
        assert episode["control_valid"][0] == pytest.approx(0.0)
        assert episode["control_valid"][-1] == pytest.approx(0.0)
        assert np.allclose(episode["controls_cov"], 0.0)

    def test_collect_episode_normalizes_obs_and_uses_current_schema(self, monkeypatch):
        from fragile.learning.rl import train_dreamer

        env = SingleEpisodeEnv(start_obs=[3.0] * OBS_DIM, next_obs=[4.0] * OBS_DIM)
        model = FakeEncoderModel()
        normalizer = train_dreamer.ObservationNormalizer(
            mean=torch.ones(OBS_DIM),
            std=torch.full((OBS_DIM,), 2.0),
            min_std=1e-3,
        )

        def fake_policy_action(
            _actor,
            _action_model,
            _closure_model,
            obs_info,
            **_kwargs,
        ):
            return _make_constant_policy_output(obs_info["z_geo"])

        monkeypatch.setattr(train_dreamer, "_policy_action", fake_policy_action)
        episode = train_dreamer._collect_episode(
            env,
            object(),
            object(),
            object(),
            model,
            torch.device("cpu"),
            control_dim=D,
            num_action_charts=K,
            obs_normalizer=normalizer,
            action_repeat=1,
            max_steps=1,
            hard_routing=True,
            hard_routing_tau=0.7,
            use_motor_texture=True,
        )

        assert model.encoder.calls == [(True, -1.0)]
        torch.testing.assert_close(model.encoder.obs_seen[0], torch.ones(1, OBS_DIM))
        assert episode["controls_tan"].shape == (2, D)
        assert episode["controls_cov"].shape == (2, D)
        assert episode["action_latents"].shape == (2, D)
        assert episode["action_router_weights"].shape == (2, K)
        assert episode["action_charts"][0] == 0
        assert episode["action_codes"][0] == 1
        assert episode["control_valid"][0] == pytest.approx(1.0)

    def test_collect_parallel_episodes_returns_replay_compatible_schema(self, monkeypatch):
        from fragile.learning.rl import train_dreamer

        env = ParallelEnv()
        model = FakeEncoderModel()

        def fake_policy_action(
            _actor,
            _action_model,
            _closure_model,
            obs_info,
            **_kwargs,
        ):
            return _make_constant_policy_output(obs_info["z_geo"])

        monkeypatch.setattr(train_dreamer, "_policy_action", fake_policy_action)
        episodes = train_dreamer._collect_parallel_episodes(
            env,
            object(),
            object(),
            object(),
            model,
            torch.device("cpu"),
            control_dim=D,
            num_action_charts=K,
            num_episodes=2,
            action_repeat=1,
            max_steps=2,
            hard_routing=True,
            hard_routing_tau=0.7,
            use_motor_texture=True,
        )

        assert len(episodes) == 2
        assert episodes[0]["obs"].shape == (2, OBS_DIM)
        assert episodes[1]["obs"].shape == (3, OBS_DIM)
        assert episodes[0]["controls_cov"].shape == (2, D)
        assert episodes[1]["action_router_weights"].shape == (3, K)
        assert episodes[0]["action_latents"].shape == (2, D)
        assert episodes[1]["action_code_latents"].shape == (3, D)
        assert episodes[1]["dones"][-1] == pytest.approx(1.0)
        assert env.batch_calls == [[0, 1], [1]]

    def test_eval_policy_uses_deterministic_mean_actions(self, monkeypatch):
        from fragile.learning.rl import train_dreamer

        env = SingleEpisodeEnv(start_obs=[0.1] * OBS_DIM, next_obs=[0.2] * OBS_DIM)
        model = FakeEncoderModel()
        calls: list[bool] = []

        def fake_policy_action(
            _actor,
            _action_model,
            _closure_model,
            obs_info,
            *,
            use_motor_texture,
            **_kwargs,
        ):
            calls.append(use_motor_texture)
            return _make_constant_policy_output(obs_info["z_geo"], action_value=0.44)

        monkeypatch.setattr(train_dreamer, "_policy_action", fake_policy_action)
        metrics = train_dreamer._eval_policy(
            env,
            object(),
            object(),
            object(),
            model,
            torch.device("cpu"),
            action_repeat=1,
            num_episodes=1,
            max_steps=1,
            hard_routing=True,
            hard_routing_tau=0.7,
        )

        assert calls == [False]
        assert model.encoder.calls == [(True, -1.0)]
        assert metrics["eval/reward_mean"] == pytest.approx(1.0)


class TestGasCollection:
    def test_collect_gas_episodes_clones_current_schema_prefix(self, monkeypatch, config):
        from fragile.learning.rl import train_dreamer

        gas_config = config
        gas_config.use_gas = True
        gas_config.gas_walkers = 2
        gas_config.gas_steps = 2
        gas_config.obs_dim = 1
        gas_config.action_dim = 1
        gas_config.latent_dim = D
        gas_config.num_charts = K
        gas_config.num_action_charts = K
        gas_config.hard_routing = True
        gas_config.hard_routing_tau = 0.5

        class FakeEnv:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.action_space = FakeActionSpec(action_dim=1)

        class FakeKineticOp:
            def __init__(self):
                self.last_actions = np.zeros((2, 1), dtype=np.float64)

        class FakeState:
            def __init__(self, observations, rewards, step_rewards, dones):
                self.observations = torch.tensor(observations, dtype=torch.float32)
                self.rewards = torch.tensor(rewards, dtype=torch.float32)
                self.step_rewards = torch.tensor(step_rewards, dtype=torch.float32)
                self.dones = torch.tensor(dones, dtype=torch.bool)

        class FakeGas:
            def __init__(self, env, N, **kwargs):
                del kwargs
                self.env = env
                self.N = N
                self.kinetic_op = FakeKineticOp()
                self.total_clones = 0
                self._step = 0

            def reset(self):
                self._step = 0
                return FakeState(
                    observations=[[0.10], [0.90]],
                    rewards=[0.0, 0.0],
                    step_rewards=[0.0, 0.0],
                    dones=[False, False],
                )

            def step(self, state, actions=None):
                del state
                if self._step == 0:
                    companions = np.array([0, 1], dtype=np.int64)
                    will_clone = np.array([False, False], dtype=bool)
                    self.kinetic_op.last_actions = actions.copy()
                    next_state = FakeState(
                        observations=[[0.20], [0.80]],
                        rewards=[1.0, 2.0],
                        step_rewards=[1.0, 2.0],
                        dones=[False, False],
                    )
                else:
                    companions = np.array([0, 0], dtype=np.int64)
                    will_clone = np.array([False, True], dtype=bool)
                    cloned_actions = actions.copy()
                    cloned_actions[will_clone] = cloned_actions[companions[will_clone]]
                    self.kinetic_op.last_actions = cloned_actions
                    self.total_clones += int(will_clone.sum())
                    next_state = FakeState(
                        observations=[[0.30], [0.30]],
                        rewards=[4.0, 4.0],
                        step_rewards=[3.0, 3.0],
                        dones=[False, False],
                    )
                self._step += 1
                info = {
                    "clone_companions": torch.tensor(companions, dtype=torch.long),
                    "will_clone": torch.tensor(will_clone, dtype=torch.bool),
                }
                return next_state, info

        def fake_policy_action(
            _actor,
            _action_model,
            _closure_model,
            obs_info,
            **_kwargs,
        ):
            out = _make_constant_policy_output(obs_info["z_geo"], action_value=0.3)
            out["action"] = out["action"][:, :1]
            out["action_mean"] = out["action_mean"][:, :1]
            return out

        monkeypatch.setattr(train_dreamer, "VectorizedDMControlEnv", FakeEnv)
        monkeypatch.setattr(train_dreamer, "RoboticFractalGas", FakeGas)
        monkeypatch.setattr(train_dreamer, "_policy_action", fake_policy_action)

        model = FakeEncoderModel(latent_dim=D, num_charts=K)
        episodes, metrics = train_dreamer._collect_gas_episodes(
            object(),
            object(),
            object(),
            model,
            torch.device("cpu"),
            gas_config,
        )

        assert model.encoder.calls == [(True, -1.0), (True, -1.0)]
        np.testing.assert_allclose(episodes[1]["obs"][:2], episodes[0]["obs"][:2])
        np.testing.assert_allclose(episodes[1]["actions"][:1], episodes[0]["actions"][:1])
        np.testing.assert_allclose(episodes[1]["controls_cov"][:1], episodes[0]["controls_cov"][:1])
        np.testing.assert_allclose(
            episodes[1]["action_latents"][:1],
            episodes[0]["action_latents"][:1],
        )
        np.testing.assert_allclose(
            episodes[1]["action_router_weights"][:1],
            episodes[0]["action_router_weights"][:1],
        )
        np.testing.assert_allclose(episodes[1]["rewards"][:1], episodes[0]["rewards"][:1])
        assert metrics["gas/total_clones"] == pytest.approx(1.0)


class TestImagination:
    def test_outputs_current_shapes_and_exact_split(
        self,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        closure_model,
        reward_head,
        z,
        rw,
    ):
        from fragile.learning.rl.train_dreamer import _imagine, _sync_rl_atlas

        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, closure_model, reward_head)
        out = _imagine(
            obs_model,
            world_model,
            reward_head,
            critic,
            actor,
            action_model,
            closure_model,
            z,
            rw,
            horizon=H_IMAGINATION,
            gamma=0.99,
            reward_curl_batch_limit=2,
        )
        assert out["z_states"].shape == (B, H_IMAGINATION, D)
        assert out["rw_states"].shape == (B, H_IMAGINATION, K)
        assert out["z_traj"].shape == (B, H_IMAGINATION, D)
        assert out["rw_traj"].shape == (B, H_IMAGINATION, K)
        assert out["controls_tan"].shape == (B, H_IMAGINATION, D)
        assert out["controls_cov"].shape == (B, H_IMAGINATION, D)
        assert out["action_latents"].shape == (B, H_IMAGINATION, D)
        assert out["action_router_weights"].shape == (B, H_IMAGINATION, K)
        assert out["actions"].shape == (B, H_IMAGINATION, A)
        assert out["reward_conservative"].shape == (B, H_IMAGINATION)
        assert out["reward_nonconservative"].shape == (B, H_IMAGINATION)
        assert out["reward_curl_norm"].shape == (B, H_IMAGINATION)
        assert out["reward_curl_valid"].shape == (B, H_IMAGINATION)
        assert out["phi_eff"].shape == (B, H_IMAGINATION, 1)
        assert not out["actions"].requires_grad
        assert not out["z_traj"].requires_grad
        torch.testing.assert_close(
            out["rewards"],
            out["reward_conservative"] + out["reward_nonconservative"],
        )

    def test_no_jump_world_model_keeps_router_weights(
        self,
        monkeypatch,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        closure_model,
        reward_head,
        z,
        rw,
    ):
        from fragile.learning.rl.train_dreamer import _imagine, _sync_rl_atlas

        def bad_chart_logits(self, z_in, control_in, rw_in):
            del z_in, control_in, rw_in
            logits = torch.full((B, K), -50.0)
            logits[:, -1] = 50.0
            return logits

        monkeypatch.setattr(
            world_model.chart_predictor,
            "forward",
            bad_chart_logits.__get__(world_model.chart_predictor, type(world_model.chart_predictor)),
        )
        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, closure_model, reward_head)
        out = _imagine(
            obs_model,
            world_model,
            reward_head,
            critic,
            actor,
            action_model,
            closure_model,
            z,
            rw,
            horizon=H_IMAGINATION,
            gamma=0.99,
        )
        expected = rw.unsqueeze(1).expand_as(out["rw_traj"])
        torch.testing.assert_close(out["rw_traj"], expected)

    def test_imagination_uses_world_model_chart_state_for_actor_inputs(
        self,
        monkeypatch,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        closure_model,
        reward_head,
        z,
    ):
        from fragile.learning.rl.action_manifold import symbolize_latent_with_atlas
        from fragile.learning.rl.train_dreamer import _imagine, _sync_rl_atlas

        initial_info = symbolize_latent_with_atlas(
            obs_model,
            z,
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        forced_chart_idx = (initial_info["chart_idx"] + 1) % K
        forced_rw = torch.nn.functional.one_hot(
            forced_chart_idx,
            num_classes=K,
        ).to(dtype=z.dtype, device=z.device)

        seen_chart_idx: list[torch.Tensor] = []
        original_forward = actor.forward

        def recording_forward(obs_chart_idx, obs_code_idx, obs_z_n, **kwargs):
            seen_chart_idx.append(obs_chart_idx.detach().clone())
            return original_forward(obs_chart_idx, obs_code_idx, obs_z_n, **kwargs)

        monkeypatch.setattr(actor, "forward", recording_forward)
        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, closure_model, reward_head)
        _imagine(
            obs_model,
            world_model,
            reward_head,
            critic,
            actor,
            action_model,
            closure_model,
            z,
            forced_rw,
            horizon=H_IMAGINATION,
            gamma=0.99,
        )

        assert seen_chart_idx
        for chart_idx in seen_chart_idx:
            torch.testing.assert_close(chart_idx, forced_chart_idx)

    def test_actor_return_optimizes_only_nonconservative_work(self, monkeypatch):
        from fragile.learning.rl import train_dreamer

        class FakeActor(nn.Module):
            def forward(
                self,
                obs_chart_idx,
                obs_code_idx,
                obs_z_n,
                *,
                hard_routing=False,
                hard_routing_tau=1.0,
            ):
                del obs_chart_idx, obs_code_idx, hard_routing, hard_routing_tau
                batch = obs_z_n.shape[0]
                chart_logits = torch.zeros(batch, K, device=obs_z_n.device)
                chart_logits[:, 0] = 1.0
                code_logits = torch.zeros(batch, K, CODES_PER_CHART, device=obs_z_n.device)
                code_logits[:, :, 0] = 1.0
                return {
                    "action_chart_logits": chart_logits,
                    "action_chart_idx": torch.zeros(batch, dtype=torch.long, device=obs_z_n.device),
                    "action_code_logits": code_logits,
                    "action_code_idx": torch.zeros(batch, dtype=torch.long, device=obs_z_n.device),
                    "action_z_n": torch.full_like(obs_z_n, 0.2),
                    "action_z_q": torch.zeros_like(obs_z_n),
                    "action_z_geo": torch.full_like(obs_z_n, 0.2),
                    "action_router_weights": torch.nn.functional.one_hot(
                        torch.zeros(batch, dtype=torch.long, device=obs_z_n.device),
                        num_classes=K,
                    ).to(dtype=obs_z_n.dtype),
                }

        class FakeActionModel(nn.Module):
            def decoder(
                self,
                z_geo,
                _unused,
                *,
                router_weights,
                hard_routing,
                hard_routing_tau,
            ):
                del router_weights, hard_routing, hard_routing_tau
                return z_geo[:, :A], None, None

        class FakeClosure(nn.Module):
            def forward(
                self,
                obs_chart_idx,
                obs_code_idx,
                obs_z_n,
                action_chart_idx,
                action_code_idx,
                action_z_n,
            ):
                del obs_chart_idx, obs_code_idx, action_chart_idx, action_code_idx
                return {
                    "control_tan": obs_z_n + action_z_n,
                    "control_cov": obs_z_n + action_z_n,
                }

        class FakeWorldModel:
            def momentum_init(self, z_0):
                return torch.zeros_like(z_0)

            def _rollout_transition(self, z, p, control_cov, rw, track_energy=False):
                del control_cov, track_energy
                return {"z": z + 0.01, "p": p, "rw": rw, "phi_eff": torch.zeros(z.shape[0], 1)}

        class FakeRewardHead:
            def decompose(
                self,
                z,
                rw,
                action_z,
                action_rw,
                action_code_z,
                control,
                *,
                exact_covector=None,
            ):
                del z, rw, action_z, action_rw, action_code_z, control, exact_covector
                return {"reward_nonconservative": torch.full((2, 1), 3.0)}

        class FakeCritic(nn.Module):
            def task_value(self, z, rw):
                del rw
                return z[:, :1]

        def fake_symbolize_latent_with_atlas(_atlas, z_in, **_kwargs):
            batch = z_in.shape[0]
            router = torch.zeros(batch, K, device=z_in.device, dtype=z_in.dtype)
            router[:, 0] = 1.0
            return {
                "z_geo": z_in,
                "router_weights": router,
                "chart_idx": torch.zeros(batch, dtype=torch.long, device=z_in.device),
                "code_idx": torch.zeros(batch, dtype=torch.long, device=z_in.device),
                "z_q": torch.zeros_like(z_in),
                "z_n": torch.zeros_like(z_in),
            }

        monkeypatch.setattr(train_dreamer, "symbolize_latent_with_atlas", fake_symbolize_latent_with_atlas)

        def zero_covector(_critic, z_in, _rw, **_kwargs):
            return torch.zeros_like(z_in)

        monkeypatch.setattr(train_dreamer, "_value_covector_from_critic", zero_covector)

        z_0 = torch.randn(2, D) * 0.1
        rw_0 = torch.softmax(torch.randn(2, K), dim=-1)
        out = train_dreamer._imagine_actor_return(
            object(),
            FakeWorldModel(),
            FakeRewardHead(),
            FakeCritic(),
            FakeActor(),
            FakeActionModel(),
            FakeClosure(),
            z_0,
            rw_0,
            horizon=3,
            gamma=0.5,
        )

        expected_objective = torch.full((2,), 3.0 * (1.0 + 0.5 + 0.25))
        torch.testing.assert_close(out["objective"], expected_objective)
        assert out["reward_nonconservative"].shape == (2, 3)
        assert out["actions"].shape == (2, 3, A)
        assert out["controls_tan"].shape == (2, 3, D)


class TestTrainStep:
    def test_updates_parameters_and_reports_reward_split_metrics(
        self,
        device,
        config,
        phase1_cfg,
        action_phase1_cfg,
        obs_model,
        jump_op,
        action_model,
        action_jump_op,
        closure_model,
        world_model,
        critic,
        reward_head,
        actor,
    ):
        from fragile.learning.rl.train_dreamer import _train_step

        batch = _make_training_batch(device)
        optimizer_enc, optimizer_wm, optimizer_boundary = _make_optimizers(
            obs_model,
            jump_op,
            action_model,
            action_jump_op,
            closure_model,
            world_model,
            reward_head,
            actor,
        )
        actor_before = {name: param.detach().clone() for name, param in actor.named_parameters()}
        reward_before = {
            name: param.detach().clone()
            for name, param in reward_head.named_parameters()
            if "chart_tok" not in name and "z_embed" not in name
        }

        metrics = _train_step(
            obs_model,
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
            None,
            optimizer_boundary,
            batch,
            config,
            phase1_cfg,
            action_phase1_cfg,
            epoch=0,
            current_hard_routing=False,
            current_tau=1.0,
            update_idx=0,
            compute_diagnostics=False,
        )

        expected_keys = {
            "enc/L_total",
            "wm/L_reward",
            "wm/L_reward_nonconservative",
            "wm/L_reward_exact_orth",
            "wm/reward_form_exact_leakage_mean",
            "critic/L_critic",
            "actor/L_total",
            "actor/L_return",
            "actor/update_applied",
            "time/step",
        }
        assert expected_keys.issubset(metrics)
        assert metrics["actor/update_applied"] == 1.0
        assert np.isfinite([metrics[key] for key in expected_keys]).all()

        actor_changed = any(
            not torch.equal(actor_before[name], param.detach())
            for name, param in actor.named_parameters()
        )
        reward_changed = any(
            not torch.equal(reward_before[name], param.detach())
            for name, param in reward_head.named_parameters()
            if name in reward_before
        )
        assert actor_changed
        assert reward_changed

    def test_frozen_encoder_keeps_encoder_weights_fixed(
        self,
        device,
        config,
        phase1_cfg,
        action_phase1_cfg,
        obs_model,
        jump_op,
        action_model,
        action_jump_op,
        closure_model,
        world_model,
        critic,
        reward_head,
        actor,
    ):
        from fragile.learning.rl.train_dreamer import _train_step

        config.freeze_encoder = True
        batch = _make_training_batch(device)
        optimizer_enc, optimizer_wm, optimizer_boundary = _make_optimizers(
            obs_model,
            jump_op,
            action_model,
            action_jump_op,
            closure_model,
            world_model,
            reward_head,
            actor,
        )
        encoder_before = {
            name: param.detach().clone() for name, param in obs_model.encoder.named_parameters()
        }

        metrics = _train_step(
            obs_model,
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
            None,
            optimizer_boundary,
            batch,
            config,
            phase1_cfg,
            action_phase1_cfg,
            epoch=0,
            current_hard_routing=False,
            current_tau=1.0,
            update_idx=0,
            compute_diagnostics=False,
        )

        assert "enc/L_total" in metrics
        for name, param in obs_model.encoder.named_parameters():
            torch.testing.assert_close(param.detach(), encoder_before[name])


class TestCheckpoint:
    def test_save_checkpoint_stores_current_modules_and_normalizer(
        self,
        tmp_path,
        config,
        obs_model,
        jump_op,
        action_model,
        action_jump_op,
        closure_model,
        world_model,
        critic,
        reward_head,
        actor,
    ):
        from fragile.learning.rl.train_dreamer import _save_checkpoint, ObservationNormalizer

        optimizer_enc, optimizer_wm, optimizer_boundary = _make_optimizers(
            obs_model,
            jump_op,
            action_model,
            action_jump_op,
            closure_model,
            world_model,
            reward_head,
            actor,
        )
        scheduler_enc = torch.optim.lr_scheduler.LambdaLR(optimizer_enc, lr_lambda=lambda _: 1.0)
        scheduler_wm = torch.optim.lr_scheduler.LambdaLR(optimizer_wm, lr_lambda=lambda _: 1.0)
        scheduler_boundary = torch.optim.lr_scheduler.LambdaLR(
            optimizer_boundary,
            lr_lambda=lambda _: 1.0,
        )
        normalizer = ObservationNormalizer(
            mean=torch.zeros(OBS_DIM),
            std=torch.ones(OBS_DIM),
            min_std=1e-3,
        )
        path = tmp_path / "dreamer.pt"
        _save_checkpoint(
            str(path),
            obs_model,
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
            None,
            scheduler_enc,
            scheduler_wm,
            scheduler_boundary,
            None,
            epoch=3,
            config=config,
            metrics={"wm/L_reward": 1.0},
            obs_normalizer=normalizer,
        )

        checkpoint = torch.load(path, map_location="cpu")
        assert checkpoint["epoch"] == 3
        assert "action_model" in checkpoint
        assert "closure_model" in checkpoint
        assert "world_model" in checkpoint
        assert "reward_head" in checkpoint
        assert "optimizer_wm" in checkpoint
        assert "scheduler_boundary" in checkpoint
        assert checkpoint["metrics"]["wm/L_reward"] == 1.0
        assert checkpoint["obs_normalizer"]["min_std"] == pytest.approx(1e-3)
