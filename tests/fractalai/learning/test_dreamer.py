"""Tests for the Geometric Dreamer RL module."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch
from torch import nn

# Small dims for fast CPU tests
B, D, A, K = 4, 8, 6, 4
D_MODEL = 32
H_IMAGINATION = 3
H_WM = 2  # world-model prediction horizon


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def z(device):
    """Random position inside the Poincare ball (norm < 1)."""
    return torch.randn(B, D, device=device) * 0.3


@pytest.fixture
def rw(device):
    """Soft chart routing weights summing to 1."""
    return torch.softmax(torch.randn(B, K, device=device), dim=-1)


@pytest.fixture
def action(device):
    return torch.randn(B, A, device=device)


@pytest.fixture
def actor():
    from fragile.learning.rl.actor import GeometricActor

    return GeometricActor(D, A, K, D_MODEL)


@pytest.fixture
def critic():
    from fragile.learning.rl.critic import GeometricCritic

    return GeometricCritic(D, K, D_MODEL)


@pytest.fixture
def action_encoder():
    from fragile.learning.rl.boundary import GeometricActionEncoder

    return GeometricActionEncoder(D, A, K, d_model=D_MODEL)


@pytest.fixture
def action_decoder():
    from fragile.learning.rl.boundary import GeometricActionBoundaryDecoder

    return GeometricActionBoundaryDecoder(D, A, K, d_model=D_MODEL)


@pytest.fixture
def world_model():
    from fragile.learning.vla.covariant_world_model import GeometricWorldModel

    return GeometricWorldModel(
        latent_dim=D, action_dim=A, control_dim=D, num_charts=K, d_model=D_MODEL,
        hidden_dim=D_MODEL, n_refine_steps=2, use_boris=False, use_jump=False,
    )


@pytest.fixture
def world_model_full():
    """World model with Boris rotation and chart jumps enabled."""
    from fragile.learning.vla.covariant_world_model import GeometricWorldModel

    return GeometricWorldModel(
        latent_dim=D, action_dim=A, control_dim=D, num_charts=K, d_model=D_MODEL,
        hidden_dim=D_MODEL, n_refine_steps=2, use_boris=True, use_jump=True,
    )


@pytest.fixture
def reward_head(world_model):
    from fragile.learning.rl.reward_head import RewardHead

    return RewardHead(world_model.potential_net, A, D_MODEL)


@pytest.fixture
def encoder():
    from fragile.learning.core.layers.atlas import PrimitiveAttentiveAtlasEncoder

    return PrimitiveAttentiveAtlasEncoder(
        input_dim=24, hidden_dim=D_MODEL, latent_dim=D,
        num_charts=K, codes_per_chart=4,
    )


@pytest.fixture
def topo_model():
    from fragile.learning.vla.shared_dyn.encoder import SharedDynTopoEncoder

    return SharedDynTopoEncoder(
        input_dim=24, hidden_dim=D_MODEL, latent_dim=D,
        num_charts=K, codes_per_chart=4,
    )


@pytest.fixture
def jump_op():
    from fragile.learning.core.layers import FactorizedJumpOperator

    return FactorizedJumpOperator(num_charts=K, latent_dim=D)


@pytest.fixture
def dyn_trans_model():
    from fragile.learning.vla.losses import DynamicsTransitionModel

    return DynamicsTransitionModel(
        chart_dim=D,
        action_dim=A,
        num_charts=K,
        codes_per_chart=4,
        hidden_dim=D_MODEL,
    )


@pytest.fixture
def phase1_cfg():
    from fragile.learning.vla.config import VLAConfig

    return VLAConfig(
        input_dim=24, hidden_dim=D_MODEL, latent_dim=D,
        num_charts=K, codes_per_chart=4,
    )


@pytest.fixture
def dreamer_config(tmp_path):
    from fragile.learning.rl.config import DreamerConfig

    return DreamerConfig(
        device="cpu", obs_dim=24, action_dim=A,
        latent_dim=D, num_charts=K, d_model=D_MODEL, hidden_dim=D_MODEL,
        codes_per_chart=4,
        wm_prediction_horizon=H_WM, imagination_horizon=H_IMAGINATION,
        batch_size=B, seq_len=8,
        wm_n_refine_steps=2, wm_use_boris=False, wm_use_jump=False,
        gamma=0.99, lambda_gae=0.95, T_c_entropy=0.1,
        grad_clip=100.0,
        actor_return_horizon=2,
        actor_return_batch_size=2,
        actor_return_update_every=10,
        actor_return_warmup_epochs=10,
        freeze_encoder=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )


# ---------------------------------------------------------------------------
# GeometricActor
# ---------------------------------------------------------------------------


class TestGeometricActor:
    def test_forward_shapes(self, actor, z, rw):
        mu, log_std = actor(z, rw)
        assert mu.shape == (B, A)
        assert log_std.shape == (B, A)

    def test_log_std_clamped(self, actor, z, rw):
        _, log_std = actor(z, rw)
        assert (log_std >= actor.LOG_STD_MIN).all()
        assert (log_std <= actor.LOG_STD_MAX).all()

    def test_sample_shapes(self, actor, z, rw):
        action, log_prob = actor.sample(z, rw)
        assert action.shape == (B, A)
        assert log_prob.shape == (B, 1)

    def test_sample_in_range(self, actor, z, rw):
        """Squashed actions must be in (-1, 1)."""
        action, _ = actor.sample(z, rw)
        assert (action > -1.0).all()
        assert (action < 1.0).all()

    def test_sample_log_prob_finite(self, actor, z, rw):
        _, log_prob = actor.sample(z, rw)
        assert torch.isfinite(log_prob).all()

    def test_mode_shapes(self, actor, z, rw):
        action = actor.mode(z, rw)
        assert action.shape == (B, A)
        assert (action > -1.0).all()
        assert (action < 1.0).all()

    def test_mode_deterministic(self, actor, z, rw):
        """Two calls to mode() with same input should match (within SpectralLinear tolerance)."""
        a1 = actor.mode(z, rw)
        a2 = actor.mode(z, rw)
        assert torch.allclose(a1, a2, atol=1e-5)

    def test_gradient_flows_through_actor(self, actor, z, rw):
        """Gradients from sample must reach actor parameters."""
        action, log_prob = actor.sample(z, rw)
        loss = log_prob.mean()
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in actor.parameters() if p.grad is not None
        ]
        assert len(grad_norms) > 0, "No gradients reached actor params"
        assert any(g > 0 for g in grad_norms), "All gradients are zero"

    def test_no_action_tokenizer(self, actor):
        """Actor must NOT have an ActionTokenizer (it produces, not consumes)."""
        assert not hasattr(actor, "action_tok")

    def test_has_chart_tokenizer(self, actor):
        """Actor must have a ChartTokenizer like CovariantControlField."""
        assert hasattr(actor, "chart_tok")
        assert hasattr(actor, "attn")
        assert hasattr(actor, "z_embed")


# ---------------------------------------------------------------------------
# GeometricCritic
# ---------------------------------------------------------------------------


class TestGeometricCritic:
    def test_forward_shape(self, critic, z, rw):
        value = critic(z, rw)
        assert value.shape == (B, 1)

    def test_output_finite(self, critic, z, rw):
        value = critic(z, rw)
        assert torch.isfinite(value).all()

    def test_gradient_flows(self, critic, z, rw):
        z_g = z.clone().requires_grad_(True)
        value = critic(z_g, rw)
        value.sum().backward()
        assert z_g.grad is not None
        assert torch.isfinite(z_g.grad).all()


class TestBoundaryControlGeometry:
    def test_raise_lower_roundtrip(self, z):
        from fragile.learning.rl.boundary import lower_control, raise_control

        control_tan = torch.randn_like(z)
        control_cov = lower_control(z, control_tan)
        control_tan_rt = raise_control(z, control_cov)

        torch.testing.assert_close(control_tan_rt, control_tan, atol=1e-5, rtol=1e-5)

    def test_critic_control_field_returns_covector_and_tangent(self, critic, z, rw):
        from fragile.learning.rl.boundary import critic_control_field, lower_control

        control_cov, control_tan, value = critic_control_field(critic, z, rw)

        assert control_cov.shape == z.shape
        assert control_tan.shape == z.shape
        assert value.shape == (B, 1)
        torch.testing.assert_close(lower_control(z, control_tan), control_cov, atol=1e-5, rtol=1e-5)

    def test_motor_texture_std_shrinks_near_boundary(self, action_decoder, rw):
        z_center = torch.zeros(B, D)
        z_boundary = torch.zeros(B, D)
        z_boundary[:, 0] = 0.95

        std_center = action_decoder.motor_texture_std(z_center)
        std_boundary = action_decoder.motor_texture_std(z_boundary)

        assert torch.isfinite(std_center).all()
        assert torch.isfinite(std_boundary).all()
        assert (std_boundary < std_center).all()

    def test_execution_texture_is_added_before_squashing(self, action_decoder, z, rw):
        control = torch.randn_like(z)
        with torch.random.fork_rng():
            torch.manual_seed(0)
            out = action_decoder.sample_execution_action(z, control, rw)

        assert out["action"].shape == out["action_mean"].shape == (B, A)
        assert out["log_std"].shape == (B, A)
        assert torch.isfinite(out["action"]).all()
        assert torch.isfinite(out["action_mean"]).all()
        assert torch.isfinite(out["log_std"]).all()
        assert (out["action"].abs() <= 1.0).all()

    def test_action_encoder_outputs_boundary_decomposition(self, action_encoder, z, action, rw):
        out = action_encoder(z, action, rw)
        assert out["control_tan"].shape == (B, D)
        assert out["macro_probs"].shape == (B, K)
        assert out["macro_idx"].shape == (B,)
        assert out["motor_nuisance"].shape == (B, D)
        assert out["motor_compliance"].shape == (B, A, A)

    def test_action_decoder_outputs_boundary_decomposition(self, action_decoder, z, rw):
        control = torch.randn_like(z)
        out = action_decoder(z, control, rw)
        assert out["macro_probs"].shape == (B, K)
        assert out["motor_nuisance"].shape == (B, D)
        assert out["motor_compliance"].shape == (B, A, A)
        assert out["action_mean"].shape == (B, A)


# ---------------------------------------------------------------------------
# RewardHead
# ---------------------------------------------------------------------------


class TestRewardHead:
    def test_output_shape(self, reward_head, z, action, rw):
        r = reward_head(z, action, rw, control=torch.randn_like(z))
        assert r.shape == (B, 1)

    def test_output_shape_with_control(self, reward_head, z, action, rw):
        control = torch.randn_like(z)
        r = reward_head(z, action, rw, control=control)
        assert r.shape == (B, 1)

    def test_output_finite(self, reward_head, z, action, rw):
        r = reward_head(z, action, rw, control=torch.randn_like(z))
        assert torch.isfinite(r).all()

    def test_weight_sharing_chart_tok(self, reward_head, world_model):
        """chart_tok must be the SAME object as potential_net.chart_tok."""
        assert reward_head.chart_tok is world_model.potential_net.chart_tok

    def test_weight_sharing_z_embed(self, reward_head, world_model):
        """z_embed must be the SAME object as potential_net.z_embed."""
        assert reward_head.z_embed is world_model.potential_net.z_embed

    def test_gradient_flows(self, reward_head, z, action, rw):
        z_g = z.clone().requires_grad_(True)
        r = reward_head(z_g, action, rw, control=torch.randn_like(z_g))
        r.sum().backward()
        assert z_g.grad is not None
        assert torch.isfinite(z_g.grad).all()

    def test_has_action_tokenizer(self, reward_head):
        """Reward head must have its own ActionTokenizer."""
        assert hasattr(reward_head, "action_tok")

    def test_has_control_tokenizer(self, reward_head):
        """Reward head must encode latent control as part of the boundary state."""
        assert hasattr(reward_head, "control_tok")

    def test_shared_params_not_duplicated(self, reward_head, world_model):
        """Shared parameters should have the same data_ptr."""
        for name, p_rh in reward_head.named_parameters():
            if "chart_tok" in name or "z_embed" in name:
                # Must point to the same storage as in potential_net
                p_pn = dict(world_model.potential_net.named_parameters())[name]
                assert p_rh.data_ptr() == p_pn.data_ptr()


# ---------------------------------------------------------------------------
# compute_lambda_returns
# ---------------------------------------------------------------------------


class TestLambdaReturns:
    def test_shape(self):
        from fragile.learning.rl.returns import compute_lambda_returns

        rewards = torch.randn(B, 5)
        values = torch.randn(B, 5)
        ret = compute_lambda_returns(rewards, values, 0.99, 0.95)
        assert ret.shape == (B, 5)

    def test_zero_rewards_zero_values(self):
        """Zero rewards + zero values => zero returns."""
        from fragile.learning.rl.returns import compute_lambda_returns

        ret = compute_lambda_returns(
            torch.zeros(2, 4), torch.zeros(2, 4), 0.99, 0.95,
        )
        assert torch.allclose(ret, torch.zeros_like(ret))

    def test_gamma_zero(self):
        """gamma=0 => returns = rewards (no bootstrapping)."""
        from fragile.learning.rl.returns import compute_lambda_returns

        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        values = torch.tensor([[10.0, 20.0, 30.0]])
        ret = compute_lambda_returns(rewards, values, gamma=0.0, lambda_gae=0.5)
        assert torch.allclose(ret, rewards)

    def test_lambda_zero(self):
        """lambda=0 => 1-step TD targets: returns[t] = r[t] + gamma * V_boot[t].
        Last step: returns[-1] = r[-1] + gamma * V_boot[-1].
        """
        from fragile.learning.rl.returns import compute_lambda_returns

        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        values = torch.tensor([[10.0, 20.0, 30.0]])
        gamma = 0.9
        ret = compute_lambda_returns(rewards, values, gamma=gamma, lambda_gae=0.0)
        # Last step: r[-1] + gamma * V[-1] = 3 + 0.9*30 = 30.0
        assert abs(ret[0, 2].item() - 30.0) < 1e-5
        # t=1: r[1] + gamma*V[1] = 2 + 0.9*20 = 20.0
        assert abs(ret[0, 1].item() - 20.0) < 1e-5
        # t=0: r[0] + gamma*V[0] = 1 + 0.9*10 = 10.0
        assert abs(ret[0, 0].item() - 10.0) < 1e-5

    def test_lambda_one(self):
        """lambda=1 => Monte Carlo-like returns through recursive formula."""
        from fragile.learning.rl.returns import compute_lambda_returns

        rewards = torch.tensor([[1.0, 1.0, 1.0]])
        values = torch.tensor([[0.0, 0.0, 0.0]])
        gamma = 1.0
        ret = compute_lambda_returns(rewards, values, gamma=gamma, lambda_gae=1.0)
        # lambda=1, gamma=1, values=0:
        # G[2] = 1 + 0 = 1
        # G[1] = 1 + 0*(1-1)*V[2] + 1*1*G[2] = 1 + G[2] = 2
        # G[0] = 1 + 0 + G[1] = 3
        assert abs(ret[0, 0].item() - 3.0) < 1e-5
        assert abs(ret[0, 1].item() - 2.0) < 1e-5
        assert abs(ret[0, 2].item() - 1.0) < 1e-5

    def test_single_step(self):
        """H=1: returns[0] = rewards[0] + gamma * values[0]."""
        from fragile.learning.rl.returns import compute_lambda_returns

        ret = compute_lambda_returns(
            torch.tensor([[5.0]]), torch.tensor([[10.0]]),
            gamma=0.9, lambda_gae=0.95,
        )
        assert abs(ret[0, 0].item() - 14.0) < 1e-5  # 5 + 0.9*10

    def test_device_preserved(self):
        from fragile.learning.rl.returns import compute_lambda_returns

        r = torch.randn(2, 3)
        v = torch.randn(2, 3)
        ret = compute_lambda_returns(r, v)
        assert ret.device == r.device


class TestScreenedPoissonLoss:
    def test_supports_geometric_critic(self, critic):
        from fragile.learning.vla.losses import compute_screened_poisson_loss

        z_traj = torch.randn(2, 3, D) * 0.2
        z_tgt = torch.randn(2, 3, D) * 0.2
        rw_batch = torch.softmax(torch.randn(2, K), dim=-1)

        loss = compute_screened_poisson_loss(critic, z_traj, z_tgt, rw_batch)

        assert loss.ndim == 0
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# SequenceReplayBuffer
# ---------------------------------------------------------------------------


def _make_episode(length: int, obs_dim: int = 24, act_dim: int = 6):
    return {
        "obs": np.random.randn(length, obs_dim).astype(np.float32),
        "actions": np.random.randn(length, act_dim).astype(np.float32),
        "rewards": np.random.randn(length).astype(np.float32),
        "dones": np.zeros(length, dtype=np.float32),
    }


class TestSequenceReplayBuffer:
    def test_add_and_count(self):
        from fragile.learning.rl.replay_buffer import SequenceReplayBuffer

        buf = SequenceReplayBuffer(capacity=10000, seq_len=5)
        ep = _make_episode(20)
        buf.add_episode(ep)
        assert buf.num_episodes == 1
        assert buf.total_steps == 20

    def test_capacity_eviction(self):
        from fragile.learning.rl.replay_buffer import SequenceReplayBuffer

        buf = SequenceReplayBuffer(capacity=50, seq_len=3)
        for _ in range(5):
            buf.add_episode(_make_episode(20))
        # Should have evicted oldest to stay under 50
        assert buf.total_steps <= 50
        assert buf.num_episodes < 5

    def test_sample_shapes(self):
        from fragile.learning.rl.replay_buffer import SequenceReplayBuffer

        buf = SequenceReplayBuffer(capacity=10000, seq_len=5)
        buf.add_episode(_make_episode(20))
        batch = buf.sample(4, device="cpu")
        # obs should be [B, seq_len+1, obs_dim]
        assert batch["obs"].shape[0] == 4
        assert batch["obs"].shape[1] == 6  # seq_len + 1
        assert batch["obs"].shape[2] == 24
        assert batch["actions"].shape == (4, 6, 6)
        assert batch["rewards"].shape == (4, 6)
        assert batch["dones"].shape == (4, 6)

    def test_sample_dtype_is_float(self):
        from fragile.learning.rl.replay_buffer import SequenceReplayBuffer

        buf = SequenceReplayBuffer(capacity=10000, seq_len=3)
        buf.add_episode(_make_episode(10))
        batch = buf.sample(2)
        assert batch["obs"].dtype == torch.float32
        assert batch["rewards"].dtype == torch.float32

    def test_short_episodes_fallback(self):
        """Buffer should not crash with episodes shorter than seq_len."""
        from fragile.learning.rl.replay_buffer import SequenceReplayBuffer

        buf = SequenceReplayBuffer(capacity=10000, seq_len=100)
        buf.add_episode(_make_episode(5))  # shorter than seq_len
        batch = buf.sample(2)  # should not raise
        assert batch["obs"].shape[0] == 2

    def test_multiple_episodes(self):
        from fragile.learning.rl.replay_buffer import SequenceReplayBuffer

        buf = SequenceReplayBuffer(capacity=10000, seq_len=3)
        for _ in range(10):
            buf.add_episode(_make_episode(15))
        assert buf.num_episodes == 10
        batch = buf.sample(8)
        assert batch["obs"].shape[0] == 8

    def test_samples_explicit_tangent_and_covector_controls(self):
        from fragile.learning.rl.replay_buffer import SequenceReplayBuffer

        buf = SequenceReplayBuffer(capacity=10000, seq_len=3)
        ep = _make_episode(10)
        ep["controls"] = np.random.randn(10, D).astype(np.float32)
        ep["controls_tan"] = np.random.randn(10, D).astype(np.float32)
        ep["controls_cov"] = np.random.randn(10, D).astype(np.float32)
        ep["control_valid"] = np.ones(10, dtype=np.float32)
        ep["motor_macro_probs"] = np.random.rand(10, K).astype(np.float32)
        ep["motor_nuisance"] = np.random.randn(10, D).astype(np.float32)
        ep["motor_compliance"] = np.abs(np.random.randn(10, A, A).astype(np.float32))
        buf.add_episode(ep)
        batch = buf.sample(2, device="cpu")
        assert batch["controls"].shape == (2, 4, D)
        assert batch["controls_tan"].shape == (2, 4, D)
        assert batch["controls_cov"].shape == (2, 4, D)
        assert batch["motor_macro_probs"].shape == (2, 4, K)
        assert batch["motor_nuisance"].shape == (2, 4, D)
        assert batch["motor_compliance"].shape == (2, 4, A, A)


# ---------------------------------------------------------------------------
# Imagination (_imagine)
# ---------------------------------------------------------------------------


class TestImagination:
    def test_output_shapes(self, world_model, reward_head, critic, action_decoder, z, rw):
        from fragile.learning.rl.train_dreamer import _imagine

        out = _imagine(world_model, reward_head, critic, action_decoder, z, rw, H_IMAGINATION)
        assert out["z_states"].shape == (B, H_IMAGINATION, D)
        assert out["rw_states"].shape == (B, H_IMAGINATION, K)
        assert out["z_traj"].shape == (B, H_IMAGINATION, D)
        assert out["rw_traj"].shape == (B, H_IMAGINATION, K)
        assert out["controls"].shape == (B, H_IMAGINATION, D)
        assert out["controls_tan"].shape == (B, H_IMAGINATION, D)
        assert out["controls_cov"].shape == (B, H_IMAGINATION, D)
        assert out["motor_macro_probs"].shape == (B, H_IMAGINATION, K)
        assert out["motor_nuisance"].shape == (B, H_IMAGINATION, D)
        assert out["motor_compliance"].shape == (B, H_IMAGINATION, A, A)
        assert out["actions"].shape == (B, H_IMAGINATION, A)
        assert out["rewards"].shape == (B, H_IMAGINATION)
        assert out["action_log_std"].shape == (B, H_IMAGINATION, A)
        assert out["phi_eff"].shape == (B, H_IMAGINATION, 1)
        torch.testing.assert_close(out["controls"], out["controls_tan"])

    def test_z_inside_ball(self, world_model, reward_head, critic, action_decoder, z, rw):
        """Imagined z trajectory should stay inside the Poincare ball."""
        from fragile.learning.rl.train_dreamer import _imagine

        out = _imagine(world_model, reward_head, critic, action_decoder, z, rw, H_IMAGINATION)
        norms = out["z_traj"].norm(dim=-1)
        assert (norms < 1.5).all(), f"z escaped ball: max norm = {norms.max():.3f}"

    def test_action_texture_finite(
        self, world_model, reward_head, critic, action_decoder, z, rw,
    ):
        from fragile.learning.rl.train_dreamer import _imagine

        out = _imagine(world_model, reward_head, critic, action_decoder, z, rw, H_IMAGINATION)
        assert torch.isfinite(out["action_log_std"]).all()

    def test_imagination_outputs_are_detached(
        self, world_model, reward_head, critic, action_decoder, z, rw,
    ):
        from fragile.learning.rl.train_dreamer import _imagine

        out = _imagine(world_model, reward_head, critic, action_decoder, z, rw, H_IMAGINATION)
        assert not out["actions"].requires_grad
        assert not out["controls"].requires_grad
        assert not out["z_traj"].requires_grad

    def test_with_boris_and_jumps(self, world_model_full, critic, action_decoder, z, rw):
        """Imagination must work with full integration (Boris + jumps)."""
        from fragile.learning.rl.reward_head import RewardHead
        from fragile.learning.rl.train_dreamer import _imagine

        rh = RewardHead(world_model_full.potential_net, A, D_MODEL)
        out = _imagine(world_model_full, rh, critic, action_decoder, z, rw, H_IMAGINATION)
        assert out["z_traj"].shape == (B, H_IMAGINATION, D)
        assert torch.isfinite(out["rewards"]).all()

    def test_no_jump_imagination_keeps_router_weights(
        self, world_model, reward_head, critic, action_decoder, z, rw, monkeypatch,
    ):
        from fragile.learning.rl.train_dreamer import _imagine

        def _bad_chart_logits(self, z_in, control_in, rw_in):
            logits = torch.full((z_in.shape[0], K), -50.0, device=z_in.device)
            logits[:, -1] = 50.0
            return logits

        monkeypatch.setattr(
            world_model.chart_predictor,
            "forward",
            _bad_chart_logits.__get__(world_model.chart_predictor, type(world_model.chart_predictor)),
        )
        out = _imagine(world_model, reward_head, critic, action_decoder, z, rw, H_IMAGINATION)

        expected = rw.unsqueeze(1).expand_as(out["rw_traj"])
        torch.testing.assert_close(out["rw_traj"], expected)


# ---------------------------------------------------------------------------
# World-model routing
# ---------------------------------------------------------------------------


class TestWorldModelRouting:
    def test_no_jump_rollout_keeps_router_weights(self, world_model, z, rw, action, monkeypatch):
        tracked_rw = []

        def _bad_chart_logits(self, z_in, action_in, rw_in):
            logits = torch.full((z_in.shape[0], K), -50.0, device=z_in.device)
            logits[:, -1] = 50.0
            return logits

        def _identity_baoab(self, z_in, p_in, action_in, rw_in):
            tracked_rw.append(rw_in.detach().clone())
            phi = torch.zeros(z_in.shape[0], 1, device=z_in.device)
            return z_in, p_in, phi, {}

        monkeypatch.setattr(
            world_model.chart_predictor,
            "forward",
            _bad_chart_logits.__get__(world_model.chart_predictor, type(world_model.chart_predictor)),
        )
        monkeypatch.setattr(
            world_model,
            "_baoab_step",
            _identity_baoab.__get__(world_model, type(world_model)),
        )

        actions = action.unsqueeze(1).expand(-1, 2, -1)
        out = world_model(z, actions, rw)

        assert tracked_rw, "Expected BAOAB rollout calls"
        for rw_step in tracked_rw:
            torch.testing.assert_close(rw_step, rw)

        expected_logits = torch.full((B, 2, K), -50.0)
        expected_logits[:, :, -1] = 50.0
        torch.testing.assert_close(out["chart_logits"].cpu(), expected_logits)


# ---------------------------------------------------------------------------
# Atlas sync
# ---------------------------------------------------------------------------


class TestAtlasSync:
    def test_sync_rl_atlas_copies_encoder_centers(
        self, topo_model, world_model, critic, action_encoder, action_decoder,
    ):
        from fragile.learning.core.layers.atlas import _project_to_ball
        from fragile.learning.rl.train_dreamer import _sync_rl_atlas

        with torch.no_grad():
            topo_model.encoder.chart_centers.copy_(
                torch.randn_like(topo_model.encoder.chart_centers) * 0.4
            )
            world_model.potential_net.chart_tok.chart_centers.zero_()
            action_encoder.chart_tok.chart_centers.zero_()
            action_decoder.chart_tok.chart_centers.zero_()
            critic.chart_tok.chart_centers.zero_()

        _sync_rl_atlas(topo_model, world_model, critic, action_encoder, action_decoder)

        expected = _project_to_ball(topo_model.encoder.chart_centers.detach())
        torch.testing.assert_close(
            _project_to_ball(world_model.potential_net.chart_tok.chart_centers.detach()),
            expected,
        )
        torch.testing.assert_close(
            _project_to_ball(action_encoder.chart_tok.chart_centers.detach()),
            expected,
        )
        torch.testing.assert_close(
            _project_to_ball(action_decoder.chart_tok.chart_centers.detach()),
            expected,
        )
        torch.testing.assert_close(
            _project_to_ball(critic.chart_tok.chart_centers.detach()),
            expected,
        )
        assert not action_encoder.chart_tok.chart_centers.requires_grad
        assert not action_decoder.chart_tok.chart_centers.requires_grad
        assert not critic.chart_tok.chart_centers.requires_grad


# ---------------------------------------------------------------------------
# _train_step
# ---------------------------------------------------------------------------


class TestTrainStep:
    def _make_batch(self, device):
        """Create a fake batch matching buffer.sample() output."""
        T = H_IMAGINATION + 3  # keep replay horizon longer than WM supervision horizon
        controls = torch.randn(B, T, D, device=device)
        control_valid = torch.randint(0, 2, (B, T), device=device, dtype=torch.float32)
        actions = torch.randn(B, T, A, device=device)
        return {
            "obs": torch.randn(B, T, 24, device=device),
            "actions": actions,
            "action_means": actions.clone(),
            "controls": controls,
            "motor_macro_probs": torch.softmax(torch.randn(B, T, K, device=device), dim=-1),
            "motor_nuisance": torch.randn(B, T, D, device=device),
            "motor_compliance": torch.rand(B, T, A, A, device=device),
            "control_valid": control_valid,
            "rewards": torch.randn(B, T, device=device),
            "dones": torch.zeros(B, T, device=device),
        }

    def _make_optimizers(
        self,
        topo_model,
        jump_op,
        dyn_trans_model,
        world_model,
        reward_head,
        critic,
        action_encoder,
        action_decoder,
    ):
        optimizer_enc = torch.optim.Adam(
            list(topo_model.encoder.parameters())
            + list(topo_model.decoder.parameters())
            + list(jump_op.parameters())
            + list(dyn_trans_model.parameters()),
            lr=1e-3,
        )
        reward_own = [p for n, p in reward_head.named_parameters()
                      if "chart_tok" not in n and "z_embed" not in n]
        optimizer_wm = torch.optim.Adam([
            {"params": world_model.parameters(), "lr": 1e-3},
            {"params": reward_own, "lr": 1e-3},
        ])
        optimizer_boundary = torch.optim.Adam(
            list(action_encoder.parameters()) + list(action_decoder.parameters()),
            lr=1e-3,
        )
        optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
        return optimizer_enc, optimizer_wm, optimizer_boundary, optimizer_critic

    def test_returns_all_metric_keys(
        self,
        topo_model,
        jump_op,
        dyn_trans_model,
        world_model,
        reward_head,
        critic,
        action_encoder,
        action_decoder,
        dreamer_config, phase1_cfg,
    ):
        from fragile.learning.rl.train_dreamer import _train_step

        opt_enc, opt_wm, opt_boundary, opt_critic = self._make_optimizers(
            topo_model,
            jump_op,
            dyn_trans_model,
            world_model,
            reward_head,
            critic,
            action_encoder,
            action_decoder,
        )
        batch = self._make_batch("cpu")

        metrics = _train_step(
            topo_model,
            jump_op,
            dyn_trans_model,
            world_model,
            reward_head,
            critic,
            action_encoder,
            action_decoder,
            opt_enc,
            opt_wm,
            opt_critic,
            opt_boundary,
            batch,
            dreamer_config,
            phase1_cfg,
            epoch=0, current_hard_routing=True, current_tau=1.0,
        )

        expected_keys = {
            "enc/L_total", "enc/grad_norm",
            "enc/H_code_usage", "enc/code_usage_perplexity",
            "boundary/L_total", "boundary/L_action_recon", "boundary/L_decoder_supervise",
            "boundary/L_control_cycle", "boundary/L_control_supervise",
            "boundary/L_macro_supervise",
            "boundary/L_motor_nuisance_supervise",
            "boundary/L_motor_compliance_supervise",
            "boundary/L_value_intent_cos", "boundary/L_value_intent_huber",
            "boundary/value_intent_cos", "boundary/value_intent_l2",
            "boundary/control_raise_err", "boundary/control_lower_err",
            "boundary/macro_entropy", "boundary/macro_active",
            "boundary/motor_nuisance_norm_mean",
            "boundary/motor_compliance_mean", "boundary/motor_compliance_max",
            "boundary/grad_norm", "boundary/texture_std_mean", "boundary/texture_std_max",
            "wm/L_geodesic", "wm/L_chart", "wm/L_reward", "wm/L_momentum",
            "wm/L_energy", "wm/L_hodge",
            "wm/grad_norm", "wm/chart_acc", "wm/chart_entropy", "wm/chart_confidence",
            "critic/L_critic", "critic/L_value", "critic/L_poisson", "critic/grad_norm",
            "actor/L_return", "actor/return_mean", "actor/reward_mean",
            "actor/control_norm_mean", "actor/action_abs_mean",
            "actor/grad_norm_decoder", "actor/grad_norm_critic",
            "actor/update_applied", "actor/horizon", "actor/batch_size",
            "policy/control_norm_mean", "policy/control_norm_max",
            "policy/control_cov_norm_mean", "policy/control_cov_norm_max",
            "policy/action_abs_mean", "policy/action_sat_frac",
            "policy/motor_macro_entropy", "policy/motor_macro_active",
            "policy/motor_nuisance_norm_mean",
            "policy/motor_compliance_mean", "policy/motor_compliance_max",
            "imagination/reward_mean", "imagination/reward_std",
            "imagination/reward_sum_mean", "imagination/return_mean",
            "imagination/return_std", "imagination/reward_only_return_mean",
            "imagination/bootstrap0_mean", "imagination/bootstrap_ratio",
            "imagination/bootstrap_share", "imagination/router_entropy",
            "imagination/router_drift", "imagination/discounted_reward_mean",
            "imagination/terminal_value_mean", "imagination/boundary_value_mean",
            "imagination/boundary_ratio", "critic/value_mean", "critic/phi_eff_mean",
            "critic/value_bias", "critic/value_abs_err",
            "critic/replay_bellman_mean", "critic/replay_bellman_abs",
            "critic/replay_bellman_std", "critic/replay_rtg_mean",
            "critic/replay_rtg_bias", "critic/replay_rtg_abs_err",
            "critic/replay_calibration_err", "critic/replay_calibration_max",
            "critic/replay_calibration_bins", "train/replay_horizon", "train/wm_horizon",
            "geometric/z_norm_mean", "geometric/z_norm_max",
            "geometric/jump_frac",
            "geometric/hodge_conservative", "geometric/hodge_solenoidal",
            "geometric/hodge_harmonic", "geometric/energy_var",
            "chart/usage_entropy", "chart/active_charts",
            "chart/active_symbols", "chart/active_symbol_fraction",
            "chart/router_confidence", "chart/wm_center_drift",
            "chart/action_encoder_center_drift",
            "chart/action_decoder_center_drift",
            "chart/critic_center_drift",
            "time/actor",
        }
        missing = expected_keys - set(metrics.keys())
        assert not missing, f"Missing metric keys: {missing}"

    def test_all_metrics_finite(
        self,
        topo_model,
        jump_op,
        dyn_trans_model,
        world_model,
        reward_head,
        critic,
        action_encoder,
        action_decoder,
        dreamer_config, phase1_cfg,
    ):
        from fragile.learning.rl.train_dreamer import _train_step

        opt_enc, opt_wm, opt_boundary, opt_critic = self._make_optimizers(
            topo_model,
            jump_op,
            dyn_trans_model,
            world_model,
            reward_head,
            critic,
            action_encoder,
            action_decoder,
        )
        batch = self._make_batch("cpu")

        metrics = _train_step(
            topo_model, jump_op, dyn_trans_model, world_model, reward_head, critic,
            action_encoder, action_decoder, opt_enc, opt_wm, opt_critic, opt_boundary,
            batch, dreamer_config, phase1_cfg,
            epoch=0, current_hard_routing=True, current_tau=1.0,
        )
        for k, v in metrics.items():
            assert math.isfinite(v), f"Metric {k} is not finite: {v}"

    def test_parameters_updated(
        self,
        topo_model,
        jump_op,
        dyn_trans_model,
        world_model,
        reward_head,
        critic,
        action_encoder,
        action_decoder,
        dreamer_config, phase1_cfg,
    ):
        """After a train step, WM, boundary, critic, and encoder params should change."""
        from fragile.learning.rl.train_dreamer import _train_step

        opt_enc, opt_wm, opt_boundary, opt_critic = self._make_optimizers(
            topo_model,
            jump_op,
            dyn_trans_model,
            world_model,
            reward_head,
            critic,
            action_encoder,
            action_decoder,
        )

        wm_before = {n: p.clone() for n, p in world_model.named_parameters()}
        action_encoder_before = {n: p.clone() for n, p in action_encoder.named_parameters()}
        action_decoder_before = {n: p.clone() for n, p in action_decoder.named_parameters()}
        critic_before = {n: p.clone() for n, p in critic.named_parameters()}
        enc_before = {n: p.clone() for n, p in topo_model.encoder.named_parameters()}

        batch = self._make_batch("cpu")
        _train_step(
            topo_model, jump_op, dyn_trans_model, world_model, reward_head, critic,
            action_encoder, action_decoder, opt_enc, opt_wm, opt_critic, opt_boundary,
            batch, dreamer_config, phase1_cfg,
            epoch=0, current_hard_routing=True, current_tau=1.0,
        )

        wm_changed = any(
            not torch.equal(wm_before[n], p)
            for n, p in world_model.named_parameters()
        )
        action_encoder_changed = any(
            not torch.equal(action_encoder_before[n], p)
            for n, p in action_encoder.named_parameters()
        )
        action_decoder_changed = any(
            not torch.equal(action_decoder_before[n], p)
            for n, p in action_decoder.named_parameters()
        )
        critic_changed = any(
            not torch.equal(critic_before[n], p)
            for n, p in critic.named_parameters()
        )
        enc_changed = any(
            not torch.equal(enc_before[n], p)
            for n, p in topo_model.encoder.named_parameters()
        )
        assert wm_changed, "World model parameters did not change"
        assert action_encoder_changed, "Action encoder parameters did not change"
        assert action_decoder_changed, "Action decoder parameters did not change"
        assert critic_changed, "Critic parameters did not change"
        assert enc_changed, "Encoder parameters did not change"

    def test_per_chart_symbol_metrics_present(
        self,
        topo_model,
        jump_op,
        dyn_trans_model,
        world_model,
        reward_head,
        critic,
        action_encoder,
        action_decoder,
        dreamer_config, phase1_cfg,
    ):
        from fragile.learning.rl.train_dreamer import _train_step

        opt_enc, opt_wm, opt_boundary, opt_critic = self._make_optimizers(
            topo_model,
            jump_op,
            dyn_trans_model,
            world_model,
            reward_head,
            critic,
            action_encoder,
            action_decoder,
        )
        batch = self._make_batch("cpu")

        metrics = _train_step(
            topo_model, jump_op, dyn_trans_model, world_model, reward_head, critic,
            action_encoder, action_decoder, opt_enc, opt_wm, opt_critic, opt_boundary,
            batch, dreamer_config, phase1_cfg,
            epoch=0, current_hard_routing=True, current_tau=1.0,
        )

        for chart_idx in range(K):
            for suffix in ("active_codes", "code_entropy", "code_perplexity"):
                key = f"chart/{chart_idx}/{suffix}"
                assert key in metrics, f"Missing per-chart symbol metric {key}"
                assert math.isfinite(metrics[key]), f"Metric {key} is not finite"

    def test_frozen_encoder(
        self,
        topo_model,
        jump_op,
        dyn_trans_model,
        world_model,
        reward_head,
        critic,
        action_encoder,
        action_decoder,
        dreamer_config, phase1_cfg,
    ):
        """When freeze_encoder=True, encoder params must not change."""
        from fragile.learning.rl.train_dreamer import _train_step

        dreamer_config.freeze_encoder = True
        topo_model.encoder.eval()
        topo_model.decoder.eval()
        for p in topo_model.parameters():
            p.requires_grad_(False)

        enc_before = {n: p.clone() for n, p in topo_model.encoder.named_parameters()}

        # Encoder optimizer still created but won't update frozen params
        opt_enc, opt_wm, opt_boundary, opt_critic = self._make_optimizers(
            topo_model,
            jump_op,
            dyn_trans_model,
            world_model,
            reward_head,
            critic,
            action_encoder,
            action_decoder,
        )
        batch = self._make_batch("cpu")

        _train_step(
            topo_model, jump_op, dyn_trans_model, world_model, reward_head, critic,
            action_encoder, action_decoder, opt_enc, opt_wm, opt_critic, opt_boundary,
            batch, dreamer_config, phase1_cfg,
            epoch=0, current_hard_routing=True, current_tau=1.0,
        )

        for n, p in topo_model.encoder.named_parameters():
            assert torch.equal(enc_before[n], p), \
                f"Encoder param {n} changed when frozen"

    def test_frozen_encoder_still_reports_recon_metrics(
        self,
        topo_model,
        jump_op,
        dyn_trans_model,
        world_model,
        reward_head,
        critic,
        action_encoder,
        action_decoder,
        dreamer_config, phase1_cfg,
    ):
        from fragile.learning.rl.train_dreamer import _train_step

        dreamer_config.freeze_encoder = True
        topo_model.encoder.eval()
        topo_model.decoder.eval()
        for p in topo_model.parameters():
            p.requires_grad_(False)

        opt_enc, opt_wm, opt_boundary, opt_critic = self._make_optimizers(
            topo_model,
            jump_op,
            dyn_trans_model,
            world_model,
            reward_head,
            critic,
            action_encoder,
            action_decoder,
        )
        batch = self._make_batch("cpu")

        metrics = _train_step(
            topo_model, jump_op, dyn_trans_model, world_model, reward_head, critic,
            action_encoder, action_decoder, opt_enc, opt_wm, opt_critic, opt_boundary,
            batch, dreamer_config, phase1_cfg,
            epoch=0, current_hard_routing=True, current_tau=1.0,
        )

        for key in ("enc/recon", "enc/vq", "enc/H_code_usage", "enc/code_usage_perplexity"):
            assert key in metrics, f"Missing frozen-encoder metric {key}"
            assert math.isfinite(metrics[key]), f"Metric {key} is not finite"
        assert metrics["enc/grad_norm"] == pytest.approx(0.0)

    def test_two_steps_no_error(
        self,
        topo_model,
        jump_op,
        dyn_trans_model,
        world_model,
        reward_head,
        critic,
        action_encoder,
        action_decoder,
        dreamer_config, phase1_cfg,
    ):
        """Two consecutive train steps should not raise."""
        from fragile.learning.rl.train_dreamer import _train_step

        opt_enc, opt_wm, opt_boundary, opt_critic = self._make_optimizers(
            topo_model,
            jump_op,
            dyn_trans_model,
            world_model,
            reward_head,
            critic,
            action_encoder,
            action_decoder,
        )
        batch = self._make_batch("cpu")
        _train_step(
            topo_model, jump_op, dyn_trans_model, world_model, reward_head, critic,
            action_encoder, action_decoder, opt_enc, opt_wm, opt_critic, opt_boundary,
            batch, dreamer_config, phase1_cfg,
            epoch=0, current_hard_routing=True, current_tau=1.0,
        )
        batch2 = self._make_batch("cpu")
        _train_step(
            topo_model, jump_op, dyn_trans_model, world_model, reward_head, critic,
            action_encoder, action_decoder, opt_enc, opt_wm, opt_critic, opt_boundary,
            batch2, dreamer_config, phase1_cfg,
            epoch=1, current_hard_routing=True, current_tau=1.0,
        )

    def test_actor_return_update_emits_metrics(
        self,
        topo_model,
        jump_op,
        dyn_trans_model,
        world_model,
        reward_head,
        critic,
        action_encoder,
        action_decoder,
        dreamer_config, phase1_cfg,
    ):
        from fragile.learning.rl.train_dreamer import _train_step

        dreamer_config.actor_return_warmup_epochs = 0
        dreamer_config.actor_return_update_every = 1
        dreamer_config.w_actor_return = 1.0

        opt_enc, opt_wm, opt_boundary, opt_critic = self._make_optimizers(
            topo_model,
            jump_op,
            dyn_trans_model,
            world_model,
            reward_head,
            critic,
            action_encoder,
            action_decoder,
        )
        batch = self._make_batch("cpu")

        metrics = _train_step(
            topo_model, jump_op, dyn_trans_model, world_model, reward_head, critic,
            action_encoder, action_decoder, opt_enc, opt_wm, opt_critic, opt_boundary,
            batch, dreamer_config, phase1_cfg,
            epoch=0, update_idx=0, current_hard_routing=True, current_tau=1.0,
        )

        assert metrics["actor/update_applied"] == pytest.approx(1.0)
        assert metrics["actor/horizon"] == pytest.approx(dreamer_config.actor_return_horizon)
        assert metrics["actor/batch_size"] == pytest.approx(dreamer_config.actor_return_batch_size)
        assert math.isfinite(metrics["actor/L_return"])
        assert math.isfinite(metrics["actor/return_mean"])
        assert math.isfinite(metrics["actor/reward_mean"])
        assert metrics["actor/grad_norm_decoder"] > 0.0
        assert metrics["actor/grad_norm_critic"] > 0.0

    def test_replay_horizon_not_truncated_to_wm_horizon(
        self,
        topo_model,
        jump_op,
        dyn_trans_model,
        world_model,
        reward_head,
        critic,
        action_encoder,
        action_decoder,
        dreamer_config, phase1_cfg,
    ):
        from fragile.learning.rl.train_dreamer import _train_step

        opt_enc, opt_wm, opt_boundary, opt_critic = self._make_optimizers(
            topo_model,
            jump_op,
            dyn_trans_model,
            world_model,
            reward_head,
            critic,
            action_encoder,
            action_decoder,
        )
        batch = self._make_batch("cpu")

        metrics = _train_step(
            topo_model, jump_op, dyn_trans_model, world_model, reward_head, critic,
            action_encoder, action_decoder, opt_enc, opt_wm, opt_critic, opt_boundary,
            batch, dreamer_config, phase1_cfg,
            epoch=0, current_hard_routing=True, current_tau=1.0,
        )

        replay_horizon = int(metrics["train/replay_horizon"])
        wm_horizon = int(metrics["train/wm_horizon"])
        assert replay_horizon == batch["actions"].shape[1] - 1
        assert wm_horizon == max(H_WM, H_IMAGINATION)
        assert replay_horizon > wm_horizon


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------


class TestOptimizerHelpers:
    def test_optimizer_parameters_deduplicate(self):
        from fragile.learning.rl.train_dreamer import _optimizer_parameters

        p = nn.Parameter(torch.tensor([1.0]))
        q = nn.Parameter(torch.tensor([2.0]))

        class DummyOptimizer:
            def __init__(self):
                self.param_groups = [{"params": [p, q, p]}]

        params = _optimizer_parameters(DummyOptimizer())
        assert len(params) == 2
        assert params[0] is p
        assert params[1] is q


# ---------------------------------------------------------------------------
# Observation normalization
# ---------------------------------------------------------------------------


class TestObservationNormalizer:
    def test_round_trip_tensor(self):
        from fragile.learning.rl.train_dreamer import ObservationNormalizer

        normalizer = ObservationNormalizer(
            mean=torch.tensor([1.0, -2.0]),
            std=torch.tensor([2.0, 4.0]),
            min_std=1e-3,
        )
        raw = torch.tensor([[3.0, 6.0]], dtype=torch.float32)
        norm = normalizer.normalize_tensor(raw)
        restored = normalizer.denormalize_tensor(norm)

        assert torch.allclose(norm, torch.tensor([[1.0, 2.0]]))
        assert torch.allclose(restored, raw)

    def test_from_episodes_clamps_small_std(self, device):
        from fragile.learning.rl.train_dreamer import ObservationNormalizer

        episodes = [
            {"obs": np.array([[1.0, 5.0], [1.0, 7.0]], dtype=np.float32)},
            {"obs": np.array([[1.0, 9.0]], dtype=np.float32)},
        ]
        normalizer = ObservationNormalizer.from_episodes(
            episodes,
            device=device,
            min_std=0.5,
        )

        assert torch.allclose(normalizer.mean.cpu(), torch.tensor([1.0, 7.0]))
        assert normalizer.std[0].item() == pytest.approx(0.5)
        assert normalizer.std[1].item() > 0.5


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_and_load(
        self,
        tmp_path,
        topo_model,
        jump_op,
        dyn_trans_model,
        world_model,
        action_encoder,
        action_decoder,
        critic,
        reward_head,
    ):
        from fragile.learning.rl.config import DreamerConfig
        from fragile.learning.rl.train_dreamer import ObservationNormalizer, _save_checkpoint

        cfg = DreamerConfig(device="cpu")
        obs_normalizer = ObservationNormalizer(
            mean=torch.zeros(24),
            std=torch.ones(24),
            min_std=1e-3,
        )
        opt_enc = torch.optim.Adam(
            list(topo_model.encoder.parameters())
            + list(topo_model.decoder.parameters())
            + list(jump_op.parameters())
            + list(dyn_trans_model.parameters()),
            lr=1e-3,
        )
        opt_wm = torch.optim.Adam(world_model.parameters(), lr=1e-3)
        opt_boundary = torch.optim.Adam(
            list(action_encoder.parameters()) + list(action_decoder.parameters()),
            lr=1e-3,
        )
        opt_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
        sch_enc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_enc, T_max=10, eta_min=1e-5)
        sch_wm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_wm, T_max=10, eta_min=1e-5)
        sch_boundary = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_boundary,
            T_max=10,
            eta_min=1e-5,
        )
        sch_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_critic, T_max=10, eta_min=1e-5,
        )

        path = str(tmp_path / "test_ckpt.pt")
        _save_checkpoint(
            path, topo_model, jump_op, dyn_trans_model, world_model,
            action_encoder, action_decoder, critic, reward_head,
            opt_enc, opt_wm, opt_boundary, opt_critic,
            sch_enc, sch_wm, sch_boundary, sch_critic,
            epoch=42, config=cfg,
            metrics={"test/metric": 1.23},
            obs_normalizer=obs_normalizer,
        )

        assert os.path.exists(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 42
        assert "encoder" in ckpt
        assert "decoder" in ckpt
        assert "jump_op" in ckpt
        assert "dyn_trans_model" in ckpt
        assert "world_model" in ckpt
        assert "action_encoder" in ckpt
        assert "action_decoder" in ckpt
        assert "critic" in ckpt
        assert "reward_head" in ckpt
        assert "optimizer_enc" in ckpt
        assert "optimizer_wm" in ckpt
        assert "optimizer_boundary" in ckpt
        assert "optimizer_critic" in ckpt
        assert "scheduler_enc" in ckpt
        assert "scheduler_wm" in ckpt
        assert "scheduler_boundary" in ckpt
        assert "scheduler_critic" in ckpt
        assert ckpt["metrics"]["test/metric"] == 1.23
        assert ckpt["obs_normalizer"] is not None
        assert torch.equal(ckpt["obs_normalizer"]["mean"], torch.zeros(24))
        assert torch.equal(ckpt["obs_normalizer"]["std"], torch.ones(24))

    def test_reload_state_dict(
        self,
        tmp_path,
        topo_model,
        jump_op,
        dyn_trans_model,
        world_model,
        action_encoder,
        action_decoder,
        critic,
        reward_head,
    ):
        """Model state_dicts must be loadable after saving."""
        from fragile.learning.rl.config import DreamerConfig
        from fragile.learning.rl.train_dreamer import _save_checkpoint

        cfg = DreamerConfig(device="cpu")
        opt_enc = torch.optim.Adam(
            list(topo_model.encoder.parameters())
            + list(topo_model.decoder.parameters())
            + list(jump_op.parameters())
            + list(dyn_trans_model.parameters()),
            lr=1e-3,
        )
        opt_wm = torch.optim.Adam(world_model.parameters(), lr=1e-3)
        opt_boundary = torch.optim.Adam(
            list(action_encoder.parameters()) + list(action_decoder.parameters()),
            lr=1e-3,
        )
        opt_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
        sch_enc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_enc, T_max=10, eta_min=1e-5)
        sch_wm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_wm, T_max=10, eta_min=1e-5)
        sch_boundary = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_boundary,
            T_max=10,
            eta_min=1e-5,
        )
        sch_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_critic, T_max=10, eta_min=1e-5,
        )

        path = str(tmp_path / "ckpt2.pt")
        _save_checkpoint(
            path, topo_model, jump_op, dyn_trans_model, world_model,
            action_encoder, action_decoder, critic, reward_head,
            opt_enc, opt_wm, opt_boundary, opt_critic,
            sch_enc, sch_wm, sch_boundary, sch_critic,
            epoch=0, config=cfg,
        )

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Create fresh models and load
        from fragile.learning.core.layers.atlas import (
            PrimitiveAttentiveAtlasEncoder,
            PrimitiveTopologicalDecoder,
        )
        from fragile.learning.core.layers import FactorizedJumpOperator
        from fragile.learning.rl.boundary import (
            GeometricActionBoundaryDecoder,
            GeometricActionEncoder,
        )
        from fragile.learning.rl.critic import GeometricCritic
        from fragile.learning.rl.reward_head import RewardHead
        from fragile.learning.vla.losses import DynamicsTransitionModel
        from fragile.learning.vla.covariant_world_model import GeometricWorldModel

        enc2 = PrimitiveAttentiveAtlasEncoder(
            input_dim=24, hidden_dim=D_MODEL, latent_dim=D,
            num_charts=K, codes_per_chart=4,
        )
        wm2 = GeometricWorldModel(
            latent_dim=D, action_dim=A, control_dim=D, num_charts=K, d_model=D_MODEL,
            hidden_dim=D_MODEL, n_refine_steps=1, use_boris=False, use_jump=False,
        )
        action_enc2 = GeometricActionEncoder(D, A, K, d_model=D_MODEL)
        action_dec2 = GeometricActionBoundaryDecoder(D, A, K, d_model=D_MODEL)
        critic2 = GeometricCritic(D, K, D_MODEL)
        rh2 = RewardHead(wm2.potential_net, A, D_MODEL)
        jump2 = FactorizedJumpOperator(num_charts=K, latent_dim=D)
        dyn2 = DynamicsTransitionModel(
            chart_dim=D,
            action_dim=A,
            num_charts=K,
            codes_per_chart=4,
            hidden_dim=D_MODEL,
        )

        enc2.load_state_dict(ckpt["encoder"])
        wm2.load_state_dict(ckpt["world_model"])
        action_enc2.load_state_dict(ckpt["action_encoder"])
        action_dec2.load_state_dict(ckpt["action_decoder"])
        critic2.load_state_dict(ckpt["critic"])
        rh2.load_state_dict(ckpt["reward_head"], strict=False)
        jump2.load_state_dict(ckpt["jump_op"])
        dyn2.load_state_dict(ckpt["dyn_trans_model"])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestDreamerConfig:
    def test_default_construction(self):
        from fragile.learning.rl.config import DreamerConfig

        cfg = DreamerConfig()
        assert cfg.domain == "walker"
        assert cfg.task == "walk"
        assert cfg.latent_dim == 16
        assert cfg.lr_actor == pytest.approx(1e-3)
        assert cfg.lr_wm == pytest.approx(1e-3)
        assert cfg.lr_encoder == pytest.approx(1e-3)
        assert cfg.lr_min == pytest.approx(1e-5)
        assert cfg.device in ("cpu", "cuda")

    def test_override(self):
        from fragile.learning.rl.config import DreamerConfig

        cfg = DreamerConfig(domain="cartpole", latent_dim=4)
        assert cfg.domain == "cartpole"
        assert cfg.latent_dim == 4

    def test_gas_config_defaults(self):
        from fragile.learning.rl.config import DreamerConfig

        cfg = DreamerConfig()
        assert cfg.use_gas is True
        assert cfg.gas_walkers == 5000
        assert cfg.gas_steps == 150
        assert cfg.gas_reward_coef == 2.0
        assert cfg.gas_dist_coef == 1.0
        assert cfg.gas_n_elite == 0
        assert cfg.gas_collect_every == 5
        assert cfg.gas_n_env_workers == 4
        assert cfg.gas_use_death_condition is True
        assert cfg.collect_n_env_workers == 1

    def test_phase4_defaults_match_theory(self):
        from fragile.learning.rl.config import DreamerConfig

        cfg = DreamerConfig()
        assert cfg.freeze_encoder is False
        assert cfg.w_momentum_reg == pytest.approx(0.01)
        assert cfg.w_energy_conservation == pytest.approx(0.01)
        assert cfg.w_hodge == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self, monkeypatch):
        """Parser should produce a valid config with no args."""
        from fragile.learning.rl.train_dreamer import _parse_args

        monkeypatch.setattr("sys.argv", ["prog"])
        cfg = _parse_args()
        assert cfg.domain == "walker"
        assert isinstance(cfg.device, str)

    def test_override_via_cli(self, monkeypatch):
        from fragile.learning.rl.train_dreamer import _parse_args

        monkeypatch.setattr("sys.argv", [
            "prog", "--domain", "cartpole", "--latent_dim", "4",
        ])
        cfg = _parse_args()
        assert cfg.domain == "cartpole"
        assert cfg.latent_dim == 4


# ---------------------------------------------------------------------------
# Rollout policy behavior
# ---------------------------------------------------------------------------


class TestRolloutPolicyBehavior:
    def test_collect_episode_uses_control_policy_and_deterministic_routing(self):
        from fragile.learning.rl.train_dreamer import _collect_episode

        class FakeTimeStep:
            def __init__(self, obs, reward=0.0, last=False):
                self.observation = {"obs": np.asarray(obs, dtype=np.float32)}
                self.reward = reward
                self._last = last

            def last(self):
                return self._last

        class FakeSpec:
            minimum = np.array([-1.0], dtype=np.float32)
            maximum = np.array([1.0], dtype=np.float32)
            shape = (1,)

        class FakeEnv:
            def __init__(self):
                self._step = 0

            def reset(self):
                self._step = 0
                return FakeTimeStep([0.1], last=False)

            def action_spec(self):
                return FakeSpec()

            def step(self, action):
                self._step += 1
                return FakeTimeStep([0.2], reward=1.0, last=self._step >= 1)

        class FakeEncoder:
            def __init__(self):
                self.calls = []

            def __call__(self, obs_t, *, hard_routing, hard_routing_tau):
                self.calls.append((hard_routing, hard_routing_tau))
                rw = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
                z = torch.tensor([[0.25]], dtype=torch.float32)
                return (None, None, None, None, rw, z)

        class FakeModel:
            def __init__(self):
                self.encoder = FakeEncoder()

        class FakeCritic(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, z, rw):
                return z[:, :1]

            def task_value(self, z, rw):
                return self.forward(z, rw)

        class FakeActionDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.sample_texture_calls = 0

            def sample_motor_texture(self, z):
                self.sample_texture_calls += 1
                delta = torch.full_like(z[:, :1], math.atanh(0.33) - math.atanh(0.22))
                return delta, torch.zeros_like(delta)

            def forward(self, z, control, rw):
                return {
                    "macro_logits": torch.zeros(1, 2),
                    "macro_probs": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                    "macro_idx": torch.tensor([0]),
                    "motor_nuisance": torch.zeros(1, 1),
                    "motor_compliance": torch.ones(1, 1, 1),
                    "action_raw": torch.full((1, 1), math.atanh(0.22), dtype=torch.float32),
                    "action_mean": torch.tensor([[0.22]], dtype=torch.float32),
                    "texture_std": torch.zeros(1, 1),
                    "log_std": torch.zeros(1, 1),
                }

        env = FakeEnv()
        model = FakeModel()
        critic = FakeCritic()
        action_decoder = FakeActionDecoder()

        ep = _collect_episode(
            env,
            critic,
            action_decoder,
            model,
            torch.device("cpu"),
            control_dim=1,
            num_action_macros=2,
            action_repeat=1,
            max_steps=1,
            hard_routing=True,
            hard_routing_tau=0.7,
        )

        assert action_decoder.sample_texture_calls == 1
        assert model.encoder.calls == [(True, -1.0)]
        assert abs(float(ep["actions"][0, 0]) - 0.33) < 1e-6
        assert abs(float(ep["action_means"][0, 0]) - 0.22) < 1e-6
        assert ep["control_valid"][0] == pytest.approx(1.0)
        assert ep["controls_tan"].shape == (2, 1)
        assert ep["controls_cov"].shape == (2, 1)
        assert ep["motor_macro_probs"].shape == (2, 2)
        assert ep["motor_nuisance"].shape == (2, 1)
        assert ep["motor_compliance"].shape == (2, 1, 1)

    def test_collect_episode_normalizes_obs_before_encoder(self):
        from fragile.learning.rl.train_dreamer import ObservationNormalizer, _collect_episode

        class FakeTimeStep:
            def __init__(self, obs, reward=0.0, last=False):
                self.observation = {"obs": np.asarray(obs, dtype=np.float32)}
                self.reward = reward
                self._last = last

            def last(self):
                return self._last

        class FakeSpec:
            minimum = np.array([-1.0], dtype=np.float32)
            maximum = np.array([1.0], dtype=np.float32)
            shape = (1,)

        class FakeEnv:
            def __init__(self):
                self._step = 0

            def reset(self):
                self._step = 0
                return FakeTimeStep([3.0], last=False)

            def action_spec(self):
                return FakeSpec()

            def step(self, action):
                self._step += 1
                return FakeTimeStep([3.0], reward=0.0, last=self._step >= 1)

        class FakeEncoder:
            def __init__(self):
                self.obs_seen = []

            def __call__(self, obs_t, *, hard_routing, hard_routing_tau):
                self.obs_seen.append(obs_t.clone())
                rw = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
                z = torch.tensor([[0.25]], dtype=torch.float32)
                return (None, None, None, None, rw, z)

        class FakeModel:
            def __init__(self):
                self.encoder = FakeEncoder()

        class FakeCritic(nn.Module):
            def forward(self, z, rw):
                return z[:, :1]

            def task_value(self, z, rw):
                return self.forward(z, rw)

        class FakeActionDecoder(nn.Module):
            def sample_motor_texture(self, z):
                return torch.zeros(1, 1), torch.zeros(1, 1)

            def forward(self, z, control, rw):
                return {
                    "macro_logits": torch.zeros(1, 2),
                    "macro_probs": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                    "macro_idx": torch.tensor([0]),
                    "motor_nuisance": torch.zeros(1, 1),
                    "motor_compliance": torch.ones(1, 1, 1),
                    "action_raw": torch.zeros(1, 1),
                    "action_mean": torch.zeros(1, 1),
                    "texture_std": torch.zeros(1, 1),
                    "log_std": torch.zeros(1, 1),
                }

        normalizer = ObservationNormalizer(
            mean=torch.tensor([1.0]),
            std=torch.tensor([2.0]),
            min_std=1e-3,
        )
        model = FakeModel()
        _collect_episode(
            FakeEnv(),
            FakeCritic(),
            FakeActionDecoder(),
            model,
            torch.device("cpu"),
            control_dim=1,
            num_action_macros=2,
            obs_normalizer=normalizer,
            action_repeat=1,
            max_steps=1,
        )

        assert len(model.encoder.obs_seen) == 1
        assert torch.allclose(model.encoder.obs_seen[0], torch.tensor([[1.0]]))

    def test_collect_parallel_episodes_returns_replay_compatible_batches(self):
        from fragile.learning.rl.train_dreamer import _collect_parallel_episodes

        class FakeActionSpace:
            minimum = np.array([-1.0], dtype=np.float32)
            maximum = np.array([1.0], dtype=np.float32)
            shape = (1,)

        class FakeVectorEnv:
            def __init__(self):
                self.n_workers = 2
                self.action_space = FakeActionSpace()
                self.steps = np.zeros(2, dtype=np.int32)
                self.batch_calls = []

            def reset_batch(self, env_indices=None, seeds=None):
                indices = np.asarray(env_indices, dtype=int)
                self.steps[indices] = 0
                return np.stack(
                    [
                        np.array([float(idx) + 0.1], dtype=np.float32)
                        for idx in indices
                    ],
                )

            def step_actions_batch(self, actions, dt=None, env_indices=None):
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
                    obs.append(np.array([float(idx) + self.steps[idx]], dtype=np.float32))
                return np.stack(obs), rewards, dones, truncated, infos

        class FakeEncoder:
            def __call__(self, obs_t, *, hard_routing, hard_routing_tau):
                batch = obs_t.shape[0]
                rw = torch.zeros(batch, 2, dtype=torch.float32)
                rw[:, 0] = 1.0
                z = torch.full((batch, 1), 0.25, dtype=torch.float32)
                return (None, None, None, None, rw, z)

        class FakeModel:
            def __init__(self):
                self.encoder = FakeEncoder()

        class FakeCritic(nn.Module):
            def forward(self, z, rw):
                return z[:, :1]

            def task_value(self, z, rw):
                return self.forward(z, rw)

        class FakeActionDecoder(nn.Module):
            def sample_motor_texture(self, z):
                batch = z.shape[0]
                return torch.zeros(batch, 1), torch.zeros(batch, 1)

            def forward(self, z, control, rw):
                batch = z.shape[0]
                return {
                    "macro_logits": torch.zeros(batch, 2),
                    "macro_probs": torch.tensor([[1.0, 0.0]], dtype=torch.float32).repeat(
                        batch,
                        1,
                    ),
                    "macro_idx": torch.zeros(batch, dtype=torch.long),
                    "motor_nuisance": torch.zeros(batch, 1),
                    "motor_compliance": torch.ones(batch, 1, 1),
                    "action_raw": torch.zeros(batch, 1),
                    "action_mean": torch.full((batch, 1), 0.2, dtype=torch.float32),
                    "texture_std": torch.zeros(batch, 1),
                    "log_std": torch.zeros(batch, 1),
                }

        env = FakeVectorEnv()
        episodes = _collect_parallel_episodes(
            env,
            FakeCritic(),
            FakeActionDecoder(),
            FakeModel(),
            torch.device("cpu"),
            control_dim=1,
            num_action_macros=2,
            num_episodes=2,
            action_repeat=1,
            max_steps=2,
            hard_routing=True,
            hard_routing_tau=0.7,
        )

        assert len(episodes) == 2
        assert episodes[0]["obs"].shape == (2, 1)
        assert episodes[1]["obs"].shape == (3, 1)
        assert episodes[0]["actions"].shape == (2, 1)
        assert episodes[1]["actions"].shape == (3, 1)
        assert episodes[0]["controls_tan"].shape == (2, 1)
        assert episodes[1]["controls_cov"].shape == (3, 1)
        assert episodes[0]["motor_macro_probs"].shape == (2, 2)
        assert episodes[1]["motor_compliance"].shape == (3, 1, 1)
        assert episodes[0]["rewards"][-1] == pytest.approx(0.0)
        assert episodes[1]["dones"][-1] == pytest.approx(1.0)
        assert env.batch_calls == [[0, 1], [1]]

    def test_eval_policy_uses_deterministic_boundary_action(self):
        from fragile.learning.rl.train_dreamer import _eval_policy

        class FakeTimeStep:
            def __init__(self, obs, reward=0.0, last=False):
                self.observation = {"obs": np.asarray(obs, dtype=np.float32)}
                self.reward = reward
                self._last = last

            def last(self):
                return self._last

        class FakeSpec:
            minimum = np.array([-1.0], dtype=np.float32)
            maximum = np.array([1.0], dtype=np.float32)
            shape = (1,)

        class FakeEnv:
            def __init__(self):
                self._step = 0

            def reset(self):
                self._step = 0
                return FakeTimeStep([0.1], last=False)

            def action_spec(self):
                return FakeSpec()

            def step(self, action):
                self._step += 1
                return FakeTimeStep([0.2], reward=1.0, last=self._step >= 1)

        class FakeEncoder:
            def __init__(self):
                self.calls = []

            def __call__(self, obs_t, *, hard_routing, hard_routing_tau):
                self.calls.append((hard_routing, hard_routing_tau))
                rw = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
                z = torch.tensor([[0.25]], dtype=torch.float32)
                return (None, None, None, None, rw, z)

        class FakeModel:
            def __init__(self):
                self.encoder = FakeEncoder()

        class FakeCritic(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, z, rw):
                return z[:, :1]

            def task_value(self, z, rw):
                return self.forward(z, rw)

        class FakeActionDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.decode_calls = 0
                self.sample_texture_calls = 0

            def sample_motor_texture(self, z):
                self.sample_texture_calls += 1
                return torch.zeros(1, 1), torch.zeros(1, 1)

            def forward(self, z, control, rw):
                return {
                    "macro_logits": torch.zeros(1, 2),
                    "macro_probs": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                    "macro_idx": torch.tensor([0]),
                    "motor_nuisance": torch.zeros(1, 1),
                    "motor_compliance": torch.ones(1, 1, 1),
                    "action_raw": torch.zeros(1, 1),
                    "action_mean": torch.tensor([[0.44]], dtype=torch.float32),
                    "texture_std": torch.zeros(1, 1),
                    "log_std": torch.zeros(1, 1),
                }

        env = FakeEnv()
        model = FakeModel()
        critic = FakeCritic()
        action_decoder = FakeActionDecoder()

        metrics = _eval_policy(
            env,
            critic,
            action_decoder,
            model,
            torch.device("cpu"),
            action_repeat=1,
            num_episodes=1,
            max_steps=1,
            hard_routing=True,
            hard_routing_tau=0.7,
        )

        assert action_decoder.sample_texture_calls == 0
        assert action_decoder.decode_calls == 0
        assert model.encoder.calls == [(True, -1.0)]
        assert metrics["eval/reward_mean"] == 1.0


# ---------------------------------------------------------------------------
# Gas collection
# ---------------------------------------------------------------------------


class TestGasCollection:
    @pytest.fixture
    def gas_config(self, tmp_path):
        from fragile.learning.rl.config import DreamerConfig

        return DreamerConfig(
            device="cpu",
            obs_dim=24,
            action_dim=A,
            latent_dim=D,
            num_charts=K,
            d_model=D_MODEL,
            hidden_dim=D_MODEL,
            codes_per_chart=4,
            use_gas=True,
            gas_walkers=4,
            gas_steps=3,
            gas_reward_coef=2.0,
            gas_dist_coef=1.0,
            gas_n_elite=0,
            gas_collect_every=1,
            gas_n_env_workers=1,
            gas_use_death_condition=True,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )

    def test_collect_gas_episodes_format(self, topo_model, gas_config):
        from fragile.learning.rl.train_dreamer import _collect_gas_episodes

        device = torch.device(gas_config.device)
        episodes, metrics = _collect_gas_episodes(
            None, None, topo_model, device, gas_config,
        )
        assert len(episodes) == gas_config.gas_walkers
        ep = episodes[0]
        assert ep["obs"].shape == (gas_config.gas_steps + 1, gas_config.obs_dim)
        assert ep["actions"].shape == (gas_config.gas_steps + 1, gas_config.action_dim)
        assert ep["action_means"].shape == (gas_config.gas_steps + 1, gas_config.action_dim)
        assert ep["controls"].shape == (gas_config.gas_steps + 1, gas_config.latent_dim)
        assert ep["controls_tan"].shape == (gas_config.gas_steps + 1, gas_config.latent_dim)
        assert ep["controls_cov"].shape == (gas_config.gas_steps + 1, gas_config.latent_dim)
        assert ep["control_valid"].shape == (gas_config.gas_steps + 1,)
        assert ep["motor_macro_probs"].shape == (
            gas_config.gas_steps + 1,
            gas_config.num_action_macros or gas_config.num_charts,
        )
        assert ep["motor_nuisance"].shape == (gas_config.gas_steps + 1, gas_config.latent_dim)
        assert ep["motor_compliance"].shape == (
            gas_config.gas_steps + 1,
            gas_config.action_dim,
            gas_config.action_dim,
        )
        assert ep["rewards"].shape == (gas_config.gas_steps + 1,)
        assert ep["dones"].shape == (gas_config.gas_steps + 1,)
        # Final timestep must be marked done
        assert ep["dones"][-1] == 1.0

    def test_collect_gas_episodes_with_policy(self, topo_model, gas_config):
        from fragile.learning.rl.train_dreamer import _collect_gas_episodes

        class FakeCritic(nn.Module):
            def forward(self, z, rw):
                return z[:, :1]

            def task_value(self, z, rw):
                return self.forward(z, rw)

        class FakeActionDecoder(nn.Module):
            def sample_motor_texture(self, z):
                return torch.zeros(z.shape[0], A), torch.zeros(z.shape[0], A)

            def forward(self, z, control, rw):
                return {
                    "macro_logits": torch.zeros(z.shape[0], gas_config.num_charts),
                    "macro_probs": torch.full(
                        (z.shape[0], gas_config.num_charts),
                        1.0 / gas_config.num_charts,
                    ),
                    "macro_idx": torch.zeros(z.shape[0], dtype=torch.long),
                    "motor_nuisance": z,
                    "motor_compliance": torch.eye(A).unsqueeze(0).expand(z.shape[0], -1, -1),
                    "action_raw": z[:, :A],
                    "action_mean": z[:, :A],
                    "texture_std": torch.zeros(z.shape[0], A),
                    "log_std": torch.zeros(z.shape[0], A),
                }

        device = torch.device(gas_config.device)
        episodes, metrics = _collect_gas_episodes(
            FakeCritic(), FakeActionDecoder(), topo_model, device, gas_config,
        )
        assert len(episodes) == gas_config.gas_walkers
        assert "gas/max_reward" in metrics
        assert "gas/alive_frac" in metrics
        assert episodes[0]["control_valid"][:-1].sum() == pytest.approx(gas_config.gas_steps)

    def test_gas_episodes_buffer_compatible(self, topo_model, gas_config):
        from fragile.learning.rl.replay_buffer import SequenceReplayBuffer
        from fragile.learning.rl.train_dreamer import _collect_gas_episodes

        device = torch.device(gas_config.device)
        episodes, _ = _collect_gas_episodes(
            None, None, topo_model, device, gas_config,
        )
        buf = SequenceReplayBuffer(capacity=10000, seq_len=2)
        for ep in episodes:
            buf.add_episode(ep)
        assert buf.num_episodes == gas_config.gas_walkers
        batch = buf.sample(2, device="cpu")
        assert batch["obs"].shape[0] == 2

    def test_collect_gas_episodes_clones_history_prefix(self, monkeypatch, gas_config):
        from fragile.learning.rl import train_dreamer

        gas_config.gas_walkers = 2
        gas_config.gas_steps = 2
        gas_config.obs_dim = 1
        gas_config.action_dim = 1
        gas_config.hard_routing = True
        gas_config.hard_routing_tau = 0.5

        class FakeActionSpace:
            shape = (1,)
            minimum = np.array([-1.0], dtype=np.float64)
            maximum = np.array([1.0], dtype=np.float64)

        class FakeEnv:
            def __init__(self, *args, **kwargs):
                self.action_space = FakeActionSpace()

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

        class FakeEncoder:
            def __init__(self):
                self.calls = []

            def __call__(self, obs_t, *, hard_routing, hard_routing_tau):
                self.calls.append((hard_routing, hard_routing_tau))
                rw = torch.full((obs_t.shape[0], gas_config.num_charts), 1.0 / gas_config.num_charts)
                return (None, None, None, None, rw, obs_t)

        class FakeModel:
            def __init__(self):
                self.encoder = FakeEncoder()

        class FakeCritic(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, z, rw):
                return z[:, :1]

            def task_value(self, z, rw):
                return self.forward(z, rw)

        class FakeActionDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.sample_texture_calls = 0

            def sample_motor_texture(self, z):
                self.sample_texture_calls += 1
                return torch.zeros(z.shape[0], 1), torch.zeros(z.shape[0], 1)

            def forward(self, z, control, rw):
                return {
                    "macro_logits": torch.zeros(z.shape[0], gas_config.num_charts),
                    "macro_probs": torch.full(
                        (z.shape[0], gas_config.num_charts),
                        1.0 / gas_config.num_charts,
                    ),
                    "macro_idx": torch.zeros(z.shape[0], dtype=torch.long),
                    "motor_nuisance": z,
                    "motor_compliance": torch.ones(z.shape[0], 1, 1),
                    "action_raw": z[:, :1],
                    "action_mean": z[:, :1],
                    "texture_std": torch.zeros(z.shape[0], 1),
                    "log_std": torch.zeros(z.shape[0], 1),
                }

        monkeypatch.setattr(train_dreamer, "VectorizedDMControlEnv", FakeEnv)
        monkeypatch.setattr(train_dreamer, "RoboticFractalGas", FakeGas)

        model = FakeModel()
        critic = FakeCritic()
        action_decoder = FakeActionDecoder()
        episodes, metrics = train_dreamer._collect_gas_episodes(
            critic,
            action_decoder,
            model,
            torch.device("cpu"),
            gas_config,
        )

        assert action_decoder.sample_texture_calls == gas_config.gas_steps
        assert model.encoder.calls == [(True, -1.0), (True, -1.0)]
        np.testing.assert_allclose(episodes[1]["obs"][:2], episodes[0]["obs"][:2])
        np.testing.assert_allclose(episodes[1]["actions"][:1], episodes[0]["actions"][:1])
        np.testing.assert_allclose(episodes[1]["action_means"][:1], episodes[0]["action_means"][:1])
        np.testing.assert_allclose(episodes[1]["controls"][:1], episodes[0]["controls"][:1])
        np.testing.assert_allclose(episodes[1]["controls_tan"][:1], episodes[0]["controls_tan"][:1])
        np.testing.assert_allclose(episodes[1]["controls_cov"][:1], episodes[0]["controls_cov"][:1])
        np.testing.assert_allclose(
            episodes[1]["motor_macro_probs"][:1],
            episodes[0]["motor_macro_probs"][:1],
        )
        np.testing.assert_allclose(
            episodes[1]["motor_nuisance"][:1],
            episodes[0]["motor_nuisance"][:1],
        )
        np.testing.assert_allclose(
            episodes[1]["motor_compliance"][:1],
            episodes[0]["motor_compliance"][:1],
        )
        np.testing.assert_allclose(episodes[1]["rewards"][:1], episodes[0]["rewards"][:1])
        assert metrics["gas/total_clones"] == 1.0


# ---------------------------------------------------------------------------
# Observation format consistency
# ---------------------------------------------------------------------------


class TestObsFormat:
    def test_obs_format_match(self):
        """DMControlEnv._flatten_obs and train_dreamer._flatten_obs must produce
        the same key ordering (both use dict insertion order, not sorted)."""
        from fragile.fractalai.robots.dm_control_env import DMControlEnv
        from fragile.learning.rl.train_dreamer import _flatten_obs

        env_wrapper = DMControlEnv(name="walker-walk", include_rgb=False)
        ts = env_wrapper.env.reset()

        # Both functions applied to the same time_step must agree
        obs_wrapper = env_wrapper._flatten_obs(ts)
        obs_dreamer = _flatten_obs(ts)

        np.testing.assert_allclose(
            obs_wrapper.astype(np.float32),
            obs_dreamer,
            atol=1e-5,
            err_msg="DMControlEnv and train_dreamer flatten_obs produce different orderings",
        )
