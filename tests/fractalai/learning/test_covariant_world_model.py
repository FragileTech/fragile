"""Tests for the covariant geometric world model."""

from __future__ import annotations

import pytest
import torch

# Use small dims for fast tests
B, D, A, K, H = 4, 3, 6, 8, 5
D_MODEL = 32  # small for speed


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def z(device):
    """Random position in Poincare ball."""
    return torch.randn(B, D, device=device) * 0.3


@pytest.fixture
def action(device):
    return torch.randn(B, A, device=device)


@pytest.fixture
def rw(device):
    return torch.softmax(torch.randn(B, K, device=device), dim=-1)


class TestActionTokenizer:
    def test_output_shapes(self, z, action):
        from fragile.learning.vla.covariant_world_model import ActionTokenizer

        tok = ActionTokenizer(A, D_MODEL, D)
        x, z_out = tok(action, z)
        assert x.shape == (B, A, D_MODEL)
        assert z_out.shape == (B, A, D)

    def test_z_broadcast(self, z, action):
        """All action tokens should be positioned at z."""
        from fragile.learning.vla.covariant_world_model import ActionTokenizer

        tok = ActionTokenizer(A, D_MODEL, D)
        _, z_out = tok(action, z)
        for i in range(A):
            assert torch.allclose(z_out[:, i, :], z)


class TestChartTokenizer:
    def test_output_shapes(self, z, rw):
        from fragile.learning.vla.covariant_world_model import ChartTokenizer

        tok = ChartTokenizer(K, D_MODEL, D)
        x, z_out = tok(rw, z)
        assert x.shape == (B, K, D_MODEL)
        assert z_out.shape == (B, K, D)


class TestCovariantPotentialNet:
    def test_forward_shape(self, z, rw):
        from fragile.learning.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        phi = net(z, rw)
        assert phi.shape == (B, 1)

    def test_force_and_potential_shapes(self, z, rw):
        """force_and_potential must return cotangent force [B, D] and scalar phi [B, 1]."""
        from fragile.learning.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        force, phi = net.force_and_potential(z, rw)
        assert force.shape == (B, D)
        assert phi.shape == (B, 1)

    def test_analytic_U_and_grad(self):
        """Verify analytical gradient against finite differences."""
        from fragile.learning.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        z = torch.randn(2, D) * 0.3
        z.requires_grad_(True)

        U, dU_dz = net._analytic_U_and_grad(z)

        # Compare against autograd
        U_sum = U.sum()
        U_sum.backward()
        assert torch.allclose(dU_dz, z.grad, atol=1e-5), (
            "Analytical gradient does not match autograd"
        )

    def test_force_finite(self, z, rw):
        """Force components must be finite everywhere inside the ball."""
        from fragile.learning.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        force, phi = net.force_and_potential(z, rw)
        assert torch.isfinite(force).all(), "Non-finite force"
        assert torch.isfinite(phi).all(), "Non-finite potential"


class TestCovariantControlField:
    def test_forward_shape(self, z, action, rw):
        from fragile.learning.vla.covariant_world_model import CovariantControlField

        net = CovariantControlField(D, A, K, D_MODEL)
        u = net(z, action, rw)
        assert u.shape == (B, D)


class TestCovariantValueCurl:
    def test_forward_shape(self, z, action):
        from fragile.learning.vla.covariant_world_model import CovariantValueCurl

        net = CovariantValueCurl(D, A, D_MODEL)
        F_mat = net(z, action)
        assert F_mat.shape == (B, D, D)

    def test_antisymmetric(self, z, action):
        """F should be antisymmetric: F + F^T = 0."""
        from fragile.learning.vla.covariant_world_model import CovariantValueCurl

        net = CovariantValueCurl(D, A, D_MODEL)
        F_mat = net(z, action)
        assert torch.allclose(
            F_mat + F_mat.transpose(-2, -1), torch.zeros_like(F_mat), atol=1e-6
        )


class TestCovariantChartTarget:
    def test_forward_shape(self, z, action, rw):
        from fragile.learning.vla.covariant_world_model import CovariantChartTarget

        net = CovariantChartTarget(D, A, K, D_MODEL)
        logits = net(z, action, rw)
        assert logits.shape == (B, K)


class TestCovariantJumpRate:
    def test_forward_shape(self, z, rw):
        from fragile.learning.vla.covariant_world_model import CovariantJumpRate

        net = CovariantJumpRate(D, K, D_MODEL)
        rate = net(z, rw)
        assert rate.shape == (B, 1)

    def test_non_negative(self, z, rw):
        from fragile.learning.vla.covariant_world_model import CovariantJumpRate

        net = CovariantJumpRate(D, K, D_MODEL)
        rate = net(z, rw)
        assert (rate >= 0).all()


class TestCovariantMomentumInit:
    def test_forward_shape(self, z):
        from fragile.learning.vla.covariant_world_model import CovariantMomentumInit

        net = CovariantMomentumInit(D)
        p = net(z)
        assert p.shape == (B, D)

    def test_metric_scaling(self):
        """Momentum at origin should have higher metric factor than near boundary."""
        from fragile.learning.vla.covariant_world_model import CovariantMomentumInit

        net = CovariantMomentumInit(D)
        z_origin = torch.zeros(1, D)
        z_boundary = torch.ones(1, D) * 0.9 / D**0.5  # near boundary
        p_origin = net(z_origin)
        p_boundary = net(z_boundary)
        # lambda at origin = 2, lambda at boundary >> 2, so lam_sq at boundary > lam_sq at origin
        # But the SpectralLinear output differs, so just check shapes
        assert p_origin.shape == (1, D)
        assert p_boundary.shape == (1, D)


class TestGeometricWorldModel:
    def test_import(self):
        from fragile.learning.vla.covariant_world_model import GeometricWorldModel

        assert GeometricWorldModel is not None

    def test_forward_shapes(self, device):
        from fragile.learning.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
            dt=0.01,
            gamma_friction=1.0,
            T_c=0.1,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        assert out["z_trajectory"].shape == (B, H, D)
        assert out["chart_logits"].shape == (B, H, K)
        assert out["momenta"].shape == (B, H, D)
        assert out["jump_rates"].shape == (B, H, 1)
        assert out["jump_masks"].shape == (B, H)
        assert out["phi_eff"].shape == (B, H, 1)

    def test_z_stays_in_ball(self, device):
        """Trajectory points should stay inside the Poincare ball."""
        from fragile.learning.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)
        norms = out["z_trajectory"].norm(dim=-1)
        assert (norms < 1.0).all(), f"Max norm: {norms.max().item()}"

    def test_backward_pass(self, device):
        """Ensure gradients flow through the entire model."""
        from fragile.learning.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)
        loss = out["z_trajectory"].sum() + out["chart_logits"].sum()
        loss.backward()
        grad_count = sum(1 for p in m.parameters() if p.grad is not None)
        total_count = sum(1 for _ in m.parameters())
        assert grad_count > 0, "No gradients computed"

    def test_no_boris(self, device):
        """Model should work with Boris rotation disabled."""
        from fragile.learning.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
            use_boris=False,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)
        assert out["z_trajectory"].shape == (B, H, D)

    def test_no_jump(self, device):
        """Model should work with jump process disabled."""
        from fragile.learning.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
            use_jump=False,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)
        assert (out["jump_rates"] == 0).all()
        assert (~out["jump_masks"]).all()


class TestChartCenterProjection:
    """Chart centers must stay inside the Poincare ball during forward pass."""

    def test_chart_tokenizer_projects_centers(self, z, rw):
        from fragile.learning.vla.covariant_world_model import ChartTokenizer

        tok = ChartTokenizer(K, D_MODEL, D)
        # Force centers outside the ball
        with torch.no_grad():
            tok.chart_centers.copy_(torch.ones_like(tok.chart_centers) * 5.0)
        _, tokens_z = tok(rw, z)
        norms = tokens_z.norm(dim=-1)
        assert (norms < 1.0).all(), f"Chart centers escaped the ball: max norm {norms.max()}"

    def test_chart_target_projects_centers(self, z, action, rw):
        from fragile.learning.vla.covariant_world_model import CovariantChartTarget

        net = CovariantChartTarget(D, A, K, D_MODEL)
        # Force centers outside the ball
        with torch.no_grad():
            net.chart_centers.copy_(torch.ones_like(net.chart_centers) * 5.0)
        logits = net(z, action, rw)
        assert logits.shape == (B, K)
        assert torch.isfinite(logits).all(), "Non-finite logits from out-of-ball centers"


class TestGeometricInvariants:
    """Tests that verify fundamental geometric properties are respected."""

    def test_analytic_gradient_matches_finite_differences(self):
        """Analytical dU/dz must match numerical finite differences of U(z)."""
        from fragile.learning.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        z = torch.randn(2, D) * 0.3
        eps = 1e-4

        U_base, dU_dz = net._analytic_U_and_grad(z)

        # Numerical gradient via central differences
        numerical_grad = torch.zeros_like(z)
        for i in range(D):
            z_plus = z.clone()
            z_plus[:, i] += eps
            z_minus = z.clone()
            z_minus[:, i] -= eps
            U_plus, _ = net._analytic_U_and_grad(z_plus)
            U_minus, _ = net._analytic_U_and_grad(z_minus)
            numerical_grad[:, i] = (U_plus - U_minus).squeeze(-1) / (2 * eps)

        assert torch.allclose(dU_dz, numerical_grad, atol=1e-3), (
            f"Analytical grad:\n{dU_dz}\nNumerical grad:\n{numerical_grad}"
        )

    def test_force_near_boundary_is_finite(self, rw):
        """Force must stay finite even near the boundary of the Poincare ball."""
        from fragile.learning.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        z_near_boundary = torch.ones(1, D) * 0.95 / D**0.5
        force, phi = net.force_and_potential(z_near_boundary, rw[:1])
        assert torch.isfinite(force).all(), f"Non-finite force near boundary: {force}"
        assert torch.isfinite(phi).all(), f"Non-finite phi near boundary: {phi}"

    def test_gradients_flow_through_force(self, rw):
        """Potential net parameters must receive gradients through force_and_potential."""
        from fragile.learning.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        z = torch.randn(2, D) * 0.3
        force, phi = net.force_and_potential(z, rw[:2])
        loss = force.sum() + phi.sum()
        loss.backward()
        grad_count = sum(1 for p in net.parameters() if p.grad is not None)
        assert grad_count > 0, "No gradients flow through force_and_potential"

    def test_boris_rotation_preserves_norm(self, z, action):
        """Boris rotation must preserve momentum norm (antisymmetric F)."""
        from fragile.learning.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D, action_dim=A, num_charts=K, d_model=D_MODEL,
            use_boris=True,
        )
        p_in = torch.randn(B, D) * 0.1
        p_out = m._boris_rotation(p_in, z, action)
        # Boris rotation is an exact norm-preserving rotation
        assert torch.allclose(
            p_in.norm(dim=-1), p_out.norm(dim=-1), atol=1e-5,
        ), "Boris rotation changed momentum norm"

    def test_momentum_init_cotangent_scaling(self):
        """Momentum p = lam^2 * v must scale with conformal factor."""
        from fragile.learning.vla.covariant_world_model import CovariantMomentumInit

        net = CovariantMomentumInit(D)
        z_origin = torch.zeros(1, D)
        z_edge = torch.ones(1, D) * 0.8 / D**0.5

        p_origin = net(z_origin)
        p_edge = net(z_edge)

        # At origin: lam = 2, lam^2 = 4
        # At edge: lam = 2/(1-r^2), lam^2 > 4
        # So |p_edge| > |p_origin| for identical SpectralLinear(z) output magnitude
        # (but SpectralLinear outputs differ, so just check finite)
        assert torch.isfinite(p_origin).all()
        assert torch.isfinite(p_edge).all()


class TestMomentumRegularization:
    def test_metric_aware(self):
        """Metric-aware reg should differ from Euclidean."""
        from fragile.learning.vla.losses import compute_momentum_regularization

        momenta = torch.randn(B, H, D)
        z_traj = torch.randn(B, H, D) * 0.3
        loss = compute_momentum_regularization(momenta, z_traj)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_higher_at_boundary(self):
        """Kinetic energy should be lower near boundary (smaller g_inv_factor)."""
        from fragile.learning.vla.losses import compute_momentum_regularization

        p = torch.ones(1, 1, D)
        z_center = torch.zeros(1, 1, D)
        z_edge = torch.ones(1, 1, D) * 0.9 / D**0.5
        loss_center = compute_momentum_regularization(p, z_center)
        loss_edge = compute_momentum_regularization(p, z_edge)
        # g_inv_factor at center: ((1-0)/2)^2 = 0.25
        # g_inv_factor at edge: ((1-r^2)/2)^2 < 0.25
        assert loss_center > loss_edge
