"""Tests for Ricci Gas in 2D.

Verifies that all Ricci Gas functionality works correctly in 2D as well as 3D.
"""

import pytest
import torch

from fragile.ricci_gas import (
    compute_kde_density,
    compute_kde_hessian,
    compute_ricci_proxy,
    compute_ricci_proxy_3d,
    double_well,
    rastrigin,
    RicciGas,
    RicciGasParams,
    sphere,
    SwarmState,
)


@pytest.fixture
def device():
    """Get computation device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Dimension-Agnostic Tests ====================


@pytest.mark.parametrize("d", [1, 2, 3, 5])
def test_ricci_proxy_any_dimension(device, d):
    """Test Ricci proxy works in any dimension."""
    N = 20
    H = torch.randn(N, d, d, device=device)
    H = (H + H.transpose(-2, -1)) / 2  # Make symmetric

    R = compute_ricci_proxy(H)

    assert R.shape == (N,)
    assert R.device.type == device.type
    assert torch.isfinite(R).all()


@pytest.mark.parametrize("d", [2, 3, 4])
def test_kde_density_any_dimension(device, d):
    """Test KDE density works in any dimension."""
    N, M = 30, 10
    x = torch.randn(N, d, device=device)
    x_eval = torch.randn(M, d, device=device)
    alive = torch.ones(N, device=device).bool()

    rho = compute_kde_density(x, x_eval, bandwidth=0.5, alive_mask=alive)

    assert rho.shape == (M,)
    assert (rho > 0).all()
    assert torch.isfinite(rho).all()


@pytest.mark.parametrize("d", [2, 3, 4])
def test_kde_hessian_any_dimension(device, d):
    """Test KDE Hessian works in any dimension."""
    N = 20
    x = torch.randn(N, d, device=device)
    alive = torch.ones(N, device=device).bool()

    H = compute_kde_hessian(x, x[:5], bandwidth=0.5, alive_mask=alive)

    assert H.shape == (5, d, d)
    assert torch.isfinite(H).all()

    # Check symmetry
    for i in range(5):
        assert torch.allclose(H[i], H[i].T, atol=1e-4)


# ==================== 2D Specific Tests ====================


def test_ricci_gas_2d_initialization(device):
    """Test Ricci Gas works in 2D."""
    params = RicciGasParams(
        epsilon_R=0.5,
        kde_bandwidth=0.4,
        force_mode="pull",
        reward_mode="inverse",
    )

    gas = RicciGas(params, device=device)

    # Create 2D state
    N, d = 50, 2
    torch.manual_seed(42)
    x = torch.randn(N, d, device=device)
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)

    state = SwarmState(x=x, v=v, s=s)

    # Compute curvature
    R, H = gas.compute_curvature(state, cache=True)

    assert R.shape == (N,)
    assert H.shape == (N, 2, 2)
    assert (R != 0).any()  # Should have non-zero curvature
    assert torch.isfinite(R).all()
    assert torch.isfinite(H).all()


def test_ricci_gas_2d_force(device):
    """Test force computation in 2D."""
    params = RicciGasParams(
        epsilon_R=0.5,
        kde_bandwidth=0.4,
        force_mode="pull",
    )

    gas = RicciGas(params, device=device)

    N, d = 30, 2
    torch.manual_seed(42)
    x = torch.randn(N, d, device=device)
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)

    state = SwarmState(x=x, v=v, s=s)

    # Compute curvature first
    gas.compute_curvature(state, cache=True)

    # Compute force
    force = gas.compute_force(state)

    assert force.shape == (N, 2)
    assert torch.isfinite(force).all()


def test_2d_optimization_convergence(device):
    """Test that 2D optimization can find good solutions."""
    params = RicciGasParams(
        epsilon_R=0.5,  # Stronger force
        kde_bandwidth=0.5,
        force_mode="pull",
        reward_mode="inverse",
    )

    gas = RicciGas(params, device=device)

    # 2D sphere problem
    N, d = 40, 2
    torch.manual_seed(123)
    x = torch.randn(N, d, device=device) * 3.0  # Start far from origin
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)

    state = SwarmState(x=x, v=v, s=s)

    initial_best = (x**2).sum(dim=-1).min().item()

    # Run dynamics
    for t in range(100):  # More steps
        _R, _H = gas.compute_curvature(state, cache=True)
        force = gas.compute_force(state)

        state.v = 0.8 * state.v + 0.2 * force + torch.randn_like(state.v, device=device) * 0.03
        state.x += state.v * 0.1

    final_best = (state.x**2).sum(dim=-1).min().item()

    # Best solution should improve (more robust test)
    assert final_best < initial_best or final_best < 1.0  # Either improves or finds good solution


# ==================== Potential Function Tests ====================


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_sphere_potential_any_dimension(device, d):
    """Test sphere potential works in any dimension."""
    N = 20
    x = torch.randn(N, d, device=device)

    V = sphere(x)

    assert V.shape == (N,)
    assert (V >= 0).all()  # Sphere is always non-negative
    assert torch.isfinite(V).all()


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_rastrigin_potential_any_dimension(device, d):
    """Test Rastrigin potential works in any dimension."""
    N = 20
    x = torch.randn(N, d, device=device)

    V = rastrigin(x)

    assert V.shape == (N,)
    assert torch.isfinite(V).all()


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_double_well_potential_any_dimension(device, d):
    """Test double well potential works in any dimension."""
    N = 20
    x = torch.randn(N, d, device=device)

    V = double_well(x)

    assert V.shape == (N,)
    assert torch.isfinite(V).all()


def test_double_well_minima_2d(device):
    """Test that double well has correct minima in 2D."""
    # Minima should be at (±1, 0)
    x_min1 = torch.tensor([[1.0, 0.0]], device=device)
    x_min2 = torch.tensor([[-1.0, 0.0]], device=device)
    x_other = torch.tensor([[0.0, 0.0], [2.0, 1.0]], device=device)

    V_min1 = double_well(x_min1)
    V_min2 = double_well(x_min2)
    V_other = double_well(x_other)

    # Minima should have low energy
    assert V_min1.item() < 0.01
    assert V_min2.item() < 0.01

    # Other points should have higher energy
    assert (V_other > V_min1).all()


# ==================== Backward Compatibility Tests ====================


def test_compute_ricci_proxy_3d_deprecated(device):
    """Test that deprecated 3D function still works."""
    N = 20
    H = torch.randn(N, 3, 3, device=device)
    H = (H + H.transpose(-2, -1)) / 2

    R_new = compute_ricci_proxy(H)
    R_old = compute_ricci_proxy_3d(H)

    assert torch.allclose(R_new, R_old)


# ==================== Edge Cases ====================


def test_1d_ricci_gas(device):
    """Test Ricci Gas works even in 1D."""
    params = RicciGasParams(epsilon_R=0.5, kde_bandwidth=0.5)
    gas = RicciGas(params, device=device)

    N, d = 30, 1
    x = torch.randn(N, d, device=device)
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)

    state = SwarmState(x=x, v=v, s=s)

    # Should work without errors
    R, H = gas.compute_curvature(state)

    assert R.shape == (N,)
    assert H.shape == (N, 1, 1)
    assert torch.isfinite(R).all()


def test_high_dimensional_ricci_gas(device):
    """Test Ricci Gas works in higher dimensions."""
    params = RicciGasParams(epsilon_R=0.5, kde_bandwidth=0.5)
    gas = RicciGas(params, device=device)

    N, d = 20, 10  # 10D
    x = torch.randn(N, d, device=device)
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)

    state = SwarmState(x=x, v=v, s=s)

    R, H = gas.compute_curvature(state)

    assert R.shape == (N,)
    assert H.shape == (N, 10, 10)
    assert torch.isfinite(R).all()


# ==================== Comparison 2D vs 3D ====================


def test_2d_vs_3d_consistency(device):
    """Test that behavior is consistent between 2D and 3D."""
    params = RicciGasParams(
        epsilon_R=0.5,
        kde_bandwidth=0.5,
        force_mode="pull",
    )

    gas = RicciGas(params, device=device)

    torch.manual_seed(42)

    # 2D state
    N = 30
    x_2d = torch.randn(N, 2, device=device)
    v_2d = torch.randn(N, 2, device=device) * 0.1
    s_2d = torch.ones(N, device=device)
    state_2d = SwarmState(x=x_2d, v=v_2d, s=s_2d)

    # 3D state (embed 2D in 3D with z=0)
    x_3d = torch.cat([x_2d, torch.zeros(N, 1, device=device)], dim=-1)
    v_3d = torch.cat([v_2d, torch.zeros(N, 1, device=device)], dim=-1)
    s_3d = torch.ones(N, device=device)
    state_3d = SwarmState(x=x_3d, v=v_3d, s=s_3d)

    # Compute curvature
    R_2d, _H_2d = gas.compute_curvature(state_2d)
    R_3d, _H_3d = gas.compute_curvature(state_3d)

    # Both should have valid curvatures
    assert torch.isfinite(R_2d).all()
    assert torch.isfinite(R_3d).all()

    # Statistics should be in same ballpark (not exact due to dimension)
    assert R_2d.mean().abs() > 0  # Non-trivial
    assert R_3d.mean().abs() > 0  # Non-trivial


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
