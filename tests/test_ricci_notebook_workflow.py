"""Test the complete notebook workflow for device compatibility.

This test replicates the exact steps from ricci_gas_visualization.ipynb
to ensure all device handling is correct.
"""

import pytest
import torch

from fragile.ricci_gas import (
    RicciGas,
    RicciGasParams,
    SwarmState,
    compute_kde_density,
    compute_kde_hessian,
    compute_ricci_proxy_3d,
    double_well_3d,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Test on both CPU and CUDA (if available)."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


def test_notebook_workflow_step_by_step(device):
    """Replicate the exact notebook workflow to catch device errors."""

    # Step 1: Initialize Ricci Gas (from cell 3)
    params = RicciGasParams(
        epsilon_R=0.5,
        kde_bandwidth=0.4,
        epsilon_Ric=0.01,
        force_mode="pull",
        reward_mode="inverse",
        R_crit=15.0,
        gradient_clip=10.0,
    )

    gas = RicciGas(params, device=device)
    assert gas.device == device

    # Step 2: Initialize Swarm (from cell 5)
    N = 50  # Smaller for faster testing
    d = 3

    torch.manual_seed(42)
    x = torch.rand(N, d, device=device) * 4.0 - 2.0
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)

    state = SwarmState(x=x, v=v, s=s)

    # Device comparison: cuda and cuda:0 should be treated as equal
    assert state.x.device.type == device.type
    assert state.v.device.type == device.type
    assert state.s.device.type == device.type

    # Step 3: Compute Initial Geometry (from cell 7)
    R, H = gas.compute_curvature(state, cache=True)

    assert R.device.type == device.type
    assert H.device.type == device.type
    assert R.shape == (N,)
    assert H.shape == (N, 3, 3)
    assert torch.all(torch.isfinite(R))
    assert torch.all(torch.isfinite(H))

    # Verify caching worked
    assert state.R is not None
    assert state.H is not None
    assert state.R.device.type == device.type
    assert state.H.device.type == device.type

    # Step 4: Run a few dynamics steps (from cell 11)
    T = 10  # Just a few steps for testing
    dt = 0.1
    gamma = 0.9

    for t in range(T):
        # Compute geometry
        R, H = gas.compute_curvature(state, cache=True)

        # Compute force and reward
        force = gas.compute_force(state)
        reward = gas.compute_reward(state)

        # Verify all on correct device
        assert R.device.type == device.type
        assert H.device.type == device.type
        assert force.device.type == device.type
        assert reward.device.type == device.type

        # Verify shapes
        assert force.shape == (N, 3)
        assert reward.shape == (N,)

        # Simple Langevin dynamics
        state.v = gamma * state.v + (1 - gamma) * force + torch.randn_like(state.v, device=device) * 0.05
        state.x = state.x + state.v * dt

        # Apply singularity regulation
        state = gas.apply_singularity_regulation(state)

        # Verify state is still on correct device
        assert state.x.device.type == device.type
        assert state.v.device.type == device.type
        assert state.s.device.type == device.type

        # Verify all values are finite
        assert torch.all(torch.isfinite(state.x))
        assert torch.all(torch.isfinite(state.v))
        assert torch.all(torch.isfinite(R))
        assert torch.all(torch.isfinite(force))
        assert torch.all(torch.isfinite(reward))


def test_lennard_jones_workflow(device):
    """Test the Lennard-Jones optimization workflow from the notebook."""

    def lennard_jones_energy(x, epsilon=1.0, sigma=1.0):
        """Compute LJ energy."""
        N = len(x)
        diff = x.unsqueeze(0) - x.unsqueeze(1)
        r = diff.norm(dim=-1)
        r = r + torch.eye(N, device=x.device) * 1e10
        r6 = (sigma / r) ** 6
        r12 = r6 ** 2
        V_pair = 4 * epsilon * (r12 - r6)
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        E = V_pair[mask].sum()
        return E, V_pair

    def lennard_jones_force(x, epsilon=1.0, sigma=1.0):
        """Compute LJ force."""
        x_grad = x.clone().requires_grad_(True)
        E, _ = lennard_jones_energy(x_grad, epsilon, sigma)
        F = -torch.autograd.grad(E, x_grad)[0]
        return F

    # Initialize cluster (from cell 23)
    N_atoms = 7  # Smaller for faster testing

    torch.manual_seed(123)
    x_lj = torch.randn(N_atoms, 3, device=device) * 2.0
    v_lj = torch.zeros(N_atoms, 3, device=device)
    s_lj = torch.ones(N_atoms, device=device)

    state_lj = SwarmState(x=x_lj, v=v_lj, s=s_lj)

    # Ricci Gas parameters
    params_lj = RicciGasParams(
        epsilon_R=0.3,
        kde_bandwidth=0.5,
        force_mode="pull",
        reward_mode="inverse",
        R_crit=None,
    )

    gas_lj = RicciGas(params_lj, device=device)

    # Run a few optimization steps (from cell 24)
    T_lj = 10
    dt_lj = 0.05
    gamma_lj = 0.8

    best_E = float('inf')

    for t in range(T_lj):
        # Compute Ricci geometry
        R_lj, H_lj = gas_lj.compute_curvature(state_lj, cache=True)
        F_ricci = gas_lj.compute_force(state_lj)

        # Compute LJ forces
        F_lj = lennard_jones_force(state_lj.x)

        # Verify device consistency
        assert R_lj.device.type == device.type
        assert H_lj.device.type == device.type
        assert F_ricci.device.type == device.type
        assert F_lj.device.type == device.type

        # Combined dynamics
        F_total = F_lj + F_ricci

        # Langevin update
        state_lj.v = gamma_lj * state_lj.v + (1 - gamma_lj) * F_total + torch.randn_like(state_lj.v, device=device) * 0.1
        state_lj.x = state_lj.x + state_lj.v * dt_lj

        # Compute energy
        E_current, _ = lennard_jones_energy(state_lj.x)

        assert E_current.device.type == device.type
        assert torch.isfinite(E_current)

        if E_current < best_E:
            best_E = E_current.item()

        # Verify state is on correct device
        assert state_lj.x.device.type == device.type
        assert state_lj.v.device.type == device.type
        assert state_lj.s.device.type == device.type

    # Should have converged somewhat
    assert best_E < float('inf')


def test_multiple_gas_instances(device):
    """Test creating multiple RicciGas instances with different devices."""

    params = RicciGasParams(epsilon_R=0.5, kde_bandwidth=0.4)

    # Create gas with explicit device
    gas1 = RicciGas(params, device=device)
    assert gas1.device == device

    # Create another one
    gas2 = RicciGas(params, device=device)
    assert gas2.device == device

    # They should work independently
    N = 20
    state1 = SwarmState(
        x=torch.randn(N, 3, device=device),
        v=torch.randn(N, 3, device=device),
        s=torch.ones(N, device=device),
    )

    state2 = SwarmState(
        x=torch.randn(N, 3, device=device),
        v=torch.randn(N, 3, device=device),
        s=torch.ones(N, device=device),
    )

    R1, H1 = gas1.compute_curvature(state1)
    R2, H2 = gas2.compute_curvature(state2)

    assert R1.device.type == device.type
    assert R2.device.type == device.type
    assert H1.device.type == device.type
    assert H2.device.type == device.type


def test_device_string_parameter(device):
    """Test that device can be passed as string."""

    params = RicciGasParams(epsilon_R=0.5, kde_bandwidth=0.4)

    # Pass as string
    gas = RicciGas(params, device=str(device))
    assert gas.device == device

    # Test it works
    N = 20
    state = SwarmState(
        x=torch.randn(N, 3, device=device),
        v=torch.randn(N, 3, device=device),
        s=torch.ones(N, device=device),
    )

    R, H = gas.compute_curvature(state)
    assert R.device.type == device.type
    assert H.device.type == device.type


def test_default_device_selection():
    """Test that device defaults to CUDA if available, else CPU."""

    params = RicciGasParams(epsilon_R=0.5, kde_bandwidth=0.4)

    # Don't pass device - should auto-select
    gas = RicciGas(params)

    expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert gas.device == expected_device


def test_ricci_curvature_nonzero():
    """Test that Ricci curvature is non-zero for typical configurations."""
    device = torch.device("cpu")

    params = RicciGasParams(
        epsilon_R=0.5,
        kde_bandwidth=0.4,
        epsilon_Ric=0.01,
        force_mode="pull",
        reward_mode="inverse",
    )

    gas = RicciGas(params, device=device)

    # Test with various configurations
    test_cases = [
        ("Random uniform", lambda N: torch.rand(N, 3, device=device) * 4.0 - 2.0),
        ("Random Gaussian", lambda N: torch.randn(N, 3, device=device)),
        ("Clustered", lambda N: torch.randn(N, 3, device=device) * 0.5),
        ("Diffuse", lambda N: torch.randn(N, 3, device=device) * 3.0),
    ]

    for name, x_fn in test_cases:
        N = 150
        torch.manual_seed(42)
        x = x_fn(N)
        v = torch.randn(N, 3, device=device) * 0.1
        s = torch.ones(N, device=device)

        state = SwarmState(x=x, v=v, s=s)
        R, H = gas.compute_curvature(state)

        # Ricci should be non-zero for non-trivial configurations
        assert (R != 0).any(), f"{name}: Ricci curvature is all zeros"
        assert torch.isfinite(R).all(), f"{name}: Ricci contains NaN/Inf"
        assert torch.isfinite(H).all(), f"{name}: Hessian contains NaN/Inf"

        print(f"✓ {name}: R ∈ [{R.min():.4f}, {R.max():.4f}], mean={R.mean():.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
