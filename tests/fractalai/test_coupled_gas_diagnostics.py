"""Independent finite algebra checks for the coupled-gas proof diagnostics."""

import itertools

import pytest
import torch

from fragile.fractalai.theory.coupled_gas_diagnostics import (
    fermionic_spectral_diagnostics,
    gaussian_viscous_force,
    viscous_kinetic_moment_budget,
    viscous_qsd_coercivity_margin,
)


@pytest.mark.parametrize("rates", [[], [0.0], [1.0], [1.0, 2.0], [1.0, 1.0], [0.0, 2.0]])
def test_spectral_formulas_against_all_occupation_states(rates):
    """Use explicit two-mode blocks; compare with independently enumerated Fock levels."""
    coefficients = torch.zeros((2 * len(rates), 2 * len(rates)), dtype=torch.float64)
    one_particle = []
    for index, rate in enumerate(rates):
        coefficients[2 * index, 2 * index + 1] = rate
        one_particle.extend([-rate, rate])
    result = fermionic_spectral_diagnostics(coefficients)
    levels = [
        (sum(energy * occupied for energy, occupied in zip(one_particle, state)), sum(state) % 2)
        for state in itertools.product([0, 1], repeat=len(one_particle))
    ]
    assert result.sea_energy == pytest.approx(min(energy for energy, _ in levels))
    for parity, prefix in [(0, "even"), (1, "odd")]:
        sector = [energy for energy, state_parity in levels if state_parity == parity]
        if not sector:
            assert getattr(result, f"{prefix}_ground_energy") is None
            assert getattr(result, f"{prefix}_ground_dimension") == 0
            continue
        ground = min(sector)
        higher = sorted({energy for energy in sector if energy > ground})
        assert getattr(result, f"{prefix}_ground_energy") == pytest.approx(ground)
        assert getattr(result, f"{prefix}_ground_dimension") == sector.count(ground)
        expected_gap = higher[0] - ground if higher else None
        assert getattr(result, f"{prefix}_excitation_gap") == expected_gap


def test_odd_mode_count_retains_zero_mode_ground_degeneracy():
    coefficients = torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    result = fermionic_spectral_diagnostics(coefficients)
    assert result.zero_modes == 1
    assert result.ground_dimension == 2
    assert result.even_ground_dimension == result.odd_ground_dimension == 1
    assert result.even_excitation_gap == result.odd_excitation_gap == pytest.approx(2.0)


def test_count_normalized_force_balances_momentum_and_dissipates_energy():
    generator = torch.Generator().manual_seed(31)
    positions = torch.randn((12, 3), generator=generator, dtype=torch.float64)
    velocities = torch.randn((12, 3), generator=generator, dtype=torch.float64)
    viscosity = 0.3
    force = gaussian_viscous_force(positions, velocities, viscosity=viscosity, bandwidth=1.2)
    torch.testing.assert_close(
        force.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-14, rtol=0
    )
    assert float((force * velocities).sum()) <= 0
    assert float(force.norm()) <= viscosity * float(velocities.norm()) + 1e-14
    kick = velocities + force / viscosity
    assert float(kick.square().sum()) <= float(velocities.square().sum()) + 1e-12
    assert float(kick.norm(dim=1).max()) <= float(velocities.norm(dim=1).max()) + 1e-12


def test_row_normalization_is_not_momentum_conservative():
    positions = torch.tensor([[0.0], [0.1], [10.0]], dtype=torch.float64)
    velocities = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float64)
    force = gaussian_viscous_force(
        positions, velocities, viscosity=0.3, bandwidth=1.0, row_normalized=True
    )
    assert float(force.sum()) == pytest.approx(-0.3)
    consensus = torch.ones_like(velocities)
    torch.testing.assert_close(
        gaussian_viscous_force(
            positions, consensus, viscosity=0.3, bandwidth=1.0, row_normalized=True
        ),
        torch.zeros_like(velocities),
    )


def test_row_ratio_survives_raw_gaussian_underflow():
    positions = torch.tensor([[0.0], [1000.0]], dtype=torch.float64)
    velocities = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    force = gaussian_viscous_force(
        positions, velocities, viscosity=0.3, bandwidth=1.0, row_normalized=True
    )
    torch.testing.assert_close(force, torch.tensor([[0.3], [-0.3]], dtype=torch.float64))


def test_force_commutes_with_spatial_rotation_and_row_permutation():
    positions = torch.tensor([[1.0, 2.0], [0.0, 1.0], [3.0, -1.0]], dtype=torch.float64)
    velocities = torch.tensor([[0.0, 1.0], [2.0, -1.0], [-1.0, 2.0]], dtype=torch.float64)
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    permutation = torch.tensor([2, 0, 1])
    force = gaussian_viscous_force(positions, velocities, viscosity=0.4, bandwidth=0.7)
    transformed = gaussian_viscous_force(
        positions[permutation] @ rotation.T,
        velocities[permutation] @ rotation.T,
        viscosity=0.4,
        bandwidth=0.7,
    )
    torch.testing.assert_close(transformed, force[permutation] @ rotation.T)


def test_reference_qsd_margin_and_parameter_validation():
    assert viscous_qsd_coercivity_margin(
        timestep=0.04, viscosity=0.3, curvature=1
    ) == pytest.approx(0.9936)
    assert viscous_qsd_coercivity_margin(
        timestep=0.04, viscosity=0.3, curvature=1, row_normalized=True
    ) == pytest.approx(0.9876)
    with pytest.raises(ValueError, match="bandwidth"):
        gaussian_viscous_force(torch.zeros((2, 1)), torch.ones((2, 1)), viscosity=0.1, bandwidth=0)
    with pytest.raises(ValueError, match="real"):
        fermionic_spectral_diagnostics(torch.eye(2, dtype=torch.complex128))


@pytest.mark.parametrize("row_normalized", [False, True])
def test_native_color_invariant_has_the_two_explicit_variance_centers(row_normalized):
    """Check the actual paired B2 force/velocity map, not a substitute color law."""
    first_drift = torch.tensor(
        [[0.1, 0.3, -0.2], [0.2, -0.4, 0.0], [-0.5, 0.2, 0.1], [0.1, 0.1, 0.1]],
        dtype=torch.float64,
    )
    phase_scale = 1.7
    speed = torch.pi / phase_scale
    values = []
    for transverse_speed in [0.0, 2 * speed]:
        ou_velocity = torch.tensor(
            [[0.0, 0.0, 0.0]] + [[speed, transverse_speed, 0.0]] * 3,
            dtype=torch.float64,
        )
        second_drift = first_drift + 0.02 * ou_velocity
        force = gaussian_viscous_force(
            second_drift,
            ou_velocity,
            viscosity=0.3,
            bandwidth=1.0,
            row_normalized=row_normalized,
        )
        colors = force / force.norm(dim=1, keepdim=True) * (1j * phase_scale * ou_velocity).exp()
        values.append(float((colors[0].conj() * colors[1]).sum().abs().square()))
    assert values == pytest.approx([1.0, 9 / 25])


def test_moment_budget_controls_actual_noisy_two_kick_samples():
    """Exercise both intermediate uncapped velocities against the explicit envelope."""
    generator = torch.Generator().manual_seed(42)
    n_walkers = 5
    repetitions = 64
    dt, nu, q, s, cap = 0.04, 0.3, 0.2, 0.1, 2.0
    ou_coefficient = 0.95
    outputs = []
    for _ in range(repetitions):
        x = torch.rand((n_walkers, 3), generator=generator, dtype=torch.float64) - 0.5
        v = torch.randn((n_walkers, 3), generator=generator, dtype=torch.float64)
        v = cap * v / (cap + v.norm(dim=1, keepdim=True))
        v1 = v + dt / 2 * (-x + gaussian_viscous_force(x, v, viscosity=nu, bandwidth=1.0))
        x1 = x + dt / 2 * v1
        v2 = ou_coefficient * v1 + q * torch.randn(
            v.shape, generator=generator, dtype=torch.float64
        )
        x2 = x1 + dt / 2 * v2
        v3 = v2 + dt / 2 * (-x2 + gaussian_viscous_force(x2, v2, viscosity=nu, bandwidth=1.0))
        xout = x2 + s * torch.randn(x.shape, generator=generator, dtype=torch.float64)
        vout = cap * v3 / (cap + v3.norm(dim=1, keepdim=True))
        outputs.append((xout.square().sum(dim=1).mean(), vout.square().sum(dim=1).mean()))
    budget = viscous_kinetic_moment_budget(
        dimension=3,
        moment_order=2,
        timestep=dt,
        viscosity=nu,
        force_lipschitz=1.0,
        force_at_origin=0.0,
        donor_radius=1.0,
        clone_jitter=0.0,
        restitution=0.0,
        velocity_cap=cap,
        ou_coefficient=ou_coefficient,
        ou_amplitude=q,
        position_amplitude=s,
    )
    means = torch.stack([torch.stack(pair) for pair in outputs]).mean(dim=0)
    assert float(means[0]) < budget.output_position**2
    assert float(means[1]) < budget.output_velocity**2
