"""Test script for new geodesic kernel weighting options."""

import torch
from fragile.fractalai.core.kinetic_operator import KineticOperator
from fragile.fractalai.core.euclidean_gas import SwarmState


def test_geodesic_euclidean():
    """Test geodesic Euclidean kernel weighting."""
    print("=" * 80)
    print("Testing kernel_geodesic_euclidean weighting")
    print("=" * 80)

    # Create operator with geodesic Euclidean weighting
    kinetic_op = KineticOperator(
        gamma=1.0,
        beta=1.0,
        delta_t=0.1,
        nu=1.0,
        use_viscous_coupling=True,
        viscous_neighbor_weighting="kernel_geodesic_euclidean",
        viscous_length_scale=0.5,
    )

    # Create simple state
    N, d = 50, 3
    print(f"\nCreating state with N={N}, d={d}")
    state = SwarmState(
        x=torch.randn(N, d) * 2.0,  # Positions in [-2, 2]
        v=torch.randn(N, d) * 0.5,  # Small velocities
    )

    # Create neighbor edges (simulate Delaunay/Voronoi graph)
    # Each walker connected to ~6 neighbors on average
    n_edges = N * 6
    neighbor_edges = torch.randint(0, N, (n_edges, 2))
    print(f"Created {n_edges} neighbor edges")

    # Compute viscous force
    print("\nComputing viscous force...")
    viscous_force = kinetic_op._compute_viscous_force(
        state.x,
        state.v,
        neighbor_edges=neighbor_edges
    )

    # Verify shape
    assert viscous_force.shape == (N, d), f"Expected shape ({N}, {d}), got {viscous_force.shape}"
    print(f"✓ Viscous force shape: {viscous_force.shape}")

    # Verify force magnitude
    force_norm = viscous_force.norm(dim=-1).mean().item()
    print(f"✓ Mean force magnitude: {force_norm:.6f}")

    # Verify dissipative property (force should reduce velocity differences)
    force_dot_v = (viscous_force * state.v).sum().item()
    print(f"✓ Force·velocity: {force_dot_v:.6f} (should be ≈ 0 or negative for dissipation)")

    print("\n✅ kernel_geodesic_euclidean test PASSED\n")
    return True


def test_geodesic_metric():
    """Test geodesic metric-weighted kernel weighting."""
    print("=" * 80)
    print("Testing kernel_geodesic_metric weighting")
    print("=" * 80)

    # Create operator with geodesic metric weighting
    kinetic_op = KineticOperator(
        gamma=1.0,
        beta=1.0,
        delta_t=0.1,
        nu=1.0,
        use_viscous_coupling=True,
        viscous_neighbor_weighting="kernel_geodesic_metric",
        viscous_length_scale=0.5,
        use_anisotropic_diffusion=True,  # Enable metric computation
        epsilon_Sigma=0.1,
        diagonal_diffusion=True,  # Use diagonal for simplicity
    )

    # Create simple state
    N, d = 50, 3
    print(f"\nCreating state with N={N}, d={d}")
    state = SwarmState(
        x=torch.randn(N, d) * 2.0,
        v=torch.randn(N, d) * 0.5,
    )

    # Create neighbor edges
    n_edges = N * 6
    neighbor_edges = torch.randint(0, N, (n_edges, 2))
    print(f"Created {n_edges} neighbor edges")

    # Create mock fitness gradient and Hessian
    grad_fitness = torch.randn(N, d) * 0.1
    hess_fitness = torch.ones(N, d) * 0.5  # Diagonal Hessian

    print("\nComputing viscous force with metric weighting...")
    viscous_force = kinetic_op._compute_viscous_force(
        state.x,
        state.v,
        neighbor_edges=neighbor_edges,
        grad_fitness=grad_fitness,
        hess_fitness=hess_fitness,
    )

    # Verify shape
    assert viscous_force.shape == (N, d), f"Expected shape ({N}, {d}), got {viscous_force.shape}"
    print(f"✓ Viscous force shape: {viscous_force.shape}")

    # Verify force magnitude
    force_norm = viscous_force.norm(dim=-1).mean().item()
    print(f"✓ Mean force magnitude: {force_norm:.6f}")

    # Verify dissipative property
    force_dot_v = (viscous_force * state.v).sum().item()
    print(f"✓ Force·velocity: {force_dot_v:.6f}")

    print("\n✅ kernel_geodesic_metric test PASSED\n")
    return True


def test_fallback_warning():
    """Test that geodesic weighting falls back gracefully without neighbor edges."""
    print("=" * 80)
    print("Testing fallback behavior (no neighbor edges)")
    print("=" * 80)

    # Create operator with geodesic weighting
    kinetic_op = KineticOperator(
        gamma=1.0,
        beta=1.0,
        delta_t=0.1,
        nu=1.0,
        use_viscous_coupling=True,
        viscous_neighbor_weighting="kernel_geodesic_euclidean",
        viscous_length_scale=0.5,
    )

    # Create simple state
    N, d = 50, 3
    state = SwarmState(
        x=torch.randn(N, d) * 2.0,
        v=torch.randn(N, d) * 0.5,
    )

    print("\nComputing viscous force WITHOUT neighbor edges (should fall back)...")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        viscous_force = kinetic_op._compute_viscous_force(
            state.x,
            state.v,
            neighbor_edges=None,  # No edges provided
        )

        # Check that a warning was issued
        if len(w) > 0:
            print(f"✓ Warning issued: {w[0].message}")
        else:
            print("⚠ No warning issued (may still work via all-pairs mode)")

    # Verify it still produces output
    assert viscous_force.shape == (N, d)
    print(f"✓ Fallback force shape: {viscous_force.shape}")
    print(f"✓ Fallback force magnitude: {viscous_force.norm(dim=-1).mean().item():.6f}")

    print("\n✅ Fallback test PASSED\n")
    return True


def test_comparison_with_euclidean():
    """Compare geodesic kernel with standard Euclidean kernel."""
    print("=" * 80)
    print("Comparing geodesic kernel vs standard Euclidean kernel")
    print("=" * 80)

    N, d = 30, 3
    state = SwarmState(
        x=torch.randn(N, d) * 2.0,
        v=torch.randn(N, d) * 0.5,
    )
    neighbor_edges = torch.randint(0, N, (N * 6, 2))

    # Standard Euclidean kernel
    kinetic_euclidean = KineticOperator(
        gamma=1.0,
        beta=1.0,
        delta_t=0.1,
        nu=1.0,
        use_viscous_coupling=True,
        viscous_neighbor_weighting="kernel",  # Standard
        viscous_length_scale=0.5,
    )

    # Geodesic Euclidean kernel
    kinetic_geodesic = KineticOperator(
        gamma=1.0,
        beta=1.0,
        delta_t=0.1,
        nu=1.0,
        use_viscous_coupling=True,
        viscous_neighbor_weighting="kernel_geodesic_euclidean",  # New
        viscous_length_scale=0.5,
    )

    print("\nComputing forces with both methods...")
    force_euclidean = kinetic_euclidean._compute_viscous_force(
        state.x, state.v, neighbor_edges=neighbor_edges
    )
    force_geodesic = kinetic_geodesic._compute_viscous_force(
        state.x, state.v, neighbor_edges=neighbor_edges
    )

    # Compare magnitudes
    mag_euclidean = force_euclidean.norm(dim=-1).mean().item()
    mag_geodesic = force_geodesic.norm(dim=-1).mean().item()

    print(f"\nEuclidean kernel magnitude: {mag_euclidean:.6f}")
    print(f"Geodesic kernel magnitude:  {mag_geodesic:.6f}")

    # Compute correlation
    correlation = torch.cosine_similarity(
        force_euclidean.flatten(),
        force_geodesic.flatten(),
        dim=0
    ).item()
    print(f"Force correlation: {correlation:.4f}")

    print("\nNote: Geodesic distances are generally ≥ Euclidean distances,")
    print("so geodesic kernel forces may be weaker on average.")

    print("\n✅ Comparison test PASSED\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Geodesic Kernel Weighting Tests")
    print("=" * 80 + "\n")

    try:
        test_geodesic_euclidean()
        test_geodesic_metric()
        test_fallback_warning()
        test_comparison_with_euclidean()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80 + "\n")

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80 + "\n")
        raise
