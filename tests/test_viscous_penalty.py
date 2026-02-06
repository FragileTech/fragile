import pytest
import torch

from fragile.fractalai.core.kinetic_operator import KineticOperator


def _make_setup():
    """Create 3-walker 2D setup with bidirectional edges."""
    x = torch.tensor(
        [[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]],
        dtype=torch.float64,
    )
    v = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float64,
    )
    # Bidirectional edges: 0-1, 0-2, 1-2
    neighbor_edges = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
        dtype=torch.long,
    )
    # Uniform pre-normalized weights (each node has degree 2)
    edge_weights = torch.tensor(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        dtype=torch.float64,
    )
    return x, v, neighbor_edges, edge_weights


BASE_PARAMS = {
    "gamma": 1.0,
    "beta": 1.0,
    "delta_t": 0.1,
    "nu": 1.0,
    "use_viscous_coupling": True,
    "use_potential_force": False,
    "viscous_length_scale": 1.0,
    "viscous_neighbor_threshold": 0.5,
}


def test_precomputed_weights():
    """Viscous force works with precomputed edge weights."""
    x, v, neighbor_edges, edge_weights = _make_setup()
    kin = KineticOperator(**BASE_PARAMS, viscous_neighbor_weighting="inverse_riemannian_distance")
    force = kin._compute_viscous_force(x, v, neighbor_edges=neighbor_edges, edge_weights=edge_weights)
    assert force.shape == v.shape
    assert torch.norm(force) > 0


def test_kernel_weights():
    """Viscous force works with kernel weighting (computed on-the-fly)."""
    x, v, neighbor_edges, _ = _make_setup()
    kin = KineticOperator(**BASE_PARAMS, viscous_neighbor_weighting="kernel")
    force = kin._compute_viscous_force(x, v, neighbor_edges=neighbor_edges)
    assert force.shape == v.shape
    assert torch.norm(force) > 0


def test_threshold_penalty_reduces_force():
    """Threshold penalty should reduce force magnitude."""
    x, v, neighbor_edges, edge_weights = _make_setup()
    kin_no = KineticOperator(
        **BASE_PARAMS,
        viscous_neighbor_weighting="inverse_riemannian_distance",
        viscous_neighbor_penalty=0.0,
    )
    kin_pen = KineticOperator(
        **BASE_PARAMS,
        viscous_neighbor_weighting="inverse_riemannian_distance",
        viscous_neighbor_penalty=1.0,
    )
    force_no = kin_no._compute_viscous_force(x, v, neighbor_edges=neighbor_edges, edge_weights=edge_weights)
    force_pen = kin_pen._compute_viscous_force(x, v, neighbor_edges=neighbor_edges, edge_weights=edge_weights)
    assert torch.norm(force_pen) < torch.norm(force_no)


def test_no_edges_raises():
    """Calling without neighbor_edges should raise ValueError."""
    x, v, _, _ = _make_setup()
    kin = KineticOperator(**BASE_PARAMS, viscous_neighbor_weighting="inverse_riemannian_distance")
    with pytest.raises(ValueError, match="neighbor_edges required"):
        kin._compute_viscous_force(x, v)


def test_precomputed_without_edge_weights_raises():
    """Mode requiring precomputation without edge_weights should raise ValueError."""
    x, v, neighbor_edges, _ = _make_setup()
    kin = KineticOperator(**BASE_PARAMS, viscous_neighbor_weighting="inverse_riemannian_volume")
    with pytest.raises(ValueError, match="requires precomputed edge_weights"):
        kin._compute_viscous_force(x, v, neighbor_edges=neighbor_edges)


def test_disabled_returns_zero():
    """Disabled viscous coupling returns zero force."""
    x, v, neighbor_edges, edge_weights = _make_setup()
    params = {**BASE_PARAMS, "nu": 0.0}
    kin = KineticOperator(**params, viscous_neighbor_weighting="inverse_riemannian_distance")
    force = kin._compute_viscous_force(x, v, neighbor_edges=neighbor_edges, edge_weights=edge_weights)
    assert torch.allclose(force, torch.zeros_like(v))
