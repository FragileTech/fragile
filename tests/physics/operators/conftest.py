"""Shared fixtures for physics/operators test suite."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.operators.preparation import PreparedChannelData


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------


@pytest.fixture
def T() -> int:
    """Number of time frames."""
    return 10


@pytest.fixture
def N() -> int:
    """Number of walkers."""
    return 20


# ---------------------------------------------------------------------------
# Synthetic color states
# ---------------------------------------------------------------------------


@pytest.fixture
def color_states(T: int, N: int) -> Tensor:
    """Random complex color states [T, N, 3], seeded."""
    gen = torch.Generator().manual_seed(42)
    real = torch.randn(T, N, 3, generator=gen)
    imag = torch.randn(T, N, 3, generator=gen)
    c = torch.complex(real, imag)
    # Normalize to unit vectors
    norms = c.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return c / norms


@pytest.fixture
def color_valid(T: int, N: int) -> Tensor:
    """Mostly-valid color mask [T, N]."""
    mask = torch.ones(T, N, dtype=torch.bool)
    # Mark a few walkers invalid
    mask[:, -1] = False
    return mask


# ---------------------------------------------------------------------------
# Companion indices
# ---------------------------------------------------------------------------


@pytest.fixture
def companions_distance(T: int, N: int) -> Tensor:
    """Companion distance indices [T, N] — cyclic shift by 1."""
    return torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).clone()


@pytest.fixture
def companions_clone(T: int, N: int) -> Tensor:
    """Companion clone indices [T, N] — cyclic shift by 2."""
    return torch.arange(N).roll(-2).unsqueeze(0).expand(T, -1).clone()


# ---------------------------------------------------------------------------
# Positions and scores
# ---------------------------------------------------------------------------


@pytest.fixture
def positions_3d(T: int, N: int) -> Tensor:
    """Random 3D positions [T, N, 3], seeded."""
    gen = torch.Generator().manual_seed(99)
    return torch.randn(T, N, 3, generator=gen)


@pytest.fixture
def scores(T: int, N: int) -> Tensor:
    """Random cloning scores [T, N], seeded."""
    gen = torch.Generator().manual_seed(100)
    return torch.randn(T, N, generator=gen)


@pytest.fixture
def positions_axis(T: int, N: int) -> Tensor:
    """Random 1D positions for momentum projection [T, N], seeded."""
    gen = torch.Generator().manual_seed(101)
    return torch.randn(T, N, generator=gen) * 5.0


# ---------------------------------------------------------------------------
# PreparedChannelData factories
# ---------------------------------------------------------------------------


def make_prepared_data(
    T: int = 10,
    N: int = 20,
    *,
    include_positions: bool = False,
    include_scores: bool = False,
    include_momentum_axis: bool = False,
    include_multiscale: bool = False,
    n_scales: int = 3,
    seed: int = 42,
) -> PreparedChannelData:
    """Create a synthetic PreparedChannelData for testing.

    This factory builds all tensors from scratch without needing RunHistory.
    """
    gen = torch.Generator().manual_seed(seed)
    device = torch.device("cpu")

    # Color states
    real = torch.randn(T, N, 3, generator=gen)
    imag = torch.randn(T, N, 3, generator=gen)
    color = torch.complex(real, imag)
    norms = color.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
    color = color / norms

    # Validity
    color_valid = torch.ones(T, N, dtype=torch.bool)
    color_valid[:, -1] = False  # last walker invalid

    # Companions (cyclic)
    comp_dist = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).clone()
    comp_clone = torch.arange(N).roll(-2).unsqueeze(0).expand(T, -1).clone()

    # Optional fields
    positions = None
    if include_positions:
        positions = torch.randn(T, N, 3, generator=gen)

    scores_t = None
    if include_scores:
        scores_t = torch.randn(T, N, generator=gen)

    positions_axis = None
    projection_length = None
    if include_momentum_axis:
        positions_axis = torch.randn(T, N, generator=gen) * 5.0
        projection_length = 10.0

    # Multiscale
    scales = None
    pairwise_distances = None
    kernels = None
    companion_d_ij = None
    companion_d_ik = None
    companion_d_jk = None
    if include_multiscale:
        scales = torch.linspace(0.5, 5.0, n_scales)
        # Synthetic distance matrix from positions
        if positions is not None:
            pos = positions
        else:
            pos = torch.randn(T, N, 3, generator=gen)
        diffs = pos.unsqueeze(2) - pos.unsqueeze(1)  # [T, N, N, 3]
        pairwise_distances = diffs.norm(dim=-1)  # [T, N, N]
        pairwise_distances.diagonal(dim1=1, dim2=2).zero_()

        # Companion distances
        t_idx = torch.arange(T).unsqueeze(1).expand(T, N)
        i_idx = torch.arange(N).unsqueeze(0).expand(T, N)
        j_idx = comp_dist.clamp(0, N - 1)
        k_idx = comp_clone.clamp(0, N - 1)
        companion_d_ij = pairwise_distances[t_idx, i_idx, j_idx]
        companion_d_ik = pairwise_distances[t_idx, i_idx, k_idx]
        companion_d_jk = pairwise_distances[t_idx, j_idx, k_idx]

    return PreparedChannelData(
        color=color,
        color_valid=color_valid,
        companions_distance=comp_dist,
        companions_clone=comp_clone,
        scores=scores_t,
        positions=positions,
        positions_axis=positions_axis,
        projection_length=projection_length,
        frame_indices=list(range(1, T + 1)),
        device=device,
        eps=1e-12,
        scales=scales,
        pairwise_distances=pairwise_distances,
        kernels=kernels,
        companion_d_ij=companion_d_ij,
        companion_d_ik=companion_d_ik,
        companion_d_jk=companion_d_jk,
    )


@pytest.fixture
def prepared_data_basic(T: int, N: int) -> PreparedChannelData:
    """Basic PreparedChannelData without optional fields."""
    return make_prepared_data(T, N)


@pytest.fixture
def prepared_data_with_positions(T: int, N: int) -> PreparedChannelData:
    """PreparedChannelData with positions for vector/tensor ops."""
    return make_prepared_data(T, N, include_positions=True)


@pytest.fixture
def prepared_data_with_scores(T: int, N: int) -> PreparedChannelData:
    """PreparedChannelData with scores for score-directed modes."""
    return make_prepared_data(T, N, include_scores=True)


@pytest.fixture
def prepared_data_full(T: int, N: int) -> PreparedChannelData:
    """PreparedChannelData with positions, scores, and momentum axis."""
    return make_prepared_data(
        T,
        N,
        include_positions=True,
        include_scores=True,
        include_momentum_axis=True,
    )


@pytest.fixture
def prepared_data_multiscale(T: int, N: int) -> PreparedChannelData:
    """PreparedChannelData with multiscale fields populated."""
    return make_prepared_data(
        T,
        N,
        include_positions=True,
        include_scores=True,
        include_momentum_axis=True,
        include_multiscale=True,
        n_scales=3,
    )


# ---------------------------------------------------------------------------
# Simple graph fixtures for geodesics
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_edges() -> Tensor:
    """Simple chain graph edges [E, 2] for 5 nodes: 0-1-2-3-4."""
    edges = []
    for i in range(4):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    return torch.tensor(edges, dtype=torch.long)


@pytest.fixture
def simple_edge_weights(simple_edges: Tensor) -> Tensor:
    """Uniform weights for simple_edges."""
    return torch.ones(simple_edges.shape[0], dtype=torch.float32)


@pytest.fixture
def triangle_edges() -> Tensor:
    """Triangle graph: 0-1, 1-2, 0-2."""
    return torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [0, 2], [2, 0]],
        dtype=torch.long,
    )


@pytest.fixture
def distance_matrix_5x5() -> Tensor:
    """Known distance matrix for a 5-node chain with unit weights."""
    # Chain: 0-1-2-3-4
    d = torch.full((5, 5), float("inf"))
    for i in range(5):
        d[i, i] = 0.0
        for j in range(5):
            d[i, j] = float(abs(i - j))
    return d
