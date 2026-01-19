import pytest
import torch

from fragile.core.layers import (
    CovariantRetina,
    IsotropicBlock,
    NormGate,
    NormGatedGELU,
    SpectralLinear,
)


def _random_rotation(dim: int) -> torch.Tensor:
    mat = torch.randn(dim, dim)
    q, _ = torch.linalg.qr(mat)
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_spectral_linear_shapes_and_norm() -> None:
    torch.manual_seed(0)
    layer = SpectralLinear(8, 6, bias=False, n_power_iterations=5)
    x = torch.randn(4, 8)
    out = layer(x)

    weight = layer._spectral_normalized_weight(update_u=False)
    _, s, _ = torch.linalg.svd(weight, full_matrices=False)

    assert out.shape == (4, 6)
    assert s[0].item() <= 1.01


def test_norm_gate_equivariance_single_bundle() -> None:
    torch.manual_seed(1)
    gate = NormGate(bundle_size=4, n_bundles=1)
    x = torch.randn(5, 4)
    rot = _random_rotation(4)

    x_rot = x @ rot.t()
    y_rot = gate(x_rot)
    y = gate(x) @ rot.t()

    assert torch.allclose(y_rot, y, atol=1e-5)


def test_norm_gated_gelu_shape() -> None:
    torch.manual_seed(2)
    gate = NormGatedGELU(bundle_size=3, n_bundles=2)
    x = torch.randn(4, 6)
    out = gate(x)

    assert out.shape == (4, 6)


def test_isotropic_block_exact_equivariance() -> None:
    torch.manual_seed(3)
    block = IsotropicBlock(in_dim=8, out_dim=8, bundle_size=4, exact=True)
    block.eval()

    x = torch.randn(6, 8)
    rot = torch.block_diag(_random_rotation(4), _random_rotation(4))

    y_rot = block(x @ rot.t())
    y = block(x) @ rot.t()

    assert torch.allclose(y_rot, y, atol=1e-5)


def test_isotropic_block_approximate_shapes() -> None:
    torch.manual_seed(4)
    block = IsotropicBlock(in_dim=10, out_dim=12, bundle_size=4, exact=False)
    x = torch.randn(2, 10)
    out = block(x)

    assert out.shape == (2, 12)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("norm_nonlinearity", ["n_sigmoid", "n_relu"])
def test_covariant_retina_optional(norm_nonlinearity: str) -> None:
    pytest.importorskip("e2cnn")
    torch.manual_seed(5)
    retina = CovariantRetina(
        in_channels=3,
        out_dim=32,
        num_rotations=8,
        kernel_size=3,
        use_reflections=False,
        norm_nonlinearity=norm_nonlinearity,
    )
    img = torch.randn(2, 3, 32, 32)
    out = retina(img)

    assert out.shape == (2, 32)


def test_covariant_retina_rotation_invariance() -> None:
    pytest.importorskip("e2cnn")
    torch.manual_seed(6)
    retina = CovariantRetina(
        in_channels=3,
        out_dim=16,
        num_rotations=4,
        kernel_size=3,
        use_reflections=False,
        norm_nonlinearity="n_sigmoid",
    )
    retina.eval()

    img = torch.randn(1, 3, 32, 32)
    out = retina(img)
    out_rot = retina(torch.rot90(img, k=1, dims=(2, 3)))

    max_diff = (out - out_rot).abs().max().item()
    assert max_diff < 5e-2
