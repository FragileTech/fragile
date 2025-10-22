"""
Tests for MixtureOfGaussians benchmark.

Tests cover:
- Random initialization
- Custom centers/stds/weights
- Gradient computation
- Optimal value at centers
- Reproducibility with seed
- Error handling for invalid parameters
"""

import numpy as np
import pytest
import torch

from fragile.core.benchmarks import MixtureOfGaussians


class TestMixtureOfGaussians:
    """Test suite for MixtureOfGaussians benchmark."""

    def test_random_initialization(self):
        """Test that random initialization works correctly."""
        mixture = MixtureOfGaussians(dims=3, n_gaussians=5, seed=42)

        assert mixture.dims == 3
        assert mixture.n_gaussians == 5
        assert mixture.centers.shape == (5, 3)
        assert mixture.stds.shape == (5, 3)
        assert mixture.weights.shape == (5,)

        # Check weights are normalized
        torch.testing.assert_close(mixture.weights.sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)

        # Check stds are positive
        assert (mixture.stds > 0).all()

    def test_custom_parameters(self):
        """Test initialization with custom centers, stds, and weights."""
        dims = 2
        n_gaussians = 3

        # Define custom parameters
        centers = torch.tensor([[0.0, 0.0], [5.0, 5.0], [-3.0, 2.0]])

        stds = torch.tensor([[1.0, 1.0], [0.5, 0.5], [2.0, 1.5]])

        weights = torch.tensor([0.5, 0.3, 0.2])

        mixture = MixtureOfGaussians(
            dims=dims, n_gaussians=n_gaussians, centers=centers, stds=stds, weights=weights
        )

        # Check parameters are stored correctly
        torch.testing.assert_close(mixture.centers, centers)
        torch.testing.assert_close(mixture.stds, stds)
        torch.testing.assert_close(mixture.weights, weights)

    def test_weights_normalization(self):
        """Test that weights are automatically normalized."""
        centers = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        stds = torch.ones(2, 2)
        weights = torch.tensor([2.0, 3.0])  # Not normalized

        mixture = MixtureOfGaussians(
            dims=2, n_gaussians=2, centers=centers, stds=stds, weights=weights
        )

        # Should be normalized to [0.4, 0.6]
        expected_weights = torch.tensor([0.4, 0.6])
        torch.testing.assert_close(mixture.weights, expected_weights, atol=1e-6, rtol=1e-6)

    def test_evaluation_shape(self):
        """Test that function evaluation returns correct shape."""
        mixture = MixtureOfGaussians(dims=3, n_gaussians=4, seed=42)

        # Evaluate at multiple points
        x = torch.randn(10, 3)
        values = mixture(x)

        assert values.shape == (10,)
        assert torch.isfinite(values).all()

    def test_minimum_at_best_center(self):
        """Test that minimum is at the center of highest-weighted Gaussian."""
        centers = torch.tensor([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])

        stds = torch.ones(3, 2) * 0.5

        # Highest weight for the first Gaussian
        weights = torch.tensor([0.6, 0.3, 0.1])

        mixture = MixtureOfGaussians(
            dims=2, n_gaussians=3, centers=centers, stds=stds, weights=weights
        )

        # Best state should be the first center
        torch.testing.assert_close(mixture.best_state, centers[0])

        # Evaluate at all centers
        values_at_centers = mixture(centers)

        # Value at best center should be smallest (or very close)
        best_value = values_at_centers[0]
        assert best_value <= values_at_centers[1] + 1e-3
        assert best_value <= values_at_centers[2] + 1e-3

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same random initialization."""
        mixture1 = MixtureOfGaussians(dims=4, n_gaussians=3, seed=123)
        mixture2 = MixtureOfGaussians(dims=4, n_gaussians=3, seed=123)

        torch.testing.assert_close(mixture1.centers, mixture2.centers)
        torch.testing.assert_close(mixture1.stds, mixture2.stds)
        torch.testing.assert_close(mixture1.weights, mixture2.weights)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different initializations."""
        mixture1 = MixtureOfGaussians(dims=3, n_gaussians=2, seed=1)
        mixture2 = MixtureOfGaussians(dims=3, n_gaussians=2, seed=2)

        # Should be different
        assert not torch.allclose(mixture1.centers, mixture2.centers)

    def test_bounds(self):
        """Test that bounds are set correctly."""
        bounds_range = (-5.0, 15.0)
        mixture = MixtureOfGaussians(dims=3, n_gaussians=2, bounds_range=bounds_range, seed=42)

        assert mixture.bounds_range == bounds_range

        # Sample from bounds and check they're in range
        samples = mixture.sample(100)
        assert (samples >= -5.0).all()
        assert (samples <= 15.0).all()

    def test_invalid_centers_shape(self):
        """Test that invalid centers shape raises error."""
        centers = torch.randn(3, 2)  # Wrong shape

        with pytest.raises(ValueError, match="Centers shape"):
            MixtureOfGaussians(
                dims=3,  # dims=3 but centers have dims=2
                n_gaussians=3,
                centers=centers,
            )

    def test_invalid_stds_shape(self):
        """Test that invalid stds shape raises error."""
        stds = torch.rand(2, 3)  # Wrong shape

        with pytest.raises(ValueError, match="Stds shape"):
            MixtureOfGaussians(
                dims=3,
                n_gaussians=3,  # n_gaussians=3 but stds have n_gaussians=2
                stds=stds,
            )

    def test_invalid_stds_values(self):
        """Test that non-positive stds raise error."""
        stds = torch.tensor([[1.0, -0.5], [0.5, 1.0]])  # Negative std

        with pytest.raises(ValueError, match="positive"):
            MixtureOfGaussians(dims=2, n_gaussians=2, stds=stds)

    def test_invalid_weights_shape(self):
        """Test that invalid weights shape raises error."""
        weights = torch.tensor([0.5, 0.3, 0.2])  # Wrong shape

        with pytest.raises(ValueError, match="Weights shape"):
            MixtureOfGaussians(
                dims=2,
                n_gaussians=2,  # n_gaussians=2 but weights have length 3
                weights=weights,
            )

    def test_invalid_weights_values(self):
        """Test that negative weights raise error."""
        weights = torch.tensor([0.5, -0.3, 0.8])

        with pytest.raises(ValueError, match="non-negative"):
            MixtureOfGaussians(dims=2, n_gaussians=3, weights=weights)

    def test_numpy_input(self):
        """Test that numpy arrays can be used for parameters."""
        centers_np = np.array([[0.0, 0.0], [1.0, 1.0]])
        stds_np = np.ones((2, 2))
        weights_np = np.array([0.5, 0.5])

        mixture = MixtureOfGaussians(
            dims=2, n_gaussians=2, centers=centers_np, stds=stds_np, weights=weights_np
        )

        # Should be converted to tensors
        assert isinstance(mixture.centers, torch.Tensor)
        assert isinstance(mixture.stds, torch.Tensor)
        assert isinstance(mixture.weights, torch.Tensor)

    def test_device_compatibility(self):
        """Test that mixture works with different devices (if CUDA available)."""
        mixture = MixtureOfGaussians(dims=2, n_gaussians=3, seed=42)

        # Test on CPU
        x_cpu = torch.randn(5, 2)
        values_cpu = mixture(x_cpu)
        assert values_cpu.device.type == "cpu"

        # Test on CUDA if available
        if torch.cuda.is_available():
            x_cuda = x_cpu.cuda()
            values_cuda = mixture(x_cuda)
            assert values_cuda.device.type == "cuda"

            # Results should be close
            torch.testing.assert_close(values_cpu, values_cuda.cpu(), atol=1e-5, rtol=1e-5)

    def test_gradient_computation(self):
        """Test that gradients can be computed for the mixture function."""
        mixture = MixtureOfGaussians(dims=2, n_gaussians=2, seed=42)

        x = torch.randn(3, 2, requires_grad=True)
        values = mixture(x)

        # Should be able to compute gradients
        loss = values.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()

    def test_component_info(self):
        """Test get_component_info method."""
        centers = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        stds = torch.ones(2, 2) * 0.5
        weights = torch.tensor([0.4, 0.6])

        mixture = MixtureOfGaussians(
            dims=2, n_gaussians=2, centers=centers, stds=stds, weights=weights
        )

        info = mixture.get_component_info()

        assert info["n_gaussians"] == 2
        assert info["dims"] == 2
        torch.testing.assert_close(info["centers"], centers)
        torch.testing.assert_close(info["stds"], stds)
        torch.testing.assert_close(info["weights"], weights)

    def test_benchmark_property(self):
        """Test that benchmark property evaluates function at best state."""
        mixture = MixtureOfGaussians(dims=2, n_gaussians=3, seed=42)

        benchmark_value = mixture.benchmark

        # Should be the value at best_state
        best_state = mixture.best_state.unsqueeze(0)
        expected_value = mixture(best_state)[0]

        torch.testing.assert_close(benchmark_value, expected_value, atol=1e-6, rtol=1e-6)

    def test_single_gaussian(self):
        """Test mixture with single Gaussian (edge case)."""
        center = torch.tensor([[1.0, 2.0, 3.0]])
        std = torch.ones(1, 3)
        weight = torch.tensor([1.0])

        mixture = MixtureOfGaussians(
            dims=3, n_gaussians=1, centers=center, stds=std, weights=weight
        )

        # Should work like a single Gaussian
        x = torch.randn(5, 3)
        values = mixture(x)

        assert values.shape == (5,)
        assert torch.isfinite(values).all()

    def test_high_dimensional_mixture(self):
        """Test mixture in higher dimensions."""
        dims = 10
        n_gaussians = 5

        mixture = MixtureOfGaussians(dims=dims, n_gaussians=n_gaussians, seed=42)

        x = torch.randn(20, dims)
        values = mixture(x)

        assert values.shape == (20,)
        assert torch.isfinite(values).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
