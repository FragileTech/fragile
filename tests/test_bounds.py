import numpy as np
import pytest
import torch

from fragile.bounds import NumpyBounds, TorchBounds


@pytest.fixture(params=[TorchBounds, NumpyBounds])
def bounds(request):
    """
    Pytest fixture that returns an instance of Bounds (TorchBounds or NumpyBounds)
    for testing purposes.
    """
    bounds_class = request.param

    if bounds_class == TorchBounds:
        high = torch.tensor([5.0, 10.0, 15.0])
        low = torch.tensor([1.0, 2.0, 3.0])
        return bounds_class(high=high, low=low)
    if bounds_class == NumpyBounds:
        high = np.array([5.0, 10.0, 15.0])
        low = np.array([1.0, 2.0, 3.0])
        return bounds_class(high=high, low=low)
    return None


def test_sample(bounds):
    """
    Test the sample method to ensure it generates samples within the bounds.
    """
    num_samples = 10
    samples = bounds.sample(num_samples=num_samples)

    # Check the shape of the samples
    assert samples.shape == (num_samples, *bounds.shape)

    # Verify that all samples are within bounds
    assert bounds.contains(samples).all()


def test_clip(bounds):
    """
    Test the clip method to ensure it correctly clips values outside the bounds.
    """
    if isinstance(bounds, TorchBounds):
        x = torch.tensor([[0.0, 5.0, 20.0]])
        clipped = bounds.clip(x)
        # Check that clipped values are within bounds
        assert torch.all(clipped >= bounds.low)
        assert torch.all(clipped <= bounds.high)
    else:
        x = np.array([[0.0, 5.0, 20.0]])
        clipped = bounds.clip(x)
        # Check that clipped values are within bounds
        assert np.all(clipped >= bounds.low)
        assert np.all(clipped <= bounds.high)


def test_contains(bounds):
    """
    Test the contains method to ensure it accurately identifies points within bounds.
    """
    # Generate samples within bounds
    samples = bounds.sample(num_samples=5)
    assert bounds.contains(samples).all()

    # Generate a point outside the bounds
    if isinstance(bounds, TorchBounds):
        outside_point = torch.tensor([[0.0, 0.0, 0.0]])
    else:
        outside_point = np.array([[0.0, 0.0, 0.0]])
    assert not bounds.contains(outside_point).all()


def test_pbc(bounds):
    """
    Test the periodic boundary conditions (pbc) method.
    """
    if isinstance(bounds, TorchBounds):
        x = torch.tensor([[6.0, 11.0, 16.0]])
        pbc_x = bounds.pbc(x)
        # Verify that pbc_x is within bounds
        assert bounds.contains(pbc_x).all()
    else:
        x = np.array([[6.0, 11.0, 16.0]])
        pbc_x = bounds.pbc(x)
        # Verify that pbc_x is within bounds
        assert bounds.contains(pbc_x).all()


def test_safe_margin(bounds):
    """
    Test the safe_margin method to ensure it scales bounds correctly.
    """
    scale = 1.1  # 10% increase
    scaled_bounds = bounds.safe_margin(scale=scale)

    # Check that the new bounds are appropriately scaled
    if isinstance(bounds, TorchBounds):
        assert torch.all(scaled_bounds.low <= bounds.low)
        assert torch.all(scaled_bounds.high >= bounds.high)
    else:
        assert np.all(scaled_bounds.low <= bounds.low)
        assert np.all(scaled_bounds.high >= bounds.high)


def test_to_tuples(bounds):
    """
    Test the to_tuples method to ensure it returns the correct tuples.
    """
    tuples = bounds.to_tuples()
    expected = tuple(zip(bounds.low, bounds.high))

    # Check that the tuples match the expected values
    if isinstance(bounds, TorchBounds):
        for (low_t, high_t), (low_e, high_e) in zip(tuples, expected):
            assert torch.equal(low_t, low_e)
            assert torch.equal(high_t, high_e)
    else:
        for (low_t, high_t), (low_e, high_e) in zip(tuples, expected):
            assert np.array_equal(low_t, low_e)
            assert np.array_equal(high_t, high_e)


def test_from_array():
    """
    Test the from_array class method for both TorchBounds and NumpyBounds.
    """
    # For NumpyBounds
    x_np = np.random.uniform(0, 10, size=(100, 3))
    bounds_np = NumpyBounds.from_array(x_np, scale=1.0)
    assert isinstance(bounds_np, NumpyBounds)
    assert bounds_np.low.shape == (3,)
    assert bounds_np.high.shape == (3,)

    # Verify that all data points are within the new bounds
    assert bounds_np.contains(x_np).all()

    # For TorchBounds
    x_torch = torch.rand(100, 3) * 10
    bounds_torch = TorchBounds.from_array(x_torch, scale=1.0)
    assert isinstance(bounds_torch, TorchBounds)
    assert bounds_torch.low.shape == (3,)
    assert bounds_torch.high.shape == (3,)

    # Verify that all data points are within the new bounds
    assert bounds_torch.contains(x_torch).all()


def test_get_scaled_intervals():
    """
    Test the get_scaled_intervals static method.
    """
    scale = 1.1  # 10% increase

    # For NumpyBounds
    low_np = np.array([1.0, 2.0, 3.0])
    high_np = np.array([5.0, 10.0, 15.0])
    xmin_scaled_np, xmax_scaled_np = NumpyBounds.get_scaled_intervals(low_np, high_np, scale)

    assert np.all(xmin_scaled_np <= low_np)
    assert np.all(xmax_scaled_np >= high_np)

    # For TorchBounds
    low_torch = torch.tensor([1.0, 2.0, 3.0])
    high_torch = torch.tensor([5.0, 10.0, 15.0])
    xmin_scaled_torch, xmax_scaled_torch = TorchBounds.get_scaled_intervals(
        low_torch, high_torch, scale
    )

    assert torch.all(xmin_scaled_torch <= low_torch)
    assert torch.all(xmax_scaled_torch >= high_torch)


def test_points_in_bounds(bounds):
    """
    Test the points_in_bounds method to ensure it accurately checks multiple points.
    """
    # Generate points within bounds
    if isinstance(bounds, TorchBounds):
        points = torch.tensor([[1.5, 5.0, 10.0], [4.5, 9.0, 14.0]])
    else:
        points = np.array([[1.5, 5.0, 10.0], [4.5, 9.0, 14.0]])
    assert bounds.points_in_bounds(points).all()

    # Generate points outside bounds
    if isinstance(bounds, TorchBounds):
        points_outside = torch.tensor([[0.0, 5.0, 10.0], [6.0, 9.0, 14.0]])
    else:
        points_outside = np.array([[0.0, 5.0, 10.0], [6.0, 9.0, 14.0]])
    assert not bounds.points_in_bounds(points_outside).all()


# ===== Additional TorchBounds Tests =====


def test_torchbounds_on_device(test_device):
    """Test TorchBounds works correctly on different devices."""
    high = torch.tensor([5.0, 10.0, 15.0])
    low = torch.tensor([1.0, 2.0, 3.0])
    bounds = TorchBounds(high=high, low=low, device=test_device)

    # Verify device
    assert bounds.high.device.type == test_device
    assert bounds.low.device.type == test_device

    # Test that samples work on the device
    samples = bounds.sample(num_samples=10)
    assert samples.device.type == test_device
    assert bounds.contains(samples).all()

    # Test clip
    x_out = torch.tensor([[0.0, 5.0, 20.0], [6.0, 11.0, 2.5]], device=test_device)
    x_clipped = bounds.clip(x_out)
    assert x_clipped.device.type == test_device
    assert bounds.contains(x_clipped).all()

    # Test pbc
    x_pbc = bounds.pbc(x_out)
    assert x_pbc.device.type == test_device
    assert bounds.contains(x_pbc).all()


def test_torchbounds_dtype_conversions():
    """Test TorchBounds with different dtypes (float32, float64)."""
    high = torch.tensor([5.0, 10.0, 15.0])
    low = torch.tensor([1.0, 2.0, 3.0])

    # Test float32
    bounds_f32 = TorchBounds(high=high, low=low, dtype=torch.float32)
    assert bounds_f32.dtype == torch.float32
    samples_f32 = bounds_f32.sample(num_samples=5)
    assert samples_f32.dtype == torch.float32
    assert bounds_f32.contains(samples_f32).all()

    # Test float64
    bounds_f64 = TorchBounds(high=high, low=low, dtype=torch.float64)
    assert bounds_f64.dtype == torch.float64
    samples_f64 = bounds_f64.sample(num_samples=5)
    assert samples_f64.dtype == torch.float64
    assert bounds_f64.contains(samples_f64).all()


def test_torchbounds_chained_operations():
    """Test chaining clip/pbc with contains."""
    high = torch.tensor([5.0, 10.0, 15.0])
    low = torch.tensor([1.0, 2.0, 3.0])
    bounds = TorchBounds(high=high, low=low)

    # Out-of-bounds points
    x_out = torch.tensor([[0.0, 5.0, 20.0], [6.0, 11.0, 2.5]])

    # Clip then check contains
    x_clipped = bounds.clip(x_out)
    assert bounds.contains(x_clipped).all()

    # PBC then check contains
    x_pbc = bounds.pbc(x_out)
    assert bounds.contains(x_pbc).all()


def test_torchbounds_with_euclidean_gas_integration():
    """Test TorchBounds integration with EuclideanGas alive_mask logic."""
    from fragile.core.euclidean_gas import SwarmState

    N, d = 10, 2
    high = torch.tensor([5.0, 5.0])
    low = torch.tensor([-5.0, -5.0])
    bounds = TorchBounds(high=high, low=low)

    # Create state with some walkers out of bounds
    x = torch.randn(N, d) * 10  # Large variance, some will be out of bounds
    v = torch.randn(N, d)
    state = SwarmState(x, v)

    # Test contains returns boolean mask
    alive_mask = bounds.contains(state.x)
    assert alive_mask.shape == (N,)
    assert alive_mask.dtype == torch.bool

    # Test that clipped positions are all in bounds
    x_clipped = bounds.clip(state.x)
    alive_mask_clipped = bounds.contains(x_clipped)
    assert alive_mask_clipped.all()


def test_torchbounds_edge_cases():
    """Test edge cases for TorchBounds."""
    # Very small bounds (near zero width)
    small_high = torch.tensor([0.001, 0.001])
    small_low = torch.tensor([0.0, 0.0])
    small_bounds = TorchBounds(high=small_high, low=small_low)
    small_samples = small_bounds.sample(num_samples=10)
    assert small_bounds.contains(small_samples).all()

    # Large bounds (wide range)
    large_high = torch.tensor([1e6, 1e6])
    large_low = torch.tensor([-1e6, -1e6])
    large_bounds = TorchBounds(high=large_high, low=large_low)
    large_samples = large_bounds.sample(num_samples=10)
    assert large_bounds.contains(large_samples).all()

    # Single dimension
    single_dim_bounds = TorchBounds(high=torch.tensor([10.0]), low=torch.tensor([0.0]))
    single_samples = single_dim_bounds.sample(num_samples=5)
    assert single_samples.shape == (5, 1)
    assert single_dim_bounds.contains(single_samples).all()
