"""Tests for SwarmState class."""

import pytest
import torch

from fragile.euclidean_gas import SwarmState


class TestSwarmState:
    """Tests for SwarmState."""

    def test_initialization(self):
        """Test basic initialization."""
        x = torch.randn(10, 3)
        v = torch.randn(10, 3)
        state = SwarmState(x, v)

        assert state.N == 10
        assert state.d == 3
        assert torch.equal(state.x, x)
        assert torch.equal(state.v, v)

    def test_mismatched_shapes(self):
        """Test that initialization fails with mismatched shapes."""
        x = torch.randn(10, 3)
        v = torch.randn(10, 4)

        with pytest.raises(AssertionError):
            SwarmState(x, v)

    def test_wrong_dimensions(self):
        """Test that initialization fails with wrong tensor dimensions."""
        x = torch.randn(10)  # 1D tensor
        v = torch.randn(10)

        with pytest.raises(AssertionError):
            SwarmState(x, v)

    def test_N_property(self):
        """Test N property returns number of walkers."""
        x = torch.randn(15, 5)
        v = torch.randn(15, 5)
        state = SwarmState(x, v)

        assert state.N == 15

    def test_d_property(self):
        """Test d property returns spatial dimension."""
        x = torch.randn(10, 7)
        v = torch.randn(10, 7)
        state = SwarmState(x, v)

        assert state.d == 7

    def test_device_property_cpu(self):
        """Test device property on CPU."""
        x = torch.randn(10, 3)
        v = torch.randn(10, 3)
        state = SwarmState(x, v)

        assert state.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_property_cuda(self):
        """Test device property on CUDA."""
        x = torch.randn(10, 3, device="cuda")
        v = torch.randn(10, 3, device="cuda")
        state = SwarmState(x, v)

        assert state.device.type == "cuda"

    def test_dtype_property(self):
        """Test dtype property."""
        x = torch.randn(10, 3, dtype=torch.float32)
        v = torch.randn(10, 3, dtype=torch.float32)
        state = SwarmState(x, v)

        assert state.dtype == torch.float32

    def test_clone(self):
        """Test cloning creates independent copy."""
        x = torch.randn(10, 3)
        v = torch.randn(10, 3)
        state1 = SwarmState(x, v)
        state2 = state1.clone()

        # Check values are equal
        assert torch.equal(state1.x, state2.x)
        assert torch.equal(state1.v, state2.v)

        # Check they are independent
        state2.x[0, 0] = 999.0
        assert not torch.equal(state1.x, state2.x)

    @pytest.mark.parametrize("N,d", [
        (5, 2),
        (100, 3),
        (1, 10),
        (50, 1),
    ])
    def test_various_dimensions(self, N, d):
        """Test state creation with various dimensions."""
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        state = SwarmState(x, v)

        assert state.N == N
        assert state.d == d

    def test_single_walker(self):
        """Test edge case with single walker."""
        x = torch.randn(1, 5)
        v = torch.randn(1, 5)
        state = SwarmState(x, v)

        assert state.N == 1
        assert state.d == 5

    def test_high_dimension(self):
        """Test with high dimensional state."""
        x = torch.randn(10, 100)
        v = torch.randn(10, 100)
        state = SwarmState(x, v)

        assert state.N == 10
        assert state.d == 100

    def test_dtype_consistency(self):
        """Test that x and v can have same dtype."""
        x = torch.randn(10, 3, dtype=torch.float64)
        v = torch.randn(10, 3, dtype=torch.float64)
        state = SwarmState(x, v)

        assert state.x.dtype == torch.float64
        assert state.v.dtype == torch.float64

    def test_state_modification(self):
        """Test that state can be modified in place."""
        x = torch.randn(10, 3)
        v = torch.randn(10, 3)
        state = SwarmState(x, v)

        original_x = state.x.clone()
        state.x += 1.0

        assert not torch.equal(state.x, original_x)
        assert torch.allclose(state.x, original_x + 1.0)

    def test_zero_state(self):
        """Test initialization with zero state."""
        x = torch.zeros(5, 2)
        v = torch.zeros(5, 2)
        state = SwarmState(x, v)

        assert torch.all(state.x == 0)
        assert torch.all(state.v == 0)
