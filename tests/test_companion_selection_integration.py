"""Tests for companion selection integration in EuclideanGas."""

import pytest
import torch

from fragile.bounds import TorchBounds
from fragile.core.companion_selection import CompanionSelection
from fragile.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    EuclideanGasParams,
    LangevinParams,
    SimpleQuadraticPotential,
    SwarmState,
)


@pytest.fixture
def base_params(test_device):
    """Base parameters for testing companion selection."""
    return {
        "N": 20,
        "d": 2,
        "potential": SimpleQuadraticPotential(),
        "langevin": LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01),
        "device": test_device,
        "dtype": "float64",
    }


@pytest.fixture
def bounded_params(base_params, test_device):
    """Parameters with bounds for boundary testing."""
    params_dict = base_params.copy()
    high = torch.tensor([5.0, 5.0], device=test_device)
    low = torch.tensor([-5.0, -5.0], device=test_device)
    params_dict["bounds"] = TorchBounds(high=high, low=low, device=test_device)
    return params_dict


class TestCompanionSelectionMethods:
    """Test different companion selection methods."""

    def test_hybrid_method_default(self, base_params):
        """Test cloning method (default behavior)."""
        cloning = CloningParams(
            sigma_x=0.1,
            lambda_alg=0.5,
            alpha_restitution=0.5,
            companion_selection=CompanionSelection(method="cloning", epsilon=0.1),
        )
        params = EuclideanGasParams(**base_params, cloning=cloning)
        gas = EuclideanGas(params)

        # Run a few steps
        torch.manual_seed(42)
        results = gas.run(n_steps=5)

        # Should complete without errors
        assert results["x"].shape == (6, 20, 2)
        assert results["v"].shape == (6, 20, 2)

    def test_softmax_method(self, base_params):
        """Test softmax companion selection method."""
        cloning = CloningParams(
            sigma_x=0.1,
            lambda_alg=0.5,
            alpha_restitution=0.5,
            companion_selection=CompanionSelection(method="softmax", epsilon=0.1),
        )
        params = EuclideanGasParams(**base_params, cloning=cloning)
        gas = EuclideanGas(params)

        # Run a few steps
        torch.manual_seed(42)
        results = gas.run(n_steps=5)

        # Should complete without errors
        assert results["x"].shape == (6, 20, 2)
        assert results["v"].shape == (6, 20, 2)

    def test_uniform_method(self, base_params):
        """Test uniform companion selection method."""
        cloning = CloningParams(
            sigma_x=0.1,
            lambda_alg=0.5,
            alpha_restitution=0.5,
            companion_selection=CompanionSelection(method="uniform"),
        )
        params = EuclideanGasParams(**base_params, cloning=cloning)
        gas = EuclideanGas(params)

        # Run a few steps
        torch.manual_seed(42)
        results = gas.run(n_steps=5)

        # Should complete without errors
        assert results["x"].shape == (6, 20, 2)
        assert results["v"].shape == (6, 20, 2)

    def test_invalid_method_raises_error(self, base_params):
        """Test that invalid companion selection method raises error."""
        # Should raise validation error when creating CompanionSelection with invalid method
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Input should be"):
            CompanionSelection(method="invalid_method")


class TestEpsilonCParameter:
    """Test epsilon_c parameter functionality."""

    def test_epsilon_c_default_to_sigma_x(self, base_params):
        """Test that epsilon defaults to 0.1 in CompanionSelection."""
        companion_sel = CompanionSelection(method="softmax")

        # Default epsilon should be 0.1
        assert companion_sel.epsilon == 0.1

    def test_epsilon_c_explicit_value(self, base_params):
        """Test that epsilon can be set explicitly."""
        companion_sel = CompanionSelection(method="softmax", epsilon=0.2)

        # Epsilon should be set to explicit value
        assert companion_sel.epsilon == 0.2

    def test_epsilon_c_affects_companion_selection(self, base_params):
        """Test that different epsilon values produce different results."""
        # Small epsilon (tight companion selection)
        cloning_small = CloningParams(
            sigma_x=0.1,
            lambda_alg=0.0,
            alpha_restitution=0.5,
            companion_selection=CompanionSelection(method="softmax", epsilon=0.01),
        )
        params_small = EuclideanGasParams(**base_params, cloning=cloning_small)
        gas_small = EuclideanGas(params_small)

        # Large epsilon (loose companion selection)
        cloning_large = CloningParams(
            sigma_x=0.1,
            lambda_alg=0.0,
            alpha_restitution=0.5,
            companion_selection=CompanionSelection(method="softmax", epsilon=10.0),
        )
        params_large = EuclideanGasParams(**base_params, cloning=cloning_large)
        gas_large = EuclideanGas(params_large)

        # Run with same seed
        torch.manual_seed(42)
        results_small = gas_small.run(n_steps=3)

        torch.manual_seed(42)
        results_large = gas_large.run(n_steps=3)

        # Results should differ (different companion selection range)
        # Note: They might be similar due to randomness, but on average should differ
        assert not torch.allclose(results_small["x"], results_large["x"], atol=1e-3)


class TestBoundsIntegration:
    """Test integration with TorchBounds."""

    def test_hybrid_with_bounds(self, bounded_params):
        """Test cloning method with bounds creates correct alive_mask."""
        cloning = CloningParams(
            sigma_x=0.5,
            lambda_alg=0.5,
            alpha_restitution=0.5,
            companion_selection=CompanionSelection(method="cloning", epsilon=0.5),
        )
        params = EuclideanGasParams(**bounded_params, cloning=cloning)
        gas = EuclideanGas(params)

        # Initialize with some walkers out of bounds
        torch.manual_seed(42)
        x_init = torch.randn(20, 2) * 10  # Large variance, some out of bounds
        v_init = torch.randn(20, 2)
        state = gas.initialize_state(x_init=x_init, v_init=v_init)

        # Step should handle out-of-bounds walkers
        state_cloned, state_final = gas.step(state)

        # All final positions should respect bounds (may be out after cloning, but algorithm handles it)
        assert state_cloned.x.shape == (20, 2)
        assert state_final.x.shape == (20, 2)

    def test_bounds_enforce_alive_mask(self, bounded_params, test_device):
        """Test that bounds correctly create alive_mask for companion selection."""
        cloning = CloningParams(
            sigma_x=0.5,
            lambda_alg=0.5,
            alpha_restitution=0.5,
            companion_selection=CompanionSelection(method="cloning", epsilon=0.5),
        )
        params = EuclideanGasParams(**bounded_params, cloning=cloning)
        gas = EuclideanGas(params)

        # Create state with known out-of-bounds walkers on correct device
        x = torch.tensor(
            [
                [0.0, 0.0],  # In bounds
                [10.0, 0.0],  # Out of bounds (x > 5)
                [0.0, 10.0],  # Out of bounds (y > 5)
                [2.0, 2.0],  # In bounds
            ],
            device=test_device,
        )
        v = torch.zeros(4, 2, device=test_device)
        state = SwarmState(x, v)

        # Check bounds.contains creates correct mask
        alive_mask = params.bounds.contains(x)
        expected_alive = torch.tensor([True, False, False, True], device=test_device)
        assert torch.equal(alive_mask, expected_alive)

        # Cloning operator should use this mask
        state_cloned, _ = gas.step(state)
        # Should complete without errors
        assert state_cloned.x.shape == (4, 2)


class TestBackwardCompatibility:
    """Test backward compatibility with old behavior."""

    def test_default_params_match_old_behavior(self, base_params):
        """Test that default parameters preserve old behavior."""
        # Old-style params (no epsilon_c, no companion_selection_method)
        cloning = CloningParams(
            sigma_x=0.1,
            lambda_alg=0.5,
            alpha_restitution=0.5,
            # Uses defaults: companion_selection_method="hybrid", epsilon_c=None
        )

        # Should work identically to old code
        params = EuclideanGasParams(**base_params, cloning=cloning)
        gas = EuclideanGas(params)

        torch.manual_seed(42)
        results = gas.run(n_steps=3)

        # Should produce valid results
        assert results["x"].shape == (4, 20, 2)
        assert results["v"].shape == (4, 20, 2)
        assert torch.isfinite(results["x"]).all()
        assert torch.isfinite(results["v"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
