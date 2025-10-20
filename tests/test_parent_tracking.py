"""Tests for parent tracking integration between FractalSet and ScutoidTessellation."""

import pytest
import torch

from fragile.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    EuclideanGasParams,
    LangevinParams,
    SimpleQuadraticPotential,
)
from fragile.fractal_set import FractalSet
from fragile.scutoid import ScutoidTessellation


@pytest.fixture
def simple_gas():
    """Create a simple EuclideanGas instance."""
    params = EuclideanGasParams(
        N=10,
        d=2,
        potential=SimpleQuadraticPotential(),
        langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01),
        cloning=CloningParams(
            sigma_x=0.5,
            lambda_alg=0.1,
            alpha_restitution=0.5,
            companion_selection_method="softmax",
        ),
        device="cpu",
        dtype="float32",
    )
    return EuclideanGas(params)


class TestParentTracking:
    """Test parent tracking between FractalSet and ScutoidTessellation."""

    def test_fractal_set_get_parent_ids(self, simple_gas):
        """Test that FractalSet can extract parent IDs."""
        fs = FractalSet(N=simple_gas.params.N, d=simple_gas.params.d)

        n_steps = 5
        simple_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

        # Extract parent IDs for each timestep
        for t in range(1, n_steps + 1):
            parent_ids = fs.get_parent_ids(t)

            # Check shape
            assert parent_ids.shape == (simple_gas.params.N,)

            # Check all parent IDs are valid walker indices
            assert all(0 <= pid < simple_gas.params.N for pid in parent_ids)

    def test_fractal_set_parent_ids_errors(self, simple_gas):
        """Test that FractalSet.get_parent_ids validates inputs."""
        fs = FractalSet(N=simple_gas.params.N, d=simple_gas.params.d)

        n_steps = 3
        simple_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

        # Timestep 0 should error (no parents)
        with pytest.raises(ValueError, match="must be > 0"):
            fs.get_parent_ids(0)

        # Out of range should error
        with pytest.raises(ValueError, match="out of range"):
            fs.get_parent_ids(99)

    def test_scutoid_parent_ids_from_cloning_operator(self, simple_gas):
        """Test that ScutoidTessellation gets parent IDs from cloning operator."""
        tessellation = ScutoidTessellation(
            N=simple_gas.params.N,
            d=simple_gas.params.d,
        )

        n_steps = 5
        simple_gas.run(n_steps=n_steps, scutoid_tessellation=tessellation)

        # Check that scutoids have correct parent IDs
        for interval in range(n_steps):
            scutoids = tessellation.scutoid_cells[interval]
            for scutoid in scutoids:
                # Parent ID should be valid
                assert 0 <= scutoid.parent_id < simple_gas.params.N

    def test_scutoid_parent_ids_from_fractal_set(self, simple_gas):
        """Test that ScutoidTessellation can use FractalSet for parent tracking."""
        fs = FractalSet(N=simple_gas.params.N, d=simple_gas.params.d)
        tessellation = ScutoidTessellation(
            N=simple_gas.params.N,
            d=simple_gas.params.d,
        )

        n_steps = 5
        simple_gas.run(
            n_steps=n_steps,
            fractal_set=fs,
            record_fitness=True,
            scutoid_tessellation=tessellation,
        )

        # Check that scutoids have parent IDs matching FractalSet
        for t in range(1, n_steps + 1):
            fs_parent_ids = fs.get_parent_ids(t)
            scutoids = tessellation.scutoid_cells[t - 1]

            for i, scutoid in enumerate(scutoids):
                # Scutoid parent ID should match FractalSet parent ID
                assert scutoid.parent_id == fs_parent_ids[i]

    def test_cloning_operator_returns_parents(self, simple_gas):
        """Test that cloning operator can return parent IDs."""
        state = simple_gas.initialize_state()

        # Apply cloning without parent tracking
        state_cloned = simple_gas.cloning_op.apply(state, return_parents=False)
        assert isinstance(state_cloned, type(state))

        # Apply cloning with parent tracking
        state_cloned, parent_ids = simple_gas.cloning_op.apply(state, return_parents=True)
        assert isinstance(state_cloned, type(state))
        assert parent_ids.shape == (simple_gas.params.N,)
        assert all(0 <= pid < simple_gas.params.N for pid in parent_ids.cpu().numpy())

    def test_step_returns_parents(self, simple_gas):
        """Test that step() method can return parent IDs via info dict."""
        state = simple_gas.initialize_state()

        # Step without parent tracking
        state_cloned, state_final = simple_gas.step(state, return_info=False)
        assert isinstance(state_cloned, type(state))
        assert isinstance(state_final, type(state))

        # Step with info (contains companions and will_clone for parent tracking)
        state_cloned, state_final, info = simple_gas.step(state, return_info=True)
        assert isinstance(state_cloned, type(state))
        assert isinstance(state_final, type(state))

        # Compute parent IDs from info
        companions = info["companions"]
        will_clone = info["will_clone"]
        parent_ids = torch.where(
            will_clone,
            companions,  # Cloned walkers inherit from companion
            torch.arange(simple_gas.params.N, device=companions.device),
        )

        assert parent_ids.shape == (simple_gas.params.N,)
        assert all(0 <= pid < simple_gas.params.N for pid in parent_ids.cpu().numpy())

    def test_parent_tracking_consistency(self, simple_gas):
        """Test that parent tracking is consistent across methods."""
        fs = FractalSet(N=simple_gas.params.N, d=simple_gas.params.d)
        tessellation = ScutoidTessellation(
            N=simple_gas.params.N,
            d=simple_gas.params.d,
        )

        n_steps = 10
        simple_gas.run(
            n_steps=n_steps,
            fractal_set=fs,
            record_fitness=True,
            scutoid_tessellation=tessellation,
        )

        # For each timestep, verify parent tracking is consistent
        for t in range(1, n_steps + 1):
            # Get parent IDs from FractalSet
            fs_parents = fs.get_parent_ids(t)

            # Get parent IDs from Scutoid cells
            scutoids = tessellation.scutoid_cells[t - 1]
            scutoid_parents = [s.parent_id for s in scutoids]

            # Check they match
            for i in range(simple_gas.params.N):
                assert (
                    fs_parents[i] == scutoid_parents[i]
                ), f"Mismatch at timestep {t}, walker {i}: FractalSet={fs_parents[i]}, Scutoid={scutoid_parents[i]}"

    def test_cloning_edges_in_fractal_set(self, simple_gas):
        """Test that cloning edges are properly recorded in FractalSet."""
        fs = FractalSet(N=simple_gas.params.N, d=simple_gas.params.d)

        n_steps = 10
        simple_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

        # Count cloning edges
        n_cloning = 0
        n_kinetic = 0

        for u, v in fs.graph.edges():
            edge_data = fs.graph.edges[u, v]
            edge_type = edge_data.get("edge_type", "kinetic")

            if edge_type == "cloning":
                n_cloning += 1
                # Verify cloning edge structure
                parent_id, parent_t = u
                _child_id, child_t = v
                assert child_t == parent_t + 1
                assert edge_data.get("companion_id") == parent_id
            elif edge_type == "kinetic":
                n_kinetic += 1
                # Verify kinetic edge structure
                walker_id_prev, t_prev = u
                walker_id_curr, t_curr = v
                assert walker_id_prev == walker_id_curr
                assert t_curr == t_prev + 1

        # We should have edges (some kinetic, potentially some cloning)
        total_edges = n_cloning + n_kinetic
        assert total_edges == simple_gas.params.N * n_steps

    def test_backward_compatibility_without_parent_tracking(self, simple_gas):
        """Test that existing code works without parent tracking."""
        # Run without fractal set or scutoid tessellation
        n_steps = 5
        result = simple_gas.run(n_steps=n_steps)

        assert "x" in result
        assert "v" in result
        assert "fractal_set" not in result
        assert "scutoid_tessellation" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
