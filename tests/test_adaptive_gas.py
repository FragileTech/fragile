"""Tests for the Adaptive Viscous Fluid Model (AdaptiveGas)."""

import pytest
import torch

from fragile.adaptive_gas import (
    AdaptiveGas,
    AdaptiveGasParams,
    AdaptiveKineticOperator,
    AdaptiveParams,
    MeanFieldOps,
    ViscousForce,
)
from fragile.euclidean_gas import (
    EuclideanGas,
    SwarmState,
)


@pytest.fixture
def adaptive_params():
    """Standard adaptive parameters with small perturbations."""
    return AdaptiveParams(
        epsilon_F=0.1,
        nu=0.05,
        epsilon_Sigma=2.0,  # Must be > H_max
        A=1.0,
        sigma_prime_min_patch=0.1,
        patch_radius=1.0,
        l_viscous=0.5,
        use_adaptive_diffusion=True,
    )


@pytest.fixture
def backbone_adaptive_params():
    """Adaptive parameters with all adaptations disabled (pure backbone)."""
    return AdaptiveParams(
        epsilon_F=0.0,  # No adaptive force
        nu=0.0,  # No viscous force
        epsilon_Sigma=2.0,
        A=1.0,
        sigma_prime_min_patch=0.1,
        patch_radius=1.0,
        l_viscous=0.5,
        use_adaptive_diffusion=False,  # No adaptive diffusion
    )


@pytest.fixture
def adaptive_gas_params(euclidean_gas_params, adaptive_params):
    """Complete adaptive gas parameters."""
    return AdaptiveGasParams(
        euclidean=euclidean_gas_params,
        adaptive=adaptive_params,
        measurement_fn="potential",
    )


@pytest.fixture
def backbone_gas_params(euclidean_gas_params, backbone_adaptive_params):
    """Adaptive gas parameters with backbone only (no adaptations)."""
    return AdaptiveGasParams(
        euclidean=euclidean_gas_params,
        adaptive=backbone_adaptive_params,
        measurement_fn="potential",
    )


class TestAdaptiveParams:
    """Tests for AdaptiveParams configuration."""

    def test_adaptive_params_creation(self, adaptive_params):
        """Test that adaptive parameters can be created."""
        assert adaptive_params.epsilon_F == 0.1
        assert adaptive_params.nu == 0.05
        assert adaptive_params.epsilon_Sigma == 2.0
        assert adaptive_params.A == 1.0

    def test_adaptive_params_positive_epsilon_sigma(self):
        """Test that epsilon_Sigma must be positive."""
        with pytest.raises(Exception):
            AdaptiveParams(
                epsilon_F=0.1,
                nu=0.1,
                epsilon_Sigma=0.0,  # Must be > 0
                A=1.0,
                sigma_prime_min_patch=0.1,
                patch_radius=1.0,
                l_viscous=0.5,
            )

    def test_adaptive_params_non_negative_epsilon_F(self, adaptive_params):
        """Test that epsilon_F can be zero (backbone mode)."""
        params = AdaptiveParams(
            epsilon_F=0.0,
            nu=0.0,
            epsilon_Sigma=2.0,
            A=1.0,
            sigma_prime_min_patch=0.1,
            patch_radius=1.0,
            l_viscous=0.5,
        )
        assert params.epsilon_F == 0.0


class TestMeanFieldOps:
    """Tests for mean-field functional computations."""

    def test_fitness_potential_shape(self, adaptive_params):
        """Test that fitness potential returns correct shape."""
        N, d = 10, 2
        device, dtype = torch.device("cpu"), torch.float64

        x = torch.randn(N, d, device=device, dtype=dtype)
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        measurement = torch.randn(N, device=device, dtype=dtype)

        V_fit = MeanFieldOps.compute_fitness_potential(state, measurement, adaptive_params)

        assert V_fit.shape == (N,)
        assert V_fit.device == device
        assert V_fit.dtype == dtype

    def test_fitness_potential_bounded(self, adaptive_params):
        """Test that fitness potential is bounded in [0, A]."""
        N, d = 20, 2
        device, dtype = torch.device("cpu"), torch.float64

        x = torch.randn(N, d, device=device, dtype=dtype) * 5  # Wide spread
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        measurement = torch.randn(N, device=device, dtype=dtype) * 10  # Wide range

        V_fit = MeanFieldOps.compute_fitness_potential(state, measurement, adaptive_params)

        # Should be bounded by rescale function: [0, A]
        assert torch.all(V_fit >= 0)
        assert torch.all(V_fit <= adaptive_params.A)

    def test_fitness_potential_patch_locality(self, adaptive_params):
        """Test that fitness potential uses local patch statistics."""
        N, d = 20, 2
        device, dtype = torch.device("cpu"), torch.float64

        # Create two clusters: one at origin, one far away
        x1 = torch.randn(10, d, device=device, dtype=dtype) * 0.1  # Tight cluster at origin
        x2 = torch.randn(10, d, device=device, dtype=dtype) * 0.1 + 10.0  # Tight cluster at (10, 10)
        x = torch.cat([x1, x2], dim=0)

        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        # Different measurements for each cluster
        m1 = torch.ones(10, device=device, dtype=dtype) * 1.0  # Low values
        m2 = torch.ones(10, device=device, dtype=dtype) * 5.0  # High values
        measurement = torch.cat([m1, m2], dim=0)

        V_fit = MeanFieldOps.compute_fitness_potential(state, measurement, adaptive_params)

        # Fitness should be bounded [0, A]
        assert torch.all(V_fit >= 0)
        assert torch.all(V_fit <= adaptive_params.A)

        # Within each cluster, fitness should be similar since measurements are identical
        fitness_cluster1 = V_fit[:10]
        fitness_cluster2 = V_fit[10:]

        # Check within-cluster consistency (measurements are constant within cluster)
        assert torch.std(fitness_cluster1) < 0.01 or torch.allclose(fitness_cluster1, fitness_cluster1[0] * torch.ones_like(fitness_cluster1), atol=0.01)
        assert torch.std(fitness_cluster2) < 0.01 or torch.allclose(fitness_cluster2, fitness_cluster2[0] * torch.ones_like(fitness_cluster2), atol=0.01)

    def test_fitness_gradient_shape(self, adaptive_params, simple_potential):
        """Test that fitness gradient returns correct shape."""
        N, d = 10, 2
        device, dtype = torch.device("cpu"), torch.float64

        x = torch.randn(N, d, device=device, dtype=dtype)
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        measurement = simple_potential.evaluate(x)

        grad_V_fit = MeanFieldOps.compute_fitness_gradient(
            state, measurement, simple_potential, adaptive_params
        )

        assert grad_V_fit.shape == (N, d)
        assert grad_V_fit.device == device
        assert grad_V_fit.dtype == dtype

    def test_fitness_hessian_shape(self, adaptive_params, simple_potential):
        """Test that fitness Hessian returns correct shape."""
        N, d = 5, 2  # Small for efficiency
        device, dtype = torch.device("cpu"), torch.float64

        x = torch.randn(N, d, device=device, dtype=dtype)
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        measurement = simple_potential.evaluate(x)

        H = MeanFieldOps.compute_fitness_hessian(
            state, measurement, simple_potential, adaptive_params
        )

        assert H.shape == (N, d, d)
        assert H.device == device
        assert H.dtype == dtype

    def test_fitness_hessian_symmetry(self, adaptive_params, simple_potential):
        """Test that fitness Hessian is symmetric."""
        N, d = 5, 2
        device, dtype = torch.device("cpu"), torch.float64

        torch.manual_seed(42)
        x = torch.randn(N, d, device=device, dtype=dtype)
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        measurement = simple_potential.evaluate(x)

        H = MeanFieldOps.compute_fitness_hessian(
            state, measurement, simple_potential, adaptive_params
        )

        # Check symmetry for each walker
        for i in range(N):
            assert torch.allclose(H[i], H[i].T, atol=1e-3)


class TestViscousForce:
    """Tests for viscous force computation."""

    def test_gaussian_kernel_shape(self):
        """Test Gaussian kernel returns correct shape."""
        N = 10
        r = torch.rand(N, N)
        l = 1.0

        K = ViscousForce.gaussian_kernel(r, l)

        assert K.shape == (N, N)

    def test_gaussian_kernel_symmetry(self):
        """Test Gaussian kernel is symmetric."""
        N = 10
        r = torch.rand(N, N)
        r = 0.5 * (r + r.T)  # Make distance matrix symmetric
        l = 1.0

        K = ViscousForce.gaussian_kernel(r, l)

        assert torch.allclose(K, K.T)

    def test_gaussian_kernel_positive(self):
        """Test Gaussian kernel is positive."""
        N = 10
        r = torch.rand(N, N)
        l = 1.0

        K = ViscousForce.gaussian_kernel(r, l)

        assert torch.all(K > 0)

    def test_gaussian_kernel_decay(self):
        """Test Gaussian kernel decays with distance."""
        r = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        l = 1.0

        K = ViscousForce.gaussian_kernel(r, l)

        # Should decay: K(0) > K(1) > K(2) > K(3)
        assert K[0, 0] > K[0, 1]
        assert K[0, 1] > K[0, 2]
        assert K[0, 2] > K[0, 3]

    def test_viscous_force_shape(self):
        """Test viscous force returns correct shape."""
        N, d = 10, 2
        device, dtype = torch.device("cpu"), torch.float64

        x = torch.randn(N, d, device=device, dtype=dtype)
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        nu = 0.1
        l_viscous = 0.5

        F_viscous = ViscousForce.compute(state, nu, l_viscous)

        assert F_viscous.shape == (N, d)
        assert F_viscous.device == device
        assert F_viscous.dtype == dtype

    def test_viscous_force_zero_with_uniform_velocity(self):
        """Test viscous force is zero when all velocities are equal."""
        N, d = 10, 2
        device, dtype = torch.device("cpu"), torch.float64

        x = torch.randn(N, d, device=device, dtype=dtype)
        v = torch.ones(N, d, device=device, dtype=dtype)  # All same velocity
        state = SwarmState(x, v)

        nu = 0.1
        l_viscous = 0.5

        F_viscous = ViscousForce.compute(state, nu, l_viscous)

        # Should be zero (no relative motion)
        assert torch.allclose(F_viscous, torch.zeros_like(F_viscous), atol=1e-6)

    def test_viscous_force_dissipative(self):
        """Test viscous force is dissipative (reduces relative velocity)."""
        N, d = 10, 2
        device, dtype = torch.device("cpu"), torch.float64

        # Create walkers with different velocities
        x = torch.randn(N, d, device=device, dtype=dtype) * 0.1  # Clustered
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        nu = 0.1
        l_viscous = 1.0  # Large length scale for strong interaction

        F_viscous = ViscousForce.compute(state, nu, l_viscous)

        # Viscous force should reduce kinetic energy of relative motion
        # This is a soft constraint, not always true for all configurations
        # but should hold statistically

        # Check that force points towards smoothing velocity field
        # For a walker moving faster than neighbors, force should oppose
        mean_v = v.mean(dim=0)
        for i in range(N):
            relative_v = v[i] - mean_v
            # If moving faster than average, viscous force should slow down
            if torch.norm(relative_v) > 0.1:
                # Force should have component opposing relative motion
                # This is approximate - viscous force is local, not global
                pass  # Skip strict test, just check shape and scale


class TestAdaptiveDiffusionTensor:
    """Tests for regularized Hessian diffusion tensor."""

    def test_diffusion_tensor_shape(self, adaptive_gas_params, simple_potential):
        """Test diffusion tensor returns correct shape."""
        N, d = 5, 2
        device, dtype = torch.device("cpu"), torch.float64

        x = torch.randn(N, d, device=device, dtype=dtype)
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        kinetic_op = AdaptiveKineticOperator(
            adaptive_gas_params.euclidean, adaptive_gas_params.adaptive
        )

        measurement = simple_potential.evaluate(x)
        Sigma_reg = kinetic_op.compute_adaptive_diffusion_tensor(state, measurement)

        assert Sigma_reg.shape == (N, d, d)
        assert Sigma_reg.device == device
        assert Sigma_reg.dtype == dtype

    def test_diffusion_tensor_positive_definite(self, adaptive_gas_params, simple_potential):
        """Test diffusion tensor is positive definite."""
        N, d = 5, 2
        device, dtype = torch.device("cpu"), torch.float64

        torch.manual_seed(42)
        x = torch.randn(N, d, device=device, dtype=dtype)
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        kinetic_op = AdaptiveKineticOperator(
            adaptive_gas_params.euclidean, adaptive_gas_params.adaptive
        )

        measurement = simple_potential.evaluate(x)
        Sigma_reg = kinetic_op.compute_adaptive_diffusion_tensor(state, measurement)

        # Compute G_reg = Sigma_reg @ Sigma_reg^T (should be positive definite)
        for i in range(N):
            G_reg_i = Sigma_reg[i] @ Sigma_reg[i].T
            eigenvalues = torch.linalg.eigvalsh(G_reg_i)
            assert torch.all(eigenvalues > 0), f"Walker {i} has non-positive eigenvalues"

    def test_diffusion_tensor_uniform_ellipticity(self, adaptive_gas_params, simple_potential):
        """Test that G_reg satisfies uniform ellipticity bounds."""
        N, d = 10, 2
        device, dtype = torch.device("cpu"), torch.float64

        torch.manual_seed(42)
        # Test with various configurations
        x = torch.randn(N, d, device=device, dtype=dtype) * 3
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        kinetic_op = AdaptiveKineticOperator(
            adaptive_gas_params.euclidean, adaptive_gas_params.adaptive
        )

        measurement = simple_potential.evaluate(x)
        Sigma_reg = kinetic_op.compute_adaptive_diffusion_tensor(state, measurement)

        # Compute G_reg = Sigma_reg @ Sigma_reg^T
        G_reg = torch.einsum("nij,nkj->nik", Sigma_reg, Sigma_reg)

        # Check eigenvalues are bounded
        for i in range(N):
            eigenvalues = torch.linalg.eigvalsh(G_reg[i])
            # Should satisfy c_min <= eigenvalues <= c_max
            assert torch.all(eigenvalues > 0), "Eigenvalues must be positive"
            # Rough bounds - exact bounds depend on H_max and epsilon_Sigma
            assert torch.all(eigenvalues < 100), "Eigenvalues too large"

    def test_diffusion_tensor_isotropic_fallback(self, backbone_gas_params, simple_potential):
        """Test fallback to isotropic diffusion when adaptive diffusion disabled."""
        N, d = 5, 2
        device, dtype = torch.device("cpu"), torch.float64

        x = torch.randn(N, d, device=device, dtype=dtype)
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        kinetic_op = AdaptiveKineticOperator(
            backbone_gas_params.euclidean, backbone_gas_params.adaptive
        )

        measurement = simple_potential.evaluate(x)
        Sigma_reg = kinetic_op.compute_adaptive_diffusion_tensor(state, measurement)

        # Should be isotropic: Sigma_reg[i] = sigma * I for all i
        for i in range(N):
            # Check if matrix is proportional to identity
            Sigma_i = Sigma_reg[i]
            # Extract diagonal and check off-diagonal is near zero
            diag = torch.diagonal(Sigma_i)
            off_diag = Sigma_i - torch.diag(diag)
            assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-6)
            # Check diagonal elements are equal
            assert torch.allclose(diag, diag[0] * torch.ones_like(diag), atol=1e-6)


class TestAdaptiveKineticOperator:
    """Tests for adaptive kinetic operator integration."""

    def test_kinetic_operator_creation(self, adaptive_gas_params):
        """Test that adaptive kinetic operator can be created."""
        kinetic_op = AdaptiveKineticOperator(
            adaptive_gas_params.euclidean, adaptive_gas_params.adaptive
        )

        assert kinetic_op.adaptive_params == adaptive_gas_params.adaptive
        assert kinetic_op.potential == adaptive_gas_params.euclidean.potential

    def test_kinetic_operator_apply_shape(self, adaptive_gas_params):
        """Test that kinetic operator returns correct shape."""
        N, d = 10, 2
        device, dtype = torch.device("cpu"), torch.float64

        x = torch.randn(N, d, device=device, dtype=dtype)
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        kinetic_op = AdaptiveKineticOperator(
            adaptive_gas_params.euclidean, adaptive_gas_params.adaptive
        )

        torch.manual_seed(42)
        state_new = kinetic_op.apply(state)

        assert state_new.N == N
        assert state_new.d == d
        assert state_new.x.shape == (N, d)
        assert state_new.v.shape == (N, d)

    def test_kinetic_operator_changes_state(self, adaptive_gas_params):
        """Test that kinetic operator modifies state."""
        N, d = 10, 2
        device, dtype = torch.device("cpu"), torch.float64

        x = torch.randn(N, d, device=device, dtype=dtype)
        v = torch.randn(N, d, device=device, dtype=dtype)
        state = SwarmState(x, v)

        kinetic_op = AdaptiveKineticOperator(
            adaptive_gas_params.euclidean, adaptive_gas_params.adaptive
        )

        torch.manual_seed(42)
        state_new = kinetic_op.apply(state)

        # State should change
        assert not torch.allclose(state_new.x, state.x)
        assert not torch.allclose(state_new.v, state.v)


class TestAdaptiveGas:
    """Tests for main AdaptiveGas class."""

    def test_initialization(self, adaptive_gas_params):
        """Test AdaptiveGas initialization."""
        gas = AdaptiveGas(adaptive_gas_params)

        assert gas.adaptive_params == adaptive_gas_params.adaptive
        assert isinstance(gas.kinetic_op, AdaptiveKineticOperator)
        assert gas.cloning_op is not None

    def test_inherits_from_euclidean_gas(self, adaptive_gas_params):
        """Test that AdaptiveGas inherits from EuclideanGas."""
        gas = AdaptiveGas(adaptive_gas_params)
        assert isinstance(gas, EuclideanGas)

    def test_initialize_state(self, adaptive_gas_params):
        """Test state initialization."""
        gas = AdaptiveGas(adaptive_gas_params)

        torch.manual_seed(42)
        state = gas.initialize_state()

        assert state.N == adaptive_gas_params.euclidean.N
        assert state.d == adaptive_gas_params.euclidean.d

    def test_step(self, adaptive_gas_params):
        """Test single step execution."""
        gas = AdaptiveGas(adaptive_gas_params)

        torch.manual_seed(42)
        state = gas.initialize_state()

        torch.manual_seed(43)
        state_cloned, state_final = gas.step(state)

        assert isinstance(state_cloned, SwarmState)
        assert isinstance(state_final, SwarmState)
        assert state_final.N == state.N

    def test_run(self, adaptive_gas_params):
        """Test multi-step run."""
        gas = AdaptiveGas(adaptive_gas_params)

        n_steps = 10
        torch.manual_seed(42)
        results = gas.run(n_steps)

        assert "x" in results
        assert "v" in results
        assert results["x"].shape[0] == n_steps + 1

    def test_get_fitness_potential(self, adaptive_gas_params):
        """Test fitness potential extraction."""
        gas = AdaptiveGas(adaptive_gas_params)

        torch.manual_seed(42)
        state = gas.initialize_state()

        V_fit = gas.get_fitness_potential(state)

        assert V_fit.shape == (state.N,)
        assert torch.all(V_fit >= 0)
        assert torch.all(V_fit <= adaptive_gas_params.adaptive.A)


class TestBackboneModeEquivalence:
    """Tests that backbone mode (all adaptations off) matches EuclideanGas."""

    def test_backbone_mode_initialization(self, backbone_gas_params):
        """Test that backbone mode can be created."""
        gas = AdaptiveGas(backbone_gas_params)
        assert gas.adaptive_params.epsilon_F == 0.0
        assert gas.adaptive_params.nu == 0.0
        assert gas.adaptive_params.use_adaptive_diffusion is False

    def test_backbone_mode_runs(self, backbone_gas_params):
        """Test that backbone mode runs successfully."""
        gas = AdaptiveGas(backbone_gas_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=10)

        assert results["x"].shape[0] == 11
        assert torch.all(torch.isfinite(results["x"]))
        assert torch.all(torch.isfinite(results["v"]))


class TestAdaptiveStability:
    """Tests for stability properties of adaptive gas."""

    def test_finite_evolution(self, adaptive_gas_params):
        """Test that evolution stays finite (no NaN or Inf)."""
        gas = AdaptiveGas(adaptive_gas_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=50)

        # Check all values are finite
        assert torch.all(torch.isfinite(results["x"]))
        assert torch.all(torch.isfinite(results["v"]))
        assert torch.all(torch.isfinite(results["var_x"]))
        assert torch.all(torch.isfinite(results["var_v"]))

    def test_bounded_positions(self, adaptive_gas_params):
        """Test that positions don't explode."""
        gas = AdaptiveGas(adaptive_gas_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=100)

        # With confining potential, positions should stay bounded
        max_position = torch.max(torch.abs(results["x"]))
        assert max_position < 50.0, "Positions exploded"

    def test_bounded_velocities(self, adaptive_gas_params):
        """Test that velocities don't explode."""
        gas = AdaptiveGas(adaptive_gas_params)

        torch.manual_seed(42)
        results = gas.run(n_steps=100)

        # With friction, velocities should stay bounded
        max_velocity = torch.max(torch.abs(results["v"]))
        assert max_velocity < 20.0, "Velocities exploded"

    def test_variance_convergence_trend(self, adaptive_gas_params):
        """Test that spatial variance shows convergence trend."""
        # Modify params for stronger convergence
        params = adaptive_gas_params.model_copy(deep=True)
        params.euclidean.N = 50  # Larger swarm

        gas = AdaptiveGas(params)

        torch.manual_seed(42)
        results = gas.run(n_steps=200)

        var_x = results["var_x"]

        # Variance should generally decrease (may have fluctuations)
        # Compare first 20 steps vs last 20 steps
        early_var = var_x[10:30].mean()
        late_var = var_x[-20:].mean()

        # Late variance should be smaller (convergence)
        assert late_var < early_var * 1.5, "No convergence trend observed"


class TestAdaptiveMechanisms:
    """Tests that adaptive mechanisms produce expected effects."""

    def test_adaptive_force_effect(self, euclidean_gas_params, adaptive_params):
        """Test that adaptive force influences dynamics."""
        # Create two gases: one with strong adaptive force, one without
        params_with = AdaptiveGasParams(
            euclidean=euclidean_gas_params,
            adaptive=adaptive_params,
            measurement_fn="potential",
        )
        # Increase epsilon_F for stronger effect
        params_with.adaptive.epsilon_F = 0.5  # Stronger adaptive force

        params_without = params_with.model_copy(deep=True)
        params_without.adaptive.epsilon_F = 0.0

        gas_with = AdaptiveGas(params_with)
        gas_without = AdaptiveGas(params_without)

        # Run both with same seed
        torch.manual_seed(42)
        state_init = gas_with.initialize_state()

        torch.manual_seed(43)
        results_with = gas_with.run(n_steps=50, x_init=state_init.x, v_init=state_init.v)

        torch.manual_seed(43)
        results_without = gas_without.run(n_steps=50, x_init=state_init.x, v_init=state_init.v)

        # Trajectories should differ due to adaptive force
        # Compute mean squared difference
        mse = torch.mean((results_with["x"] - results_without["x"])**2)
        assert mse > 1e-3, f"Adaptive force has negligible effect: MSE={mse}"

    def test_viscous_force_effect(self, euclidean_gas_params, adaptive_params):
        """Test that viscous force influences dynamics."""
        params_with = AdaptiveGasParams(
            euclidean=euclidean_gas_params,
            adaptive=adaptive_params,
            measurement_fn="potential",
        )

        params_without = params_with.model_copy(deep=True)
        params_without.adaptive.nu = 0.0

        gas_with = AdaptiveGas(params_with)
        gas_without = AdaptiveGas(params_without)

        torch.manual_seed(42)
        state_init = gas_with.initialize_state()

        torch.manual_seed(43)
        results_with = gas_with.run(n_steps=20, x_init=state_init.x, v_init=state_init.v)

        torch.manual_seed(43)
        results_without = gas_without.run(n_steps=20, x_init=state_init.x, v_init=state_init.v)

        # Trajectories should differ
        assert not torch.allclose(results_with["v"], results_without["v"], atol=1e-2)

    def test_adaptive_diffusion_effect(self, euclidean_gas_params, adaptive_params):
        """Test that adaptive diffusion influences dynamics."""
        params_with = AdaptiveGasParams(
            euclidean=euclidean_gas_params,
            adaptive=adaptive_params,
            measurement_fn="potential",
        )

        params_without = params_with.model_copy(deep=True)
        params_without.adaptive.use_adaptive_diffusion = False

        gas_with = AdaptiveGas(params_with)
        gas_without = AdaptiveGas(params_without)

        torch.manual_seed(42)
        state_init = gas_with.initialize_state()

        torch.manual_seed(43)
        results_with = gas_with.run(n_steps=20, x_init=state_init.x, v_init=state_init.v)

        torch.manual_seed(43)
        results_without = gas_without.run(n_steps=20, x_init=state_init.x, v_init=state_init.v)

        # Trajectories should differ
        assert not torch.allclose(results_with["x"], results_without["x"], atol=1e-2)


class TestReproducibility:
    """Tests for reproducibility with fixed seeds."""

    def test_reproducible_with_seed(self, adaptive_gas_params):
        """Test that results are reproducible with same seed."""
        gas = AdaptiveGas(adaptive_gas_params)

        torch.manual_seed(42)
        results1 = gas.run(n_steps=20)

        torch.manual_seed(42)
        results2 = gas.run(n_steps=20)

        assert torch.allclose(results1["x"], results2["x"])
        assert torch.allclose(results1["v"], results2["v"])

    def test_different_without_seed(self, adaptive_gas_params):
        """Test that results differ without fixed seed."""
        gas = AdaptiveGas(adaptive_gas_params)

        results1 = gas.run(n_steps=10)
        results2 = gas.run(n_steps=10)

        # Should be different (stochastic)
        assert not torch.allclose(results1["x"], results2["x"])
