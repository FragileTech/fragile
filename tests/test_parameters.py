"""Tests for Pydantic parameter models."""

from pydantic import ValidationError
import pytest
import torch

from fragile.euclidean_gas import (
    CloningParams,
    EuclideanGasParams,
    LangevinParams,
    SimpleQuadraticPotential,
)


class TestSimpleQuadraticPotential:
    """Tests for SimpleQuadraticPotential."""

    def test_initialization(self, simple_potential):
        """Test that SimpleQuadraticPotential can be instantiated."""
        assert simple_potential is not None
        assert isinstance(simple_potential, SimpleQuadraticPotential)

    def test_evaluate_at_origin(self, simple_potential):
        """Test U(0) = 0."""
        x = torch.zeros(5, 3)
        U = simple_potential.evaluate(x)
        assert torch.allclose(U, torch.zeros(5))

    def test_evaluate_unit_vectors(self, simple_potential):
        """Test U(e_i) = 0.5 for unit vectors."""
        x = torch.eye(3)
        U = simple_potential.evaluate(x)
        expected = torch.full((3,), 0.5)
        assert torch.allclose(U, expected)

    def test_evaluate_batch(self, simple_potential):
        """Test batch evaluation."""
        x = torch.randn(10, 5)
        U = simple_potential.evaluate(x)
        assert U.shape == (10,)

        # Verify calculation: U(x) = 0.5 * sum(x^2)
        expected = 0.5 * torch.sum(x**2, dim=-1)
        assert torch.allclose(U, expected)

    def test_evaluate_1d(self, simple_potential):
        """Test 1D case."""
        x = torch.tensor([[1.0], [2.0], [3.0]])
        U = simple_potential.evaluate(x)
        expected = torch.tensor([0.5, 2.0, 4.5])
        assert torch.allclose(U, expected)


class TestLangevinParams:
    """Tests for LangevinParams."""

    def test_valid_initialization(self, langevin_params):
        """Test valid parameter initialization."""
        assert langevin_params.gamma == 1.0
        assert langevin_params.beta == 1.0
        assert langevin_params.delta_t == 0.01
        assert langevin_params.integrator == "baoab"

    @pytest.mark.parametrize("gamma", [-1.0, 0.0])
    def test_invalid_gamma(self, gamma):
        """Test that gamma must be positive."""
        with pytest.raises(ValidationError):
            LangevinParams(gamma=gamma, beta=1.0, delta_t=0.01)

    @pytest.mark.parametrize("beta", [-1.0, 0.0])
    def test_invalid_beta(self, beta):
        """Test that beta must be positive."""
        with pytest.raises(ValidationError):
            LangevinParams(gamma=1.0, beta=beta, delta_t=0.01)

    @pytest.mark.parametrize("delta_t", [-0.01, 0.0])
    def test_invalid_delta_t(self, delta_t):
        """Test that delta_t must be positive."""
        with pytest.raises(ValidationError):
            LangevinParams(gamma=1.0, beta=1.0, delta_t=delta_t)

    def test_noise_std(self, langevin_params):
        """Test noise standard deviation calculation."""
        std = langevin_params.noise_std()

        # Should be sqrt(1 - exp(-2 * gamma * dt))
        gamma = langevin_params.gamma
        delta_t = langevin_params.delta_t
        expected = (1.0 - torch.exp(torch.tensor(-2 * gamma * delta_t))).sqrt().item()

        assert abs(std - expected) < 1e-10

    @pytest.mark.parametrize(
        "gamma,delta_t",
        [
            (1.0, 0.01),
            (2.0, 0.005),
            (0.5, 0.02),
        ],
    )
    def test_noise_std_values(self, gamma, delta_t):
        """Test noise_std for various parameter values."""
        params = LangevinParams(gamma=gamma, beta=1.0, delta_t=delta_t)
        std = params.noise_std()

        # Noise std should be in (0, 1)
        assert 0 < std < 1

        # For small dt, should be approximately sqrt(2 * gamma * dt)
        if delta_t < 0.01:
            approx = (2 * gamma * delta_t) ** 0.5
            assert abs(std - approx) < 0.1


class TestCloningParams:
    """Tests for CloningParams."""

    def test_valid_initialization(self, cloning_params):
        """Test valid parameter initialization."""
        assert cloning_params.sigma_x == 0.1
        assert cloning_params.lambda_alg == 1.0
        assert cloning_params.alpha_restitution == 0.5
        assert cloning_params.use_inelastic_collision is True

    @pytest.mark.parametrize("sigma_x", [-0.1, 0.0])
    def test_invalid_sigma_x(self, sigma_x):
        """Test that sigma_x must be positive."""
        with pytest.raises(ValidationError):
            CloningParams(sigma_x=sigma_x, lambda_alg=1.0, alpha_restitution=0.5)

    @pytest.mark.parametrize("lambda_alg", [-1.0])
    def test_invalid_lambda_alg(self, lambda_alg):
        """Test that lambda_alg must be non-negative."""
        with pytest.raises(ValidationError):
            CloningParams(sigma_x=0.1, lambda_alg=lambda_alg, alpha_restitution=0.5)

    def test_lambda_alg_zero_allowed(self):
        """Test that lambda_alg=0 is allowed (position-only mode)."""
        params = CloningParams(sigma_x=0.1, lambda_alg=0.0, alpha_restitution=0.5)
        assert params.lambda_alg == 0.0

    @pytest.mark.parametrize("alpha_restitution", [-0.1, 1.1])
    def test_invalid_restitution(self, alpha_restitution):
        """Test that restitution coefficient must be in [0, 1]."""
        with pytest.raises(ValidationError):
            CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=alpha_restitution)

    @pytest.mark.parametrize("alpha_restitution", [0.0, 0.5, 1.0])
    def test_valid_restitution(self, alpha_restitution):
        """Test valid restitution coefficients."""
        params = CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=alpha_restitution)
        assert params.alpha_restitution == alpha_restitution

    def test_inelastic_collision_flag(self):
        """Test inelastic collision flag."""
        params_true = CloningParams(
            sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5, use_inelastic_collision=True
        )
        params_false = CloningParams(
            sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5, use_inelastic_collision=False
        )

        assert params_true.use_inelastic_collision is True
        assert params_false.use_inelastic_collision is False


class TestEuclideanGasParams:
    """Tests for EuclideanGasParams."""

    def test_valid_initialization(self, euclidean_gas_params):
        """Test valid parameter initialization."""
        assert euclidean_gas_params.N == 10
        assert euclidean_gas_params.d == 2
        assert isinstance(euclidean_gas_params.potential, SimpleQuadraticPotential)
        assert isinstance(euclidean_gas_params.langevin, LangevinParams)
        assert isinstance(euclidean_gas_params.cloning, CloningParams)
        assert euclidean_gas_params.device == "cpu"
        assert euclidean_gas_params.dtype == "float64"

    @pytest.mark.parametrize("N", [-1, 0])
    def test_invalid_N(self, N, simple_potential, langevin_params, cloning_params):
        """Test that N must be positive."""
        with pytest.raises(ValidationError):
            EuclideanGasParams(
                N=N,
                d=2,
                potential=simple_potential,
                langevin=langevin_params,
                cloning=cloning_params,
            )

    @pytest.mark.parametrize("d", [-1, 0])
    def test_invalid_d(self, d, simple_potential, langevin_params, cloning_params):
        """Test that d must be positive."""
        with pytest.raises(ValidationError):
            EuclideanGasParams(
                N=10,
                d=d,
                potential=simple_potential,
                langevin=langevin_params,
                cloning=cloning_params,
            )

    def test_torch_dtype_property(self, euclidean_gas_params):
        """Test torch_dtype property."""
        assert euclidean_gas_params.torch_dtype == torch.float64

    @pytest.mark.parametrize(
        "dtype_str,expected",
        [
            ("float32", torch.float32),
            ("float64", torch.float64),
        ],
    )
    def test_torch_dtype_conversion(
        self, dtype_str, expected, simple_potential, langevin_params, cloning_params
    ):
        """Test dtype string to torch.dtype conversion."""
        params = EuclideanGasParams(
            N=10,
            d=2,
            potential=simple_potential,
            langevin=langevin_params,
            cloning=cloning_params,
            dtype=dtype_str,
        )
        assert params.torch_dtype == expected

    def test_small_swarm(self, small_swarm_params):
        """Test small swarm configuration."""
        assert small_swarm_params.N == 5
        assert small_swarm_params.d == 2

    def test_large_swarm(self, large_swarm_params):
        """Test large swarm configuration."""
        assert large_swarm_params.N == 100
        assert large_swarm_params.d == 3
