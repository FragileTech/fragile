from collections.abc import Callable
import math

import einops
from numba import jit
import numpy as np
import torch

from fragile.bounds import Bounds, NumpyBounds, TorchBounds


"""
This file includes several test functions for optimization described here:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""


def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x**2, 1).flatten()


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    dims = x.shape[1]
    a = 10
    result = a * dims + torch.sum(x**2 - a * torch.cos(2 * math.pi * x), 1)
    return result.flatten()


def eggholder(x: torch.Tensor) -> torch.Tensor:
    x, y = x[:, 0], x[:, 1]
    first_root = torch.sqrt(torch.abs(x / 2.0 + (y + 47)))
    second_root = torch.sqrt(torch.abs(x - (y + 47)))
    return -1 * (y + 47) * torch.sin(first_root) - x * torch.sin(second_root)


def styblinski_tang(x) -> torch.Tensor:
    return torch.sum(x**4 - 16 * x**2 + 5 * x, 1) / 2.0


def rosenbrock(x) -> torch.Tensor:
    return 100 * torch.sum((x[:, :-2] ** 2 - x[:, 1:-1]) ** 2, 1) + torch.sum(
        (x[:, :-2] - 1) ** 2,
        1,
    )


def easom(x) -> torch.Tensor:
    exp_term = (x[:, 0] - np.pi) ** 2 + (x[:, 1] - np.pi) ** 2
    return -torch.cos(x[:, 0]) * torch.cos(x[:, 1]) * torch.exp(-exp_term)


def holder_table(_x) -> torch.Tensor:
    x, y = _x[:, 0], _x[:, 1]
    exp = torch.abs(1 - (torch.sqrt(x * x + y * y) / np.pi))
    return -torch.abs(torch.sin(x) * torch.cos(y) * torch.exp(exp))


@jit(nopython=True)
def _lennard_fast(state):
    state = state.reshape(-1, 3)
    npart = len(state)
    epot = 0.0
    for i in range(npart):
        for j in range(npart):
            if i > j:
                r2 = np.sum((state[j, :] - state[i, :]) ** 2)
                r2i = 1.0 / r2
                r6i = r2i * r2i * r2i
                epot += r6i * (r6i - 1.0)
    return epot * 4


def lennard_jones(x: torch.Tensor) -> torch.Tensor:
    result = np.zeros(x.shape[0])
    x_ = einops.asnumpy(x)
    # assert isinstance(x, torch.Tensor)
    for i in range(x.shape[0]):
        try:
            result[i] = _lennard_fast(x_[i])
        except ZeroDivisionError:  # noqa: PERF203
            result[i] = np.inf
    return torch.from_numpy(result).to(x)


class OptimBenchmark:
    benchmark = None
    best_state = None

    def __init__(self, dims: int, function: Callable, **kwargs):  # noqa: ARG002
        self.dims = dims
        self.bounds = self.get_bounds(dims=dims)
        self.function = function

    def __call__(self, x):
        return self.function(x)

    def sample(self, n_samples):
        return self.bounds.sample(n_samples)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.bounds.shape

    @staticmethod
    def get_bounds(dims: int) -> Bounds | TorchBounds | NumpyBounds:
        raise NotImplementedError


class Sphere(OptimBenchmark):
    benchmark = torch.tensor(0.0)

    def __init__(self, dims: int, **kwargs):
        super().__init__(dims=dims, function=sphere, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-1000, 1000) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return torch.zeros(self.shape)


class Rastrigin(OptimBenchmark):
    benchmark = torch.tensor(0.0)

    def __init__(self, dims: int, **kwargs):
        super().__init__(dims=dims, function=rastrigin, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-5.12, 5.12) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return torch.zeros(self.shape)


class EggHolder(OptimBenchmark):
    benchmark = torch.tensor(-959.64066271)

    def __init__(self, dims: int | None = None, **kwargs):  # noqa: ARG002
        super().__init__(dims=2, function=eggholder, **kwargs)

    @staticmethod
    def get_bounds(dims=None):  # noqa: ARG004
        bounds = [(-512.0, 512.0), (-512.0, 512.0)]
        # bounds = [(1, 512.0), (1, 512.0)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return torch.tensor([512.0, 404.2319])


class StyblinskiTang(OptimBenchmark):
    def __init__(self, dims: int, **kwargs):
        super().__init__(dims=dims, function=styblinski_tang, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-5.0, 5.0) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return torch.ones(self.shape) * -2.903534

    @property
    def benchmark(self):
        return torch.tensor(-39.16617 * self.shape[0])


class Rosenbrock(OptimBenchmark):
    def __init__(self, dims: int, **kwargs):
        super().__init__(dims=dims, function=rosenbrock, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-10.0, 10.0) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return torch.ones(self.shape)

    @property
    def benchmark(self):
        return torch.tensor(0.0)


class Easom(OptimBenchmark):
    def __init__(self, dims: int | None = None, **kwargs):  # noqa: ARG002
        super().__init__(dims=2, function=easom, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-100.0, 100.0) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return torch.ones(self.shape) * np.pi

    @property
    def benchmark(self):
        return torch.tensor(-1)


class HolderTable(OptimBenchmark):
    def __init__(self, dims: int | None = None, *args, **kwargs):  # noqa: ARG002
        super().__init__(dims=2, function=holder_table, *args, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-10.0, 10.0) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return torch.tensor([8.05502, 9.66459])

    @property
    def benchmark(self):
        return torch.tensor(-19.2085)


class LennardJones(OptimBenchmark):
    # http://doye.chem.ox.ac.uk/jon/structures/LJ/tables.150.html
    minima = {
        "2": -1,
        "3": -3,
        "4": -6,
        "5": -9.103852,
        "6": -12.712062,
        "7": -16.505384,
        "8": -19.821489,
        "9": -24.113360,
        "10": -28.422532,
        "11": -32.765970,
        "12": -37.967600,
        "13": -44.326801,
        "14": -47.845157,
        "15": -52.322627,
        "20": -77.177043,
        "25": -102.372663,
        "30": -128.286571,
        "38": -173.928427,
        "50": -244.549926,
        "100": -557.039820,
        "104": -582.038429,
    }

    benchmark = None

    def __init__(self, n_atoms: int = 10, dims=None, **kwargs):  # noqa: ARG002
        self.n_atoms = n_atoms
        self.dims = 3 * n_atoms
        self.benchmark = [torch.zeros(self.n_atoms * 3), self.minima.get(str(int(n_atoms)), 0)]
        super().__init__(dims=self.dims, function=lennard_jones, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-15, 15) for _ in range(dims)]
        return Bounds.from_tuples(bounds)


class MixtureOfGaussians(OptimBenchmark):
    """Mixture of Gaussians benchmark function.

    The function evaluates the negative log-likelihood of a Gaussian mixture:
    f(x) = -log(Σ_k w_k * N(x | μ_k, Σ_k))

    The global minimum occurs at the center of the highest-weighted Gaussian.

    Args:
        dims: Dimensionality of the space
        n_gaussians: Number of Gaussian components in the mixture
        centers: Optional array of shape [n_gaussians, dims] for Gaussian centers.
                 If None, centers are randomly sampled within bounds.
        stds: Optional array of shape [n_gaussians, dims] for standard deviations.
              If None, stds are randomly sampled from [0.1, 2.0].
        weights: Optional array of shape [n_gaussians] for mixture weights.
                 If None, uniform weights are used.
        bounds_range: Tuple (low, high) defining the bounds for each dimension.
                      Default: (-10.0, 10.0)
        seed: Random seed for reproducibility when generating random parameters
    """

    def __init__(
        self,
        dims: int,
        n_gaussians: int = 3,
        centers: torch.Tensor | np.ndarray | None = None,
        stds: torch.Tensor | np.ndarray | None = None,
        weights: torch.Tensor | np.ndarray | None = None,
        bounds_range: tuple[float, float] = (-10.0, 10.0),
        seed: int | None = None,
        **kwargs,
    ):
        self.n_gaussians = n_gaussians
        self.bounds_range = bounds_range

        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Initialize or validate centers
        if centers is None:
            # Random centers within bounds
            low, high = bounds_range
            self.centers = torch.rand(n_gaussians, dims) * (high - low) + low
        else:
            self.centers = torch.as_tensor(centers, dtype=torch.float32)
            if self.centers.shape != (n_gaussians, dims):
                msg = f"Centers shape {self.centers.shape} doesn't match (n_gaussians={n_gaussians}, dims={dims})"
                raise ValueError(msg)

        # Initialize or validate standard deviations
        if stds is None:
            # Random stds between 0.1 and 2.0
            self.stds = torch.rand(n_gaussians, dims) * 1.9 + 0.1
        else:
            self.stds = torch.as_tensor(stds, dtype=torch.float32)
            if self.stds.shape != (n_gaussians, dims):
                msg = f"Stds shape {self.stds.shape} doesn't match (n_gaussians={n_gaussians}, dims={dims})"
                raise ValueError(msg)
            if (self.stds <= 0).any():
                msg = "All standard deviations must be positive"
                raise ValueError(msg)

        # Initialize or validate weights
        if weights is None:
            # Uniform weights
            self.weights = torch.ones(n_gaussians) / n_gaussians
        else:
            self.weights = torch.as_tensor(weights, dtype=torch.float32)
            if self.weights.shape != (n_gaussians,):
                msg = f"Weights shape {self.weights.shape} doesn't match n_gaussians={n_gaussians}"
                raise ValueError(msg)
            if (self.weights < 0).any():
                msg = "All weights must be non-negative"
                raise ValueError(msg)
            # Normalize weights
            self.weights /= self.weights.sum()

        # Create the mixture function
        def mixture_function(x: torch.Tensor) -> torch.Tensor:
            """Evaluate negative log-likelihood of Gaussian mixture.

            Args:
                x: Input tensor of shape [batch_size, dims]

            Returns:
                Negative log-likelihood of shape [batch_size]
            """
            batch_size = x.shape[0]
            device = x.device
            dtype = x.dtype

            # Move parameters to the same device and dtype as input
            centers = self.centers.to(device=device, dtype=dtype)
            stds = self.stds.to(device=device, dtype=dtype)
            weights = self.weights.to(device=device, dtype=dtype)

            # Compute log-probabilities for each Gaussian component
            # Shape: [batch_size, n_gaussians]
            log_probs = torch.zeros(batch_size, self.n_gaussians, device=device, dtype=dtype)

            for k in range(self.n_gaussians):
                # Compute Gaussian log-probability
                # log N(x | μ, σ²) = -0.5 * [log(2π) + log(σ²) + (x-μ)²/σ²]
                diff = x - centers[k]  # [batch_size, dims]
                normalized_diff = diff / stds[k]  # [batch_size, dims]

                # Sum over dimensions
                squared_dist = torch.sum(normalized_diff**2, dim=1)  # [batch_size]
                log_det = torch.sum(torch.log(stds[k] ** 2))  # scalar

                dims = x.shape[1]
                log_prob_k = -0.5 * (dims * np.log(2 * np.pi) + log_det + squared_dist)

                # Add log weight
                log_probs[:, k] = torch.log(weights[k]) + log_prob_k

            # Log-sum-exp trick for numerical stability
            # log(Σ exp(x_i)) = max(x_i) + log(Σ exp(x_i - max(x_i)))
            max_log_prob = torch.max(log_probs, dim=1, keepdim=True)[0]
            log_mixture = max_log_prob + torch.log(
                torch.sum(torch.exp(log_probs - max_log_prob), dim=1, keepdim=True)
            )

            # Return negative log-likelihood
            return -log_mixture.squeeze(1)

        super().__init__(dims=dims, function=mixture_function, **kwargs)

    def get_bounds(self, dims: int) -> Bounds:
        """Get bounds for this instance."""
        low, high = self.bounds_range
        bounds = [(low, high) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self) -> torch.Tensor:
        """Return the center of the highest-weighted Gaussian."""
        best_idx = torch.argmax(self.weights)
        return self.centers[best_idx]

    @property
    def benchmark(self) -> torch.Tensor:
        """Return the optimal value (negative log-likelihood at best center)."""
        # At the center of the highest-weighted Gaussian, the negative log-likelihood
        # is approximately -log(weight_max) (ignoring normalization constants)
        torch.max(self.weights)

        # Evaluate the actual function at the best center
        best_center = self.best_state.unsqueeze(0)  # [1, dims]
        return self.function(best_center)[0]

    def get_component_info(self) -> dict:
        """Return information about the mixture components."""
        return {
            "n_gaussians": self.n_gaussians,
            "centers": self.centers.clone(),
            "stds": self.stds.clone(),
            "weights": self.weights.clone(),
            "dims": self.dims,
            "bounds_range": self.bounds_range,
        }


ALL_BENCHMARKS = [
    Sphere,
    Rastrigin,
    EggHolder,
    StyblinskiTang,
    HolderTable,
    Easom,
    MixtureOfGaussians,
]
