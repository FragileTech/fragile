from collections.abc import Callable
import math
from typing import ClassVar

import einops
import holoviews as hv
from numba import jit
import numpy as np
import pandas as pd
import panel as pn
import param
import torch

from fragile.bounds import Bounds, NumpyBounds, TorchBounds
from fragile.core.panel_model import INPUT_WIDTH, PanelModel


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


class OptimBenchmark(PanelModel):
    """Base class for optimization benchmark functions.

    Attributes:
        dims: Spatial dimensionality of the benchmark.
        function: Callable that evaluates the benchmark at given positions.
        bounds: State space bounds for the benchmark.
    """

    dims = param.Integer(doc="OptimBenchmark: Spatial dimensionality")
    function = param.Callable(doc="OptimBenchmark: Benchmark evaluation function")
    bounds = param.Parameter(doc="OptimBenchmark: State space bounds")

    # UI configuration
    _n_widget_columns = param.Integer(default=2, doc="Number of widget columns")
    _max_widget_width = param.Integer(default=600, doc="Max widget width")

    def __init__(
        self,
        dims: int,
        function: Callable,
        bounds: Bounds | TorchBounds | NumpyBounds | None = None,
        **kwargs,
    ):
        if bounds is None:
            bounds = self.get_bounds(dims=dims)
        super().__init__(dims=dims, function=function, bounds=bounds, **kwargs)

    @property
    def widgets(self) -> dict[str, dict]:
        """Widget configurations for benchmark parameters."""
        return {
            "dims": {
                "type": pn.widgets.IntInput,
                "width": INPUT_WIDTH,
                "name": "Dimensions",
            },
        }

    @property
    def widget_parameters(self) -> list[str]:
        """Parameters to display in UI (only dims is editable)."""
        return ["dims"]

    def __call__(self, x):
        return self.function(x)

    def sample(self, n_samples):
        return self.bounds.sample(n_samples)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.bounds.shape

    @property
    def benchmark(self) -> torch.Tensor | None:
        """Optimal value for this benchmark."""
        return None

    @property
    def best_state(self) -> torch.Tensor | None:
        """Optimal state for this benchmark."""
        return None

    @staticmethod
    def get_bounds(dims: int) -> Bounds | TorchBounds | NumpyBounds:
        raise NotImplementedError

    def show(
        self,
        show_optimum: bool = True,
        show_density: bool = True,
        show_contours: bool = True,
        n_cells: int = 100,
        dims_to_show: tuple[int, int] = (0, 1),
    ) -> hv.Overlay:
        """Create an interactive HoloViews visualization of the benchmark function.

        For 2D benchmarks, creates a complete visualization of the function landscape.
        For higher-dimensional benchmarks, shows a 2D slice with other dimensions
        fixed at their optimal values (if known) or at the center of bounds.

        Args:
            show_optimum: Whether to show the global optimum as a red star
            show_density: Whether to show the density heatmap with viridis colormap
            show_contours: Whether to show contour lines in black
            n_cells: Grid resolution (n_cells × n_cells)
            dims_to_show: Which dimensions to visualize (default: first two)

        Returns:
            HoloViews overlay with the requested visualization components

        Example:
            >>> benchmark = Rastrigin(dims=2)
            >>> plot = benchmark.show(show_contours=True, n_cells=200)
            >>> plot  # Display in Jupyter notebook
        """
        # Validate dims_to_show
        dim_i, dim_j = dims_to_show
        if dim_i >= self.dims or dim_j >= self.dims:
            msg = f"dims_to_show {dims_to_show} out of range for dims={self.dims}"
            raise ValueError(msg)

        # Extract bounds for the dimensions to visualize
        # Handle different Bounds types (Bounds, TorchBounds, NumpyBounds)
        if hasattr(self.bounds, "low") and hasattr(self.bounds, "high"):
            # TorchBounds or NumpyBounds
            bounds_low = self.bounds.low
            bounds_high = self.bounds.high
            if isinstance(bounds_low, torch.Tensor):
                bounds_low = bounds_low.cpu().numpy()
                bounds_high = bounds_high.cpu().numpy()
        else:
            # Bounds class - extract from shape
            # Sample to get the bounds structure
            sample = self.bounds.sample(1)
            if isinstance(sample, torch.Tensor):
                sample = sample.cpu().numpy()
            # Use a heuristic: sample many points and find min/max
            # This is a fallback for generic Bounds class
            samples = self.bounds.sample(1000)
            if isinstance(samples, torch.Tensor):
                samples = samples.cpu().numpy()
            bounds_low = np.min(samples, axis=0)
            bounds_high = np.max(samples, axis=0)

        x_range = (bounds_low[dim_i], bounds_high[dim_i])
        y_range = (bounds_low[dim_j], bounds_high[dim_j])

        # Create grid
        x_grid = np.linspace(x_range[0], x_range[1], n_cells)
        y_grid = np.linspace(y_range[0], y_range[1], n_cells)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Prepare evaluation points
        if self.dims > 2:
            # For higher dimensions, fix non-visualized dimensions
            if self.best_state is not None:
                base_point = self.best_state.cpu().numpy()
            else:
                # Use midpoint of bounds
                base_point = (bounds_low + bounds_high) / 2

            # Create grid points by varying only the two dimensions
            grid_points = np.tile(base_point, (n_cells * n_cells, 1))
            grid_points[:, dim_i] = X.ravel()
            grid_points[:, dim_j] = Y.ravel()

            # Build title with fixed dimension info
            fixed_dims_info = ", ".join(
                f"x_{k}={base_point[k]:.1f}" for k in range(self.dims) if k not in dims_to_show
            )
            title = f"{self.__class__.__name__} (dims {dim_i},{dim_j} | {fixed_dims_info})"
        else:
            # For 2D, use the grid directly
            grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
            title = self.__class__.__name__

        # Evaluate function on grid
        grid_tensor = torch.from_numpy(grid_points).float()
        with torch.no_grad():
            values = self.function(grid_tensor).cpu().numpy().reshape(X.shape)

        # Handle non-finite values
        values = np.where(np.isfinite(values), values, np.nan)

        # Create base image (always needed for contours even if not displayed)
        base_image = hv.Image(
            (x_grid, y_grid, values),
            kdims=[f"x_{dim_i}", f"x_{dim_j}"],
            vdims="value",
        )

        # Create visualization layers
        layers = []

        # Density plot
        if show_density:
            density = base_image.clone(label="Density").opts(
                cmap="viridis",
                colorbar=True,
                colorbar_position="right",
                width=720,
                height=620,
                title=title,
            )
            layers.append(density)

        # Contour lines
        if show_contours:
            from holoviews.operation import contours as hv_contours

            contours = (
                hv_contours(base_image, levels=10)
                .relabel("Contours")
                .opts(
                    color="black",
                    line_width=1,
                )
            )
            layers.append(contours)

        # Optimum point
        if show_optimum and self.best_state is not None:
            best = self.best_state.cpu().numpy()
            opt_x, opt_y = best[dim_i], best[dim_j]

            optimum = hv.Points(
                [(opt_x, opt_y)],
                kdims=[f"x_{dim_i}", f"x_{dim_j}"],
                label="Optimum",
            ).opts(
                marker="star",
                color="red",
                size=20,
                line_color="white",
                line_width=2,
            )
            layers.append(optimum)

        # Return overlay
        if not layers:
            # If no layers requested, return placeholder
            return hv.Text(0, 0, "No visualization layers enabled").opts(
                xlim=x_range,
                ylim=y_range,
                width=720,
                height=620,
            )

        # Build overlay
        overlay = layers[0]
        for layer in layers[1:]:
            overlay *= layer

        # Apply legend positioning only if we have multiple layers (actual overlay)
        if len(layers) > 1:
            # Apply overlay options including legend positioning
            return overlay.opts(
                legend_position="right",  # Places legend outside to the right
                legend_offset=(10, 0),  # Additional offset for spacing from plot
            )
        # Single layer - just return it (no legend positioning needed)
        return overlay


class Sphere(OptimBenchmark):
    def __init__(self, dims: int, **kwargs):
        super().__init__(dims=dims, function=sphere, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-1000, 1000) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def benchmark(self) -> torch.Tensor:
        return torch.tensor(0.0)

    @property
    def best_state(self) -> torch.Tensor:
        return torch.zeros(self.shape)


class Rastrigin(OptimBenchmark):
    def __init__(self, dims: int, **kwargs):
        super().__init__(dims=dims, function=rastrigin, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-5.12, 5.12) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def benchmark(self) -> torch.Tensor:
        return torch.tensor(0.0)

    @property
    def best_state(self) -> torch.Tensor:
        return torch.zeros(self.shape)


class EggHolder(OptimBenchmark):
    def __init__(self, dims: int | None = None, **kwargs):  # noqa: ARG002
        super().__init__(dims=2, function=eggholder, **kwargs)

    @staticmethod
    def get_bounds(dims=None):  # noqa: ARG004
        bounds = [(-512.0, 512.0), (-512.0, 512.0)]
        # bounds = [(1, 512.0), (1, 512.0)]
        return Bounds.from_tuples(bounds)

    @property
    def benchmark(self) -> torch.Tensor:
        return torch.tensor(-959.64066271)

    @property
    def best_state(self) -> torch.Tensor:
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
    minima: ClassVar[dict[str, float]] = {
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

    n_atoms = param.Integer(default=10, doc="Number of atoms")

    def __init__(self, n_atoms: int = 10, dims=None, **kwargs):  # noqa: ARG002
        calculated_dims = 3 * n_atoms
        super().__init__(dims=calculated_dims, function=lennard_jones, n_atoms=n_atoms, **kwargs)

    @property
    def benchmark(self) -> list[torch.Tensor | float]:
        return [torch.zeros(self.n_atoms * 3), self.minima.get(str(int(self.n_atoms)), 0)]

    @staticmethod
    def get_bounds(dims):
        bounds = [(-15, 15) for _ in range(dims)]
        return Bounds.from_tuples(bounds)


class Constant(OptimBenchmark):
    """Constant (zero) potential benchmark.

    This benchmark returns U(x) = 0 for all x, useful for testing pure
    diffusion dynamics without any potential forces.
    """

    def __init__(self, dims: int, **kwargs):
        """Initialize constant potential.

        Args:
            dims: Dimensionality of the space
            **kwargs: Additional parameters (ignored)
        """

        def constant_potential(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

        super().__init__(dims=dims, function=constant_potential, **kwargs)

    @property
    def benchmark(self) -> torch.Tensor:
        return torch.tensor(0.0)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-10.0, 10.0) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        # For constant potential, all states are equally optimal
        return torch.zeros(self.shape)


class StochasticGaussian(OptimBenchmark):
    """Stochastic Gaussian noise benchmark.

    This benchmark returns U(x) = N(0, 1) sampled independently each time it
    is evaluated, providing a purely stochastic background reward signal with
    no spatial structure.

    The potential is sampled fresh on each call, making this useful for testing
    algorithm behavior under purely random reward signals where no optimization
    strategy should emerge.

    Mathematical definition:
        U(x) ~ N(0, σ²)  (sampled independently for each evaluation)

    Args:
        dims: Dimensionality of the space
        std: Standard deviation of the Gaussian noise (default: 1.0)

    Important:
        Since the potential has no spatial structure (pure noise), it has no
        meaningful gradient. When using this benchmark with EuclideanGas,
        set use_potential_force=False in the KineticOperator to avoid
        attempting gradient computation.
    """

    std = param.Number(default=1.0, doc="Standard deviation of Gaussian noise")

    def __init__(self, dims: int, std: float = 1.0, **kwargs):
        """Initialize stochastic Gaussian benchmark.

        Args:
            dims: Dimensionality of the space
            std: Standard deviation of Gaussian noise
            **kwargs: Additional parameters passed to OptimBenchmark
        """
        # Capture std in closure for stochastic function
        _std = std

        def stochastic_gaussian(x: torch.Tensor) -> torch.Tensor:
            """Sample N(0, σ²) independently for each walker on each call."""
            return torch.randn(x.shape[0], dtype=x.dtype, device=x.device) * _std

        super().__init__(dims=dims, function=stochastic_gaussian, std=std, **kwargs)

    @property
    def benchmark(self) -> torch.Tensor:
        # Expected value of stochastic potential
        return torch.tensor(0.0)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-10.0, 10.0) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        # No optimal state for stochastic potential
        return torch.zeros(self.shape)


class FluidBenchmark(OptimBenchmark):
    """Base class for fluid dynamics benchmarks.

    Extends OptimBenchmark with fluid-specific functionality:
    - Initial velocity field generation via get_initial_conditions()
    - Validation metrics computation via compute_validation_metrics()

    Fluid benchmarks are used to validate that the viscous-coupled Euclidean Gas
    can simulate incompressible fluid behavior consistent with Navier-Stokes equations.
    """

    bounds_extent = param.Number(default=np.pi, doc="Half-width of spatial domain")

    def get_initial_conditions(
        self, N: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate initial positions and velocities for fluid simulation.

        Args:
            N: Number of walkers
            device: PyTorch device
            dtype: PyTorch dtype

        Returns:
            x_init: Initial positions [N, d]
            v_init: Initial velocities [N, d]
        """
        raise NotImplementedError

    def compute_validation_metrics(self, history, t_idx: int, params: dict) -> list:
        """Compute validation metrics at given time index.

        Args:
            history: RunHistory object from simulation
            t_idx: Time index to evaluate
            params: Dictionary of simulation parameters

        Returns:
            List of ValidationMetrics objects
        """
        raise NotImplementedError


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

    n_gaussians = param.Integer(default=3, doc="Number of Gaussian components")
    centers = param.Parameter(default=None, doc="Gaussian centers [n_gaussians, dims]")
    stds = param.Parameter(default=None, doc="Standard deviations [n_gaussians, dims]")
    weights = param.Parameter(default=None, doc="Mixture weights [n_gaussians]")
    bounds_range = param.Tuple(default=(-10.0, 10.0), doc="Bounds for each dimension")

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
        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Initialize or validate centers
        if centers is None:
            # Random centers within bounds
            low, high = bounds_range
            centers_tensor = torch.rand(n_gaussians, dims) * (high - low) + low
        else:
            centers_tensor = torch.as_tensor(centers, dtype=torch.float32)
            if centers_tensor.shape != (n_gaussians, dims):
                msg = (
                    f"Centers shape {centers_tensor.shape} doesn't match "
                    f"(n_gaussians={n_gaussians}, dims={dims})"
                )
                raise ValueError(msg)

        # Initialize or validate standard deviations
        if stds is None:
            # Random stds between 0.1 and 2.0
            stds_tensor = torch.rand(n_gaussians, dims) * 1.9 + 0.1
        else:
            stds_tensor = torch.as_tensor(stds, dtype=torch.float32)
            if stds_tensor.shape != (n_gaussians, dims):
                msg = (
                    f"Stds shape {stds_tensor.shape} doesn't match "
                    f"(n_gaussians={n_gaussians}, dims={dims})"
                )
                raise ValueError(msg)
            if (stds_tensor <= 0).any():
                msg = "All standard deviations must be positive"
                raise ValueError(msg)

        # Initialize or validate weights
        if weights is None:
            # Uniform weights
            weights_tensor = torch.ones(n_gaussians) / n_gaussians
        else:
            weights_tensor = torch.as_tensor(weights, dtype=torch.float32)
            if weights_tensor.shape != (n_gaussians,):
                msg = (
                    f"Weights shape {weights_tensor.shape} doesn't match n_gaussians={n_gaussians}"
                )
                raise ValueError(msg)
            if (weights_tensor < 0).any():
                msg = "All weights must be non-negative"
                raise ValueError(msg)
            # Normalize weights
            weights_tensor /= weights_tensor.sum()

        # Create the mixture function - must capture local variables, not self
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
            centers = centers_tensor.to(device=device, dtype=dtype)
            stds = stds_tensor.to(device=device, dtype=dtype)
            weights = weights_tensor.to(device=device, dtype=dtype)

            # Compute log-probabilities for each Gaussian component
            # Shape: [batch_size, n_gaussians]
            log_probs = torch.zeros(batch_size, n_gaussians, device=device, dtype=dtype)

            for k in range(n_gaussians):
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

        # Create bounds
        low, high = bounds_range
        bounds_tuples = [(low, high) for _ in range(dims)]
        bounds_obj = Bounds.from_tuples(bounds_tuples)

        # Call parent init with all parameters
        super().__init__(
            dims=dims,
            function=mixture_function,
            bounds=bounds_obj,
            n_gaussians=n_gaussians,
            centers=centers_tensor,
            stds=stds_tensor,
            weights=weights_tensor,
            bounds_range=bounds_range,
            **kwargs,
        )

    @staticmethod
    def get_bounds(dims: int) -> Bounds:
        """Get default bounds (not used for instances, kept for compatibility)."""
        bounds = [(-10.0, 10.0) for _ in range(dims)]
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


class TaylorGreenVortex(FluidBenchmark):
    """Taylor-Green vortex: 2D decaying vortex with analytical solution.

    The Taylor-Green vortex is a classical test case for incompressible
    Navier-Stokes solvers with known analytical solution:

    Stream function: ψ(x,y,t) = A·exp(-2νt)·sin(x)·sin(y)
    Velocity: u = -∂ψ/∂y, v = ∂ψ/∂x
    Vorticity: ω = ∇²ψ = -2A·exp(-2νt)·sin(x)·sin(y)

    The kinetic energy decays exponentially: E(t) = E₀·exp(-2νt)

    Reference: Taylor & Green (1937), "Mechanism of the production of small
    eddies from large ones", Proc. R. Soc. Lond. A 158, 499–521.
    """

    amplitude = param.Number(default=1.0, doc="Amplitude of initial vortex")

    def __init__(self, dims: int | None = None, amplitude: float = 1.0, **kwargs):
        """Initialize Taylor-Green benchmark.

        Args:
            dims: Spatial dimension (must be 2, ignored if provided)
            amplitude: Amplitude A of initial vortex (default: 1.0)
            **kwargs: Additional parameters
        """

        def zero_potential(x: torch.Tensor) -> torch.Tensor:
            # Return zero potential while maintaining autograd graph connection
            # Using x * 0.0 instead of torch.zeros() ensures gradients can be computed
            return x[:, 0] * 0.0

        bounds = TorchBounds(low=torch.full((2,), -np.pi), high=torch.full((2,), np.pi))

        super().__init__(
            dims=2,
            function=zero_potential,
            bounds=bounds,
            bounds_extent=np.pi,
            amplitude=amplitude,
            **kwargs,
        )

    @staticmethod
    def get_bounds(dims: int | None = None):
        bounds = [(-np.pi, np.pi), (-np.pi, np.pi)]
        return Bounds.from_tuples(bounds)

    def get_initial_conditions(
        self, N: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate initial conditions from Taylor-Green velocity field.

        Positions: Uniformly distributed in [-π, π]²
        Velocities: Sampled from analytical velocity field at t=0
        """
        # Uniform positions in [-π, π]²
        x_init = torch.rand(N, 2, device=device, dtype=dtype) * 2 * np.pi - np.pi

        # Compute velocities from stream function ψ = A·sin(x)·sin(y)
        # u = -∂ψ/∂y = -A·sin(x)·cos(y)
        # v = ∂ψ/∂x = A·cos(x)·sin(y)
        u = -self.amplitude * torch.sin(x_init[:, 0]) * torch.cos(x_init[:, 1])
        v = self.amplitude * torch.cos(x_init[:, 0]) * torch.sin(x_init[:, 1])

        v_init = torch.stack([u, v], dim=1)

        return x_init, v_init

    def compute_validation_metrics(self, history, t_idx: int, params: dict) -> list:
        """Compute energy decay validation for Taylor-Green.

        The kinetic energy should decay as E(t) = E₀·exp(-2νt) where
        ν is the effective kinematic viscosity.
        """
        from fragile.experiments.fluid_utils import ValidationMetrics

        # Extract velocities at t_idx
        v = history.v_final[t_idx]  # [N, 2]

        # Compute kinetic energy E = (1/N) Σ ||vi||²
        E_measured = torch.mean(torch.sum(v**2, dim=1)).item()

        # Initial energy (t=0)
        v0 = history.v_final[0]
        E0 = torch.mean(torch.sum(v0**2, dim=1)).item()

        # Theoretical energy decay E(t) = E₀·exp(-2νt)
        dt = params.get("delta_t", 0.02)
        t = t_idx * dt

        # Effective viscosity: ν_eff ≈ nu * viscous_length_scale²
        nu = params.get("nu", 1.0)
        length_scale = params.get("viscous_length_scale", 0.8)
        nu_eff = nu * length_scale**2

        E_theory = E0 * np.exp(-2 * nu_eff * t)

        # Compute relative error
        rel_error = abs(E_measured - E_theory) / E_theory if E_theory > 1e-10 else 0.0

        passed = rel_error < 0.1  # 10% tolerance

        return [
            ValidationMetrics(
                metric_name="Energy Decay",
                measured_value=E_measured,
                theoretical_value=E_theory,
                tolerance=0.1,
                passed=passed,
                description=f"E(t)/{E0:.3f} = {E_measured / E0:.4f} vs theory {E_theory / E0:.4f}",
            )
        ]


class LidDrivenCavity(FluidBenchmark):
    """Lid-driven cavity: Flow in square cavity with moving top wall.

    The lid-driven cavity is a classic CFD benchmark where flow is driven
    by a lid moving with constant velocity along the top boundary. The
    problem exhibits:
    - Primary recirculation vortex in center
    - Secondary corner vortices (at high Re)
    - Well-documented reference solutions

    Reference: Ghia, Ghia & Shin (1982), "High-Re solutions for incompressible
    flow using the Navier-Stokes equations and a multigrid method",
    J. Comput. Phys. 48, 387-411.
    """

    reynolds_number = param.Number(default=100.0, doc="Reynolds number Re = U·L/ν")
    lid_velocity = param.Number(default=1.0, doc="Velocity of top lid (U)")

    def __init__(
        self,
        dims: int | None = None,
        reynolds_number: float = 100.0,
        lid_velocity: float = 1.0,
        **kwargs,
    ):
        """Initialize lid-driven cavity benchmark.

        Args:
            dims: Spatial dimension (must be 2, ignored if provided)
            reynolds_number: Reynolds number Re = U·L/ν
            lid_velocity: Velocity of top lid (U)
            **kwargs: Additional parameters
        """

        def wall_potential(x: torch.Tensor) -> torch.Tensor:
            """Soft wall potential using exponential repulsion."""
            strength = 10.0
            width = 0.05

            # Distance to walls
            d_left = x[:, 0]
            d_right = 1.0 - x[:, 0]
            d_bottom = x[:, 1]
            d_top = 1.0 - x[:, 1]

            # Exponential repulsion (maintain autograd graph connection)
            U = strength * torch.exp(-d_left / width)
            U += strength * torch.exp(-d_right / width)
            U += strength * torch.exp(-d_bottom / width)
            return U + strength * torch.exp(-d_top / width)

        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))

        super().__init__(
            dims=2,
            function=wall_potential,
            bounds=bounds,
            bounds_extent=0.5,
            reynolds_number=reynolds_number,
            lid_velocity=lid_velocity,
            **kwargs,
        )

    @staticmethod
    def get_bounds(dims: int | None = None):
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        return Bounds.from_tuples(bounds)

    def get_initial_conditions(
        self, N: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate initial conditions: uniform positions, zero velocities."""
        # Uniform positions in [0, 1]²
        x_init = torch.rand(N, 2, device=device, dtype=dtype)

        # Zero initial velocities (fluid at rest)
        v_init = torch.zeros(N, 2, device=device, dtype=dtype)

        return x_init, v_init

    def compute_validation_metrics(self, history, t_idx: int, params: dict) -> list:
        """Compute cavity flow validation metrics."""
        from fragile.experiments.fluid_utils import ValidationMetrics

        # For now, just check that flow has developed
        v = history.v_final[t_idx]
        v_mag = torch.mean(torch.linalg.vector_norm(v, dim=1)).item()

        return [
            ValidationMetrics(
                metric_name="Flow Development",
                measured_value=v_mag,
                theoretical_value=self.lid_velocity / 2,  # Rough estimate
                tolerance=0.5,
                passed=v_mag > 0.01,  # Just check flow exists
                description=f"Mean velocity magnitude: {v_mag:.4f}",
            )
        ]


class KelvinHelmholtzInstability(FluidBenchmark):
    """Kelvin-Helmholtz instability: Shear layer with vortex roll-up.

    The Kelvin-Helmholtz instability occurs at the interface between two
    fluid layers moving at different velocities. The shear creates an
    instability that grows exponentially and eventually rolls up into
    coherent vortices.

    This benchmark demonstrates:
    - Instability growth from small perturbations
    - Vortex formation and roll-up
    - Mixing layer evolution
    - Beautiful visual demonstration of fluid instability
    """

    shear_velocity = param.Number(default=1.0, doc="Velocity difference between layers (ΔU)")
    layer_thickness = param.Number(default=0.2, doc="Thickness of shear layer (δ)")

    def __init__(
        self,
        dims: int | None = None,
        shear_velocity: float = 1.0,
        layer_thickness: float = 0.2,
        **kwargs,
    ):
        """Initialize Kelvin-Helmholtz benchmark.

        Args:
            dims: Spatial dimension (must be 2, ignored if provided)
            shear_velocity: Velocity difference between layers (ΔU)
            layer_thickness: Thickness of shear layer (δ)
            **kwargs: Additional parameters
        """

        def zero_potential(x: torch.Tensor) -> torch.Tensor:
            # Return zero potential while maintaining autograd graph connection
            # Using x * 0.0 instead of torch.zeros() ensures gradients can be computed
            return x[:, 0] * 0.0

        bounds = TorchBounds(low=torch.full((2,), -np.pi), high=torch.full((2,), np.pi))

        super().__init__(
            dims=2,
            function=zero_potential,
            bounds=bounds,
            bounds_extent=np.pi,
            shear_velocity=shear_velocity,
            layer_thickness=layer_thickness,
            **kwargs,
        )

    @staticmethod
    def get_bounds(dims: int | None = None):
        bounds = [(-np.pi, np.pi), (-np.pi, np.pi)]
        return Bounds.from_tuples(bounds)

    def get_initial_conditions(
        self, N: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate initial conditions: two layers with velocity shear.

        Initial velocity profile: u(y) = U·tanh(y/δ)
        Small sinusoidal perturbation added to trigger instability
        """
        # Uniform positions in [-π, π]²
        x_init = torch.rand(N, 2, device=device, dtype=dtype) * 2 * np.pi - np.pi

        # Base velocity profile: u(y) = U·tanh(y/δ)
        y = x_init[:, 1]
        u_base = self.shear_velocity * torch.tanh(y / self.layer_thickness)

        # Add sinusoidal perturbation to trigger instability
        k_unstable = 1.0  # Most unstable mode
        amplitude = 0.1 * self.shear_velocity
        perturbation_v = amplitude * torch.cos(k_unstable * x_init[:, 0])

        # Construct velocity field
        u = u_base
        v = perturbation_v

        v_init = torch.stack([u, v], dim=1)

        return x_init, v_init

    def compute_validation_metrics(self, history, t_idx: int, params: dict) -> list:
        """Compute K-H validation metrics (primarily qualitative)."""
        from fragile.experiments.fluid_utils import ValidationMetrics

        # Measure vorticity magnitude as proxy for vortex development
        v = history.v_final[t_idx]
        v_rms = torch.sqrt(torch.mean(v**2)).item()

        return [
            ValidationMetrics(
                metric_name="Vortex Development",
                measured_value=v_rms,
                theoretical_value=None,  # Qualitative
                tolerance=float("inf"),
                passed=v_rms > 0.1,  # Just check instability developed
                description=f"RMS velocity: {v_rms:.4f} (qualitative check)",
            )
        ]


ALL_BENCHMARKS = [
    Sphere,
    Rastrigin,
    EggHolder,
    StyblinskiTang,
    HolderTable,
    Easom,
    MixtureOfGaussians,
    Constant,
    StochasticGaussian,
    TaylorGreenVortex,
    LidDrivenCavity,
    KelvinHelmholtzInstability,
]


# Benchmark name mapping for UI
BENCHMARK_NAMES = {
    "Sphere": Sphere,
    "Rastrigin": Rastrigin,
    "EggHolder": EggHolder,
    "Styblinski-Tang": StyblinskiTang,
    "Holder Table": HolderTable,
    "Easom": Easom,
    "Mixture of Gaussians": MixtureOfGaussians,
    "Lennard-Jones": LennardJones,
    "Constant (Zero)": Constant,
    "Stochastic Gaussian": StochasticGaussian,
    "Taylor-Green Vortex": TaylorGreenVortex,
    "Lid-Driven Cavity": LidDrivenCavity,
    "Kelvin-Helmholtz Instability": KelvinHelmholtzInstability,
}


def create_benchmark_background(
    benchmark: OptimBenchmark,
    bounds_range: tuple[float, float] = (-6.0, 6.0),
    resolution: int = 200,
    beta_bg: float = 1.0,
) -> hv.Image:
    """Create background density visualization for a benchmark function.

    Args:
        benchmark: Benchmark function instance
        bounds_range: (min, max) spatial bounds for visualization
        resolution: Grid resolution for background density
        beta_bg: Inverse temperature for density = exp(-beta * U)

    Returns:
        HoloViews Image with density background
    """
    if benchmark.dims != 2:
        # For non-2D, return empty background
        return hv.Image(
            ([], [], np.zeros((0, 0))),
            kdims=["x₁", "x₂"],
            vdims="density",
        ).opts(
            cmap="Greys",
            alpha=0.35,
            colorbar=False,
        )

    grid_axis = np.linspace(bounds_range[0], bounds_range[1], resolution)
    X, Y = np.meshgrid(grid_axis, grid_axis)
    grid_points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)

    with torch.no_grad():
        U_grid = benchmark(grid_points).cpu().numpy().reshape(X.shape)

    density = np.exp(-beta_bg * U_grid)
    density /= np.max(density) if np.max(density) > 0 else 1.0

    return hv.Image(
        (grid_axis, grid_axis, density),
        kdims=["x₁", "x₂"],
        vdims="density",
    ).opts(
        cmap="Greys",
        alpha=0.35,
        colorbar=False,
        width=720,
        height=620,
    )


def prepare_benchmark_for_explorer(
    benchmark_name: str,
    dims: int = 2,
    bounds_range: tuple[float, float] = (-6.0, 6.0),
    resolution: int = 200,
    beta_bg: float = 1.0,
    **benchmark_kwargs,
) -> tuple[OptimBenchmark, hv.Image, hv.Points]:
    """Prepare benchmark function for use with SwarmExplorer.

    Args:
        benchmark_name: Name of benchmark (key from BENCHMARK_NAMES)
        dims: Spatial dimension
        bounds_range: (min, max) bounds for spatial domain
        resolution: Grid resolution for background
        beta_bg: Inverse temperature for density = exp(-beta * U)
        **benchmark_kwargs: Additional parameters for benchmark initialization

    Returns:
        Tuple of (benchmark, background_image, mode_points)
        - benchmark: OptimBenchmark instance (callable, with bounds attribute)
        - background_image: HoloViews Image for visualization
        - mode_points: HoloViews Points showing target modes
    """
    benchmark_cls = BENCHMARK_NAMES[benchmark_name]

    # Create benchmark instance
    if benchmark_cls == MixtureOfGaussians:
        # Special handling for MixtureOfGaussians
        n_gaussians = benchmark_kwargs.get("n_gaussians", 3)
        seed = benchmark_kwargs.get("seed", 42)
        centers = benchmark_kwargs.get("centers", None)
        stds = benchmark_kwargs.get("stds", None)
        weights = benchmark_kwargs.get("weights", None)

        benchmark = MixtureOfGaussians(
            dims=dims,
            n_gaussians=n_gaussians,
            centers=centers,
            stds=stds,
            weights=weights,
            bounds_range=bounds_range,
            seed=seed,
        )

        # Create mode points for MixtureOfGaussians
        if dims == 2:
            mode_df = pd.DataFrame({
                "x₁": benchmark.centers[:, 0].cpu().numpy(),
                "x₂": benchmark.centers[:, 1].cpu().numpy(),
                "size": 50 * benchmark.weights.cpu().numpy(),
            })

            mode_points = hv.Points(
                mode_df,
                kdims=["x₁", "x₂"],
                vdims="size",
                label="Target Modes",
            ).opts(
                size="size",
                color="red",
                marker="star",
                line_color="white",
                line_width=2,
                alpha=0.8,
            )
        else:
            mode_points = hv.Points([], kdims=["x₁", "x₂"])

    elif benchmark_cls == LennardJones:
        # Special handling for LennardJones
        n_atoms = benchmark_kwargs.get("n_atoms", 10)
        benchmark = LennardJones(n_atoms=n_atoms)
        # No mode points for Lennard-Jones
        mode_points = hv.Points([], kdims=["x₁", "x₂"])

    elif benchmark_cls in {EggHolder, Easom, HolderTable}:
        # Fixed-dimension benchmarks
        benchmark = benchmark_cls()
        # Add mode points at known optima
        best_state = benchmark.best_state.cpu().numpy()
        if len(best_state) == 2:
            mode_points = hv.Points(
                [(best_state[0], best_state[1])],
                kdims=["x₁", "x₂"],
                label="Global Optimum",
            ).opts(
                size=15,
                color="red",
                marker="x",
                line_color="white",
                line_width=2,
            )
        else:
            mode_points = hv.Points([], kdims=["x₁", "x₂"])

    else:
        # Standard benchmarks (Sphere, Rastrigin, etc.)
        benchmark = benchmark_cls(dims=dims)

        # Add mode point at origin for benchmarks with known optima
        if dims == 2 and hasattr(benchmark, "best_state"):
            best_state = benchmark.best_state.cpu().numpy()
            mode_points = hv.Points(
                [(best_state[0], best_state[1])],
                kdims=["x₁", "x₂"],
                label="Global Optimum",
            ).opts(
                size=15,
                color="red",
                marker="x",
                line_color="white",
                line_width=2,
            )
        else:
            mode_points = hv.Points([], kdims=["x₁", "x₂"])

    # Create background
    background = create_benchmark_background(
        benchmark=benchmark,
        bounds_range=bounds_range,
        resolution=resolution,
        beta_bg=beta_bg,
    )

    return benchmark, background, mode_points


class BenchmarkSelector(param.Parameterized):
    """Interactive dashboard for selecting and configuring benchmark functions.

    This class provides a Panel-based UI for selecting benchmark functions,
    configuring their parameters, and visualizing the resulting potential landscape.

    Example:
        >>> selector = BenchmarkSelector()
        >>> dashboard = selector.panel()
        >>> dashboard.show()
    """

    # Benchmark selection
    benchmark_name = param.ObjectSelector(
        default="Mixture of Gaussians",
        objects=list(BENCHMARK_NAMES.keys()),
        doc="Select benchmark function",
    )

    # Common parameters
    dims = param.Integer(default=2, bounds=(2, 10), doc="Spatial dimension")
    bounds_extent = param.Number(
        default=6.0, bounds=(1.0, 20.0), doc="Spatial bounds half-width (±extent)"
    )
    resolution = param.Integer(default=200, bounds=(50, 500), doc="Background grid resolution")

    # MixtureOfGaussians parameters
    n_gaussians = param.Integer(default=3, bounds=(1, 10), doc="Number of Gaussian modes")
    seed = param.Integer(default=42, bounds=(0, 9999), doc="Random seed for MoG")

    # LennardJones parameters
    n_atoms = param.Integer(default=10, bounds=(2, 30), doc="Number of atoms")

    def __init__(self, **params):
        """Initialize BenchmarkSelector.

        Args:
            **params: Override default parameter values
        """
        super().__init__(**params)

        # Current benchmark instance
        self.current_benchmark: OptimBenchmark | None = None
        self.current_background: hv.Image | None = None
        self.current_mode_points: hv.Points | None = None

        # Status display
        self.status_pane = pn.pane.Markdown("", sizing_mode="stretch_width")

        # Visualization pane
        self.viz_pane = pn.pane.HoloViews(sizing_mode="stretch_both", min_height=600)

        # Watch parameter changes
        self.param.watch(
            self._update_benchmark,
            [
                "benchmark_name",
                "dims",
                "bounds_extent",
                "resolution",
                "n_gaussians",
                "seed",
                "n_atoms",
            ],
        )

        # Initial update
        self._update_benchmark()

    def _update_benchmark(self, *_):
        """Update benchmark instance and visualization."""
        try:
            # Determine bounds range
            bounds_range = (-self.bounds_extent, self.bounds_extent)

            # Prepare benchmark-specific kwargs
            benchmark_kwargs = {}
            if self.benchmark_name == "Mixture of Gaussians":
                benchmark_kwargs["n_gaussians"] = self.n_gaussians
                benchmark_kwargs["seed"] = self.seed
            elif self.benchmark_name == "Lennard-Jones":
                benchmark_kwargs["n_atoms"] = self.n_atoms

            # Create benchmark
            (
                self.current_benchmark,
                self.current_background,
                self.current_mode_points,
            ) = prepare_benchmark_for_explorer(
                benchmark_name=self.benchmark_name,
                dims=self.dims,
                bounds_range=bounds_range,
                resolution=self.resolution,
                **benchmark_kwargs,
            )

            # Update visualization
            if self.dims == 2:
                overlay = self.current_background * self.current_mode_points
                overlay = overlay.opts(
                    title=f"{self.benchmark_name} Potential Landscape",
                    xlim=bounds_range,
                    ylim=bounds_range,
                    width=720,
                    height=620,
                    show_grid=True,
                )
                self.viz_pane.object = overlay
            else:
                # For higher dims, show placeholder
                placeholder = hv.Text(
                    0,
                    0,
                    f"{self.benchmark_name}\n(dims={self.dims})\n"
                    "Visualization only available for 2D",
                ).opts(
                    xlim=bounds_range,
                    ylim=bounds_range,
                    width=720,
                    height=620,
                    text_font_size="14pt",
                )
                self.viz_pane.object = placeholder

            # Update status
            info = (
                f"**{self.benchmark_name}** | dims={self.dims} | "
                f"bounds=[{bounds_range[0]:.1f}, {bounds_range[1]:.1f}]"
            )
            if hasattr(self.current_benchmark, "benchmark"):
                opt_val = self.current_benchmark.benchmark
                if isinstance(opt_val, torch.Tensor):
                    info += f" | optimal value={opt_val.item():.4f}"
            self.status_pane.object = info

        except Exception as e:
            self.status_pane.object = f"**Error:** {e!s}"
            self.viz_pane.object = hv.Text(0, 0, f"Error: {e!s}").opts(width=720, height=620)

    def get_potential(self) -> OptimBenchmark:
        """Get the current potential (OptimBenchmark instance).

        Returns:
            OptimBenchmark with callable interface U(x) -> [N]
        """
        if self.current_benchmark is None:
            self._update_benchmark()
        return self.current_benchmark

    def get_benchmark(self) -> OptimBenchmark:
        """Get the current benchmark instance.

        Returns:
            Current OptimBenchmark instance
        """
        if self.current_benchmark is None:
            self._update_benchmark()
        return self.current_benchmark

    def get_background(self) -> hv.Image:
        """Get the current background visualization.

        Returns:
            HoloViews Image with background density
        """
        if self.current_background is None:
            self._update_benchmark()
        return self.current_background

    def get_mode_points(self) -> hv.Points:
        """Get the current mode points visualization.

        Returns:
            HoloViews Points with mode markers
        """
        if self.current_mode_points is None:
            self._update_benchmark()
        return self.current_mode_points

    def panel(self) -> pn.Column:
        """Create Panel dashboard for benchmark selection.

        Returns:
            Panel Column with controls and visualization
        """
        # Group parameters by benchmark type
        common_params = ["benchmark_name", "dims", "bounds_extent", "resolution"]

        # Conditional parameter panels
        mog_params = pn.Param(
            self.param,
            parameters=["n_gaussians", "seed"],
            show_name=False,
            widgets={
                "n_gaussians": pn.widgets.IntSlider,
                "seed": pn.widgets.IntInput,
            },
        )

        lj_params = pn.Param(
            self.param,
            parameters=["n_atoms"],
            show_name=False,
            widgets={"n_atoms": pn.widgets.IntSlider},
        )

        # Dynamic parameter display
        def get_specific_params(benchmark_name):
            if benchmark_name == "Mixture of Gaussians":
                return pn.Column(
                    pn.pane.Markdown("#### Mixture of Gaussians Parameters"),
                    mog_params,
                )
            if benchmark_name == "Lennard-Jones":
                return pn.Column(
                    pn.pane.Markdown("#### Lennard-Jones Parameters"),
                    lj_params,
                )
            return pn.pane.Markdown("*No additional parameters*")

        specific_params_pane = pn.bind(get_specific_params, self.param.benchmark_name)

        controls = pn.Column(
            pn.pane.Markdown("## Benchmark Selector"),
            pn.pane.Markdown("### Common Parameters"),
            pn.Param(
                self.param,
                parameters=common_params,
                show_name=False,
                widgets={
                    "benchmark_name": pn.widgets.Select,
                    "dims": pn.widgets.IntSlider,
                    "bounds_extent": pn.widgets.FloatSlider,
                    "resolution": pn.widgets.IntSlider,
                },
            ),
            specific_params_pane,
            pn.pane.Markdown("---"),
            self.status_pane,
            sizing_mode="stretch_width",
            min_width=350,
            max_width=450,
        )

        return pn.Row(
            controls,
            self.viz_pane,
            sizing_mode="stretch_both",
        )


# Module exports
# ruff: noqa: RUF022
__all__ = [
    # Benchmark function implementations
    "eggholder",
    "easom",
    "holder_table",
    "lennard_jones",
    "rastrigin",
    "rosenbrock",
    "sphere",
    "styblinski_tang",
    # Benchmark classes
    "Constant",
    "Easom",
    "EggHolder",
    "FluidBenchmark",
    "HolderTable",
    "KelvinHelmholtzInstability",
    "LennardJones",
    "LidDrivenCavity",
    "MixtureOfGaussians",
    "OptimBenchmark",
    "Rastrigin",
    "Rosenbrock",
    "Sphere",
    "StochasticGaussian",
    "StyblinskiTang",
    "TaylorGreenVortex",
    # Collections
    "ALL_BENCHMARKS",
    "BENCHMARK_NAMES",
    # Visualization utilities
    "BenchmarkSelector",
    "create_benchmark_background",
    "prepare_benchmark_for_explorer",
]
