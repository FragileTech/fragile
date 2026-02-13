"""Interactive parameter selector for Euclidean Gas swarms.

This module provides a Panel-based dashboard for interactively selecting
all parameters needed to configure an Euclidean Gas swarm in notebooks.
"""

import panel as pn
import param

from fragile.fractalai.core.benchmarks import (
    Easom,
    EggHolder,
    HolderTable,
    OptimBenchmark,
    Rastrigin,
    Rosenbrock,
    Sphere,
    StyblinskiTang,
)


class EuclideanGasParamSelector(param.Parameterized):
    """Interactive parameter selector for Euclidean Gas configuration.

    This class creates an interactive dashboard using Panel widgets that allows
    users to configure all parameters for an Euclidean Gas swarm. The dashboard
    includes sections for swarm configuration, Langevin dynamics, cloning
    mechanism, and benchmark selection.

    The epsilon parameter (ε) controls the interaction range for the ε-dependent
    spatial kernel used in companion selection. This implements the Sequential
    Stochastic Greedy Pairing Operator from the mathematical specification:
    - Smaller ε → stronger preference for nearby companions
    - Larger ε → more uniform companion selection
    - No hard bounds → can be adjusted freely for exploration

    Example:
        >>> import panel as pn
        >>> from fragile.fractalai.shaolin import EuclideanGasParamSelector
        >>> pn.extension()
        >>> selector = EuclideanGasParamSelector()
        >>> selector.epsilon = 0.8  # Adjust companion selection range
        >>> selector.panel()  # Display the dashboard
        >>> params = selector.get_params()  # Get configured parameters
        >>> gas = EuclideanGas(params)  # Initialize gas with parameters

    """

    # Swarm parameters
    n_walkers = param.Integer(default=50, bounds=(10, 1000), doc="Number of walkers in the swarm")
    dimensions = param.Integer(default=2, bounds=(1, 20), doc="Dimensionality of state space")
    device = param.Selector(default="cpu", objects=["cpu", "cuda"], doc="Computation device")
    dtype = param.Selector(
        default="float32", objects=["float32", "float64"], doc="Data type precision"
    )

    # Langevin dynamics parameters
    gamma = param.Number(
        default=1.0,
        bounds=(0.01, 10.0),
        step=0.1,
        doc="Friction coefficient (higher = faster convergence)",
    )
    beta = param.Number(
        default=2.0,
        bounds=(0.1, 10.0),
        step=0.1,
        doc="Inverse temperature (higher = less random exploration)",
    )
    delta_t = param.Number(
        default=0.1, bounds=(0.0001, 1.0), step=0.001, doc="Integration timestep"
    )
    integrator = param.Selector(
        default="baoab",
        objects=["baoab", "aboba", "babo", "obab"],
        doc="Langevin integrator type",
    )

    # Cloning parameters
    sigma_x = param.Number(
        default=0.5,
        bounds=(0.01, 5.0),
        step=0.05,
        doc="Collision radius for cloning (position space)",
    )
    epsilon = param.Number(
        default=0.5,
        softbounds=(0.01, 10.0),
        doc="Interaction range for companion selection (ε-dependent spatial kernel)",
    )
    lambda_alg = param.Number(
        default=0.1, bounds=(0.0, 1.0), step=0.01, doc="Algorithmic cloning parameter"
    )
    alpha_restitution = param.Number(
        default=0.0,
        bounds=(0.0, 1.0),
        step=0.05,
        doc="Coefficient of restitution (0=inelastic, 1=elastic)",
    )
    use_inelastic_collision = param.Boolean(
        default=True, doc="Use inelastic collision model for cloning"
    )

    # Benchmark selection
    benchmark_type = param.Selector(
        default="Rastrigin",
        objects=[
            "Sphere",
            "Rastrigin",
            "StyblinskiTang",
            "Rosenbrock",
            "EggHolder",
            "Easom",
            "HolderTable",
        ],
        doc="Optimization benchmark function",
    )

    def __init__(self, **params):
        """Initialize the parameter selector with default values."""
        super().__init__(**params)

        # Create widgets
        self._create_widgets()

        # Store potential and params for export
        self._cached_potential = None
        self._cached_params = None

    def _create_widgets(self):
        """Create Panel widgets for all parameters."""
        # Header
        self.header = pn.pane.Markdown(
            "## 🌊 Euclidean Gas Parameter Selector\n"
            "Configure all parameters for your swarm interactively.",
            styles={"background": "#f0f0f0", "padding": "10px", "border-radius": "5px"},
        )

        # Swarm configuration section
        self.swarm_section = pn.Card(
            pn.widgets.IntSlider.from_param(
                self.param.n_walkers, name="Number of Walkers (N)", width=400
            ),
            pn.widgets.IntSlider.from_param(
                self.param.dimensions, name="Dimensions (d)", width=400
            ),
            pn.widgets.Select.from_param(self.param.device, name="Device", width=200),
            pn.widgets.Select.from_param(self.param.dtype, name="Data Type", width=200),
            title="⚙️ Swarm Configuration",
            collapsed=False,
        )

        # Langevin dynamics section
        self.langevin_section = pn.Card(
            pn.widgets.FloatSlider.from_param(self.param.gamma, name="Friction (γ)", width=400),
            pn.widgets.FloatSlider.from_param(
                self.param.beta, name="Inverse Temperature (β)", width=400
            ),
            pn.widgets.FloatSlider.from_param(self.param.delta_t, name="Timestep (Δt)", width=400),
            pn.widgets.Select.from_param(self.param.integrator, name="Integrator", width=200),
            title="🔥 Langevin Dynamics",
            collapsed=False,
        )

        # Cloning mechanism section
        self.cloning_section = pn.Card(
            pn.widgets.FloatSlider.from_param(
                self.param.sigma_x, name="Collision Radius (σ_x)", width=400
            ),
            pn.Row(
                pn.pane.Markdown("**Interaction Range (ε):**", width=200),
                pn.widgets.FloatInput.from_param(
                    self.param.epsilon, name="", width=150, step=0.01
                ),
                pn.pane.Markdown(
                    "_Spatial kernel scale for companion selection_",
                    styles={"font-size": "11px", "color": "#666"},
                    width=300,
                ),
            ),
            pn.widgets.FloatSlider.from_param(
                self.param.lambda_alg, name="Algorithmic Parameter (λ)", width=400
            ),
            pn.widgets.FloatSlider.from_param(
                self.param.alpha_restitution, name="Restitution Coefficient (α)", width=400
            ),
            pn.widgets.Checkbox.from_param(
                self.param.use_inelastic_collision, name="Use Inelastic Collisions"
            ),
            title="👯 Cloning Mechanism",
            collapsed=False,
        )

        # Benchmark selection section
        self.benchmark_section = pn.Card(
            pn.widgets.Select.from_param(
                self.param.benchmark_type, name="Benchmark Function", width=300
            ),
            pn.pane.Markdown(self._get_benchmark_info()),
            title="🎯 Optimization Benchmark",
            collapsed=False,
        )

        # Summary display
        self.summary_pane = pn.pane.Markdown(
            self._generate_summary(), styles={"background": "#e8f4f8", "padding": "15px"}
        )

        # Watch for parameter changes to update summary
        self.param.watch(self._update_summary, list(self.param))

    def _get_benchmark_info(self) -> str:
        """Get information about the selected benchmark."""
        benchmark_info = {
            "Sphere": "**Sphere**: Sum of squares. Global minimum at origin (0, ..., 0).",
            "Rastrigin": "**Rastrigin**: Highly multimodal function with many local minima. "
            "Global minimum at origin.",
            "StyblinskiTang": "**Styblinski-Tang**: Multimodal function. "
            "Global minimum at (-2.903534, ..., -2.903534).",
            "Rosenbrock": "**Rosenbrock**: Narrow valley benchmark. "
            "Global minimum at (1, ..., 1).",
            "EggHolder": "**EggHolder**: Complex landscape with many local minima. **Fixed 2D**.",
            "Easom": "**Easom**: Flat surface with sharp global minimum. **Fixed 2D**.",
            "HolderTable": "**HolderTable**: Multiple global minima. **Fixed 2D**.",
        }
        return benchmark_info.get(self.benchmark_type, "")

    def _update_summary(self, *events):  # noqa: ARG002
        """Update the summary display when parameters change."""
        self.summary_pane.object = self._generate_summary()
        if hasattr(self, "benchmark_section"):
            # Update benchmark info
            self.benchmark_section[1].object = self._get_benchmark_info()

    def _generate_summary(self) -> str:
        """Generate a summary of current parameter configuration."""
        return f"""
### 📋 Current Configuration

**Swarm**: {self.n_walkers} walkers in {self.dimensions}D space
({self.device}/{self.dtype})

**Langevin**: γ={self.gamma:.2f}, β={self.beta:.2f}, Δt={self.delta_t:.3f},
integrator={self.integrator}

**Cloning**: σ_x={self.sigma_x:.2f}, ε={self.epsilon:.2f}, λ={self.lambda_alg:.2f},
α={self.alpha_restitution:.2f}, inelastic={self.use_inelastic_collision}

**Benchmark**: {self.benchmark_type}
"""

    def _get_benchmark(self) -> OptimBenchmark:
        """Get the configured benchmark instance."""
        benchmark_classes = {
            "Sphere": Sphere,
            "Rastrigin": Rastrigin,
            "StyblinskiTang": StyblinskiTang,
            "Rosenbrock": Rosenbrock,
            "EggHolder": EggHolder,
            "Easom": Easom,
            "HolderTable": HolderTable,
        }

        benchmark_cls = benchmark_classes[self.benchmark_type]

        # Some benchmarks have fixed dimensions
        if self.benchmark_type in {"EggHolder", "Easom", "HolderTable"}:
            return benchmark_cls()
        return benchmark_cls(dims=self.dimensions)

    def get_params(self):
        """Get configured EuclideanGasParams object.

        Returns:
            EuclideanGasParams: Configured parameter object ready for EuclideanGas initialization.

        Note:
            This method requires importing from fragile.fractalai.core modules:
            >>> from fragile.fractalai.core.euclidean_gas import EuclideanGas
            >>> from fragile.fractalai.core.kinetic_operator import KineticOperator
            >>> from fragile.fractalai.core.cloning import CloneOperator

        Example:
            >>> selector = EuclideanGasParamSelector()
            >>> params = selector.get_params()
            >>> gas = EuclideanGas(params)

        """
        # Get benchmark
        self._get_benchmark()

        # NOTE: This method returns parameters but cannot construct
        # KineticOperator or CloneOperator directly because those require
        # additional objects (potential, device, dtype) that aren't part
        # of this selector's state.
        #
        # Users should construct EuclideanGas directly using the parameter
        # values from this selector. See the example in the class docstring.

        msg = (
            "The get_params() method has been removed because LangevinParams "
            "and other intermediate parameter classes no longer exist. "
            "Instead, directly construct EuclideanGas components:\n\n"
            "Example:\n"
            "  from fragile.fractalai.core.euclidean_gas import EuclideanGas\n"
            "  from fragile.fractalai.core.kinetic_operator import KineticOperator\n"
            "  from fragile.fractalai.core.cloning import CloneOperator\n"
            "  from fragile.fractalai.core.companion_selection import CompanionSelection\n"
            "  from fragile.fractalai.core.fitness import FitnessOperator\n\n"
            "  benchmark = selector.get_benchmark()\n"
            "  kinetic_op = KineticOperator(\n"
            "      gamma=selector.gamma,\n"
            "      beta=selector.beta,\n"
            "      delta_t=selector.delta_t,\n"
            "      integrator=selector.integrator,\n"
            "      potential=benchmark,\n"
            "      device=torch.device(selector.device),\n"
            "      dtype=getattr(torch, selector.dtype),\n"
            "  )\n"
            "  clone_op = CloneOperator(\n"
            "      sigma_x=selector.sigma_x,\n"
            "      alpha_restitution=selector.alpha_restitution,\n"
            "      ...\n"
            "  )\n"
            "  gas = EuclideanGas(...)\n"
        )
        raise NotImplementedError(msg)

    def get_benchmark(self) -> OptimBenchmark:
        """Get the configured benchmark function.

        Returns:
            OptimBenchmark: The selected benchmark function instance.

        """
        return self._get_benchmark()

    def panel(self):
        """Create and return the Panel dashboard.

        Returns:
            pn.Column: Panel column containing all dashboard components.

        Example:
            >>> selector = EuclideanGasParamSelector()
            >>> dashboard = selector.panel()
            >>> dashboard.servable()  # In a Panel app
            >>> # Or in a notebook:
            >>> dashboard  # Display inline

        """
        return pn.Column(
            self.header,
            self.swarm_section,
            self.langevin_section,
            self.cloning_section,
            self.benchmark_section,
            pn.pane.Markdown("---"),
            self.summary_pane,
            width=800,
        )

    def __panel__(self):
        """Support for direct Panel rendering."""
        return self.panel()


# Convenience function for quick usage
def create_param_selector(**kwargs) -> EuclideanGasParamSelector:
    """Create a parameter selector with custom defaults.

    Args:
        **kwargs: Parameter values to override defaults.

    Returns:
        EuclideanGasParamSelector: Configured parameter selector instance.

    Example:
        >>> selector = create_param_selector(n_walkers=100, gamma=2.0)
        >>> selector.panel()

    """
    return EuclideanGasParamSelector(**kwargs)
