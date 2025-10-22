"""Interactive parameter selector for Adaptive Gas swarms.

This module provides a Panel-based dashboard for interactively selecting
all parameters needed to configure an Adaptive Gas (Adaptive Viscous Fluid Model) swarm.
"""

import panel as pn
import param

from fragile.shaolin.euclidean_gas_params import EuclideanGasParamSelector


class AdaptiveGasParamSelector(param.Parameterized):
    """Interactive parameter selector for Adaptive Gas configuration.

    This class creates an interactive dashboard using Panel widgets that allows
    users to configure all parameters for an Adaptive Gas swarm, including both
    the Euclidean Gas backbone and the adaptive mechanisms.

    Example:
        >>> import panel as pn
        >>> from fragile.shaolin import AdaptiveGasParamSelector
        >>> pn.extension()
        >>> selector = AdaptiveGasParamSelector()
        >>> selector.panel()  # Display the dashboard
        >>> params = selector.get_params()  # Get configured parameters
        >>> gas = AdaptiveGas(params)  # Initialize gas with parameters

    """

    # Adaptive mechanism parameters
    epsilon_F = param.Number(
        default=0.1,
        bounds=(0.0, 1.0),
        step=0.05,
        doc="Adaptation rate for fitness-driven force (ε_F)",
    )
    nu = param.Number(
        default=0.05,
        bounds=(0.0, 1.0),
        step=0.05,
        doc="Viscosity coefficient for velocity coupling (ν)",
    )
    epsilon_Sigma = param.Number(
        default=2.0,
        bounds=(0.1, 10.0),
        step=0.1,
        doc="Diffusion regularization parameter (ε_Σ > H_max)",
    )
    A = param.Number(
        default=1.0,
        bounds=(0.1, 10.0),
        step=0.1,
        doc="Fitness potential amplitude (A)",
    )
    sigma_prime_min_patch = param.Number(
        default=0.1,
        bounds=(0.01, 1.0),
        step=0.01,
        doc="Z-score regularization (σ'_min,patch)",
    )
    patch_radius = param.Number(
        default=1.0,
        bounds=(0.1, 10.0),
        step=0.1,
        doc="Local patch radius for fitness computation (ρ)",
    )
    l_viscous = param.Number(
        default=0.5,
        bounds=(0.1, 5.0),
        step=0.1,
        doc="Viscous kernel length scale",
    )
    use_adaptive_diffusion = param.Boolean(
        default=True,
        doc="Enable adaptive Hessian-based diffusion tensor",
    )

    # Mode selection
    mode = param.Selector(
        default="adaptive",
        objects=["backbone", "adaptive", "adaptive_force_only", "viscous_only"],
        doc="Operating mode: backbone (no adaptations), adaptive (all), or selective",
    )

    def __init__(self, **params):
        """Initialize the parameter selector with default values."""
        super().__init__(**params)

        # Create Euclidean Gas selector for backbone parameters
        self.euclidean_selector = EuclideanGasParamSelector()

        # Create widgets
        self._create_widgets()

        # Watch for mode changes
        self.param.watch(self._update_mode, "mode")

    def _create_widgets(self):
        """Create Panel widgets for all parameters."""
        # Header
        self.header = pn.pane.Markdown(
            "## 🌊 Adaptive Gas Parameter Selector\n"
            "Configure the Adaptive Viscous Fluid Model with backbone and adaptive mechanisms.\n\n"
            "**Implements**: `clean_build/source/07_adaptative_gas.md`",
            styles={"background": "#e3f2fd", "padding": "10px", "border-radius": "5px"},
        )

        # Mode selection
        self.mode_section = pn.Card(
            pn.widgets.Select.from_param(self.param.mode, name="Operating Mode", width=300),
            pn.pane.Markdown(self._get_mode_info()),
            title="🎛️ Operating Mode",
            collapsed=False,
        )

        # Adaptive force parameters
        self.adaptive_force_section = pn.Card(
            pn.widgets.FloatSlider.from_param(
                self.param.epsilon_F, name="Adaptation Rate (ε_F)", width=400
            ),
            pn.pane.Markdown(
                "**Adaptive Force**: Fitness-driven guidance pushing walkers "
                "towards regions of higher fitness. Set to 0 for backbone mode."
            ),
            title="🎯 Adaptive Force (Fitness-Driven)",
            collapsed=False,
        )

        # Viscous force parameters
        self.viscous_force_section = pn.Card(
            pn.widgets.FloatSlider.from_param(
                self.param.nu, name="Viscosity Coefficient (ν)", width=400
            ),
            pn.widgets.FloatSlider.from_param(
                self.param.l_viscous, name="Kernel Length Scale", width=400
            ),
            pn.pane.Markdown(
                "**Viscous Force**: Navier-Stokes-inspired velocity coupling "
                "that smooths the velocity field and dissipates relative kinetic energy."
            ),
            title="💨 Viscous Force (Velocity Coupling)",
            collapsed=False,
        )

        # Adaptive diffusion parameters
        self.adaptive_diffusion_section = pn.Card(
            pn.widgets.Checkbox.from_param(
                self.param.use_adaptive_diffusion, name="Enable Adaptive Diffusion"
            ),
            pn.widgets.FloatSlider.from_param(
                self.param.epsilon_Sigma, name="Regularization (ε_Σ)", width=400
            ),
            pn.pane.Markdown(
                "**Adaptive Diffusion**: Anisotropic noise based on fitness "
                "landscape curvature. Explores flat directions (large noise) and "
                "exploits curved directions (small noise). Regularization ε_Σ "
                "ensures uniform ellipticity."
            ),
            title="🔀 Adaptive Diffusion (Hessian-Based)",
            collapsed=False,
        )

        # Fitness potential parameters
        self.fitness_section = pn.Card(
            pn.widgets.FloatSlider.from_param(
                self.param.A, name="Fitness Amplitude (A)", width=400
            ),
            pn.widgets.FloatSlider.from_param(
                self.param.sigma_prime_min_patch, name="Z-Score Regularization", width=400
            ),
            pn.widgets.FloatSlider.from_param(
                self.param.patch_radius, name="Patch Radius (ρ)", width=400
            ),
            pn.pane.Markdown(
                "**Fitness Potential**: Mean-field functional V_fit based on patched Z-scores. "
                "Uses local patch statistics for robust, non-local fitness evaluation."
            ),
            title="📊 Fitness Potential Configuration",
            collapsed=True,
        )

        # Summary display
        self.summary_pane = pn.pane.Markdown(
            self._generate_summary(), styles={"background": "#fff3e0", "padding": "15px"}
        )

        # Watch for parameter changes to update summary
        self.param.watch(self._update_summary, list(self.param))

    def _get_mode_info(self) -> str:
        """Get information about the selected mode."""
        mode_info = {
            "backbone": "**Backbone Mode**: Pure Euclidean Gas with no adaptive mechanisms. "
            "Provably stable baseline (ε_F=0, ν=0, adaptive diffusion off).",
            "adaptive": "**Full Adaptive Mode**: All three adaptive mechanisms enabled. "
            "Maximum adaptivity with stability via regularization.",
            "adaptive_force_only": "**Adaptive Force Only**: Only fitness-driven force enabled. "
            "Tests the effect of intelligent guidance without viscosity or adaptive noise.",
            "viscous_only": "**Viscous Force Only**: Only velocity coupling enabled. "
            "Tests fluid-like behavior without fitness guidance or adaptive diffusion.",
        }
        return mode_info.get(self.mode, "")

    def _update_mode(self, *events):  # noqa: ARG002
        """Update parameters when mode changes."""
        if self.mode == "backbone":
            self.epsilon_F = 0.0
            self.nu = 0.0
            self.use_adaptive_diffusion = False
        elif self.mode == "adaptive":
            self.epsilon_F = 0.1
            self.nu = 0.05
            self.use_adaptive_diffusion = True
        elif self.mode == "adaptive_force_only":
            self.epsilon_F = 0.1
            self.nu = 0.0
            self.use_adaptive_diffusion = False
        elif self.mode == "viscous_only":
            self.epsilon_F = 0.0
            self.nu = 0.05
            self.use_adaptive_diffusion = False

        # Update mode info
        if hasattr(self, "mode_section"):
            self.mode_section[1].object = self._get_mode_info()

    def _update_summary(self, *events):  # noqa: ARG002
        """Update the summary display when parameters change."""
        self.summary_pane.object = self._generate_summary()

    def _generate_summary(self) -> str:
        """Generate a summary of current parameter configuration."""
        # Get Euclidean Gas summary
        euclidean_params = self.euclidean_selector.get_params()

        return f"""
### 📋 Adaptive Gas Configuration

**Mode**: {self.mode}

**Backbone (Euclidean Gas)**:
- {euclidean_params.N} walkers in {euclidean_params.d}D space
- γ={euclidean_params.langevin.gamma:.2f}, β={euclidean_params.langevin.beta:.2f},
  Δt={euclidean_params.langevin.delta_t:.3f}
- Cloning: σ_x={euclidean_params.cloning.sigma_x:.2f},
  λ={euclidean_params.cloning.lambda_alg:.2f}

**Adaptive Mechanisms**:
- Adaptive Force: ε_F={self.epsilon_F:.2f}
  {"(disabled)" if self.epsilon_F == 0 else "(active)"}
- Viscous Force: ν={self.nu:.2f}, l={self.l_viscous:.2f}
  {"(disabled)" if self.nu == 0 else "(active)"}
- Adaptive Diffusion: {"enabled" if self.use_adaptive_diffusion else "disabled"}
  (ε_Σ={self.epsilon_Sigma:.2f})

**Fitness Potential**:
- Amplitude: A={self.A:.2f}
- Patch radius: ρ={self.patch_radius:.2f}
- Regularization: σ'_min={self.sigma_prime_min_patch:.2f}
"""

    def get_params(self):
        """Get configured AdaptiveGasParams object.

        Returns:
            AdaptiveGasParams: Configured parameter object ready for AdaptiveGas initialization.

        Example:
            >>> selector = AdaptiveGasParamSelector()
            >>> params = selector.get_params()
            >>> gas = AdaptiveGas(params)

        """
        # Import here to avoid circular dependencies
        from fragile.adaptive_gas import AdaptiveGasParams, AdaptiveParams  # noqa: PLC0415

        # Get Euclidean Gas parameters
        euclidean_params = self.euclidean_selector.get_params()

        # Create adaptive parameters
        adaptive_params = AdaptiveParams(
            epsilon_F=self.epsilon_F,
            nu=self.nu,
            epsilon_Sigma=self.epsilon_Sigma,
            A=self.A,
            sigma_prime_min_patch=self.sigma_prime_min_patch,
            patch_radius=self.patch_radius,
            l_viscous=self.l_viscous,
            use_adaptive_diffusion=self.use_adaptive_diffusion,
        )

        # Create complete adaptive gas parameters
        return AdaptiveGasParams(
            euclidean=euclidean_params,
            adaptive=adaptive_params,
            measurement_fn="potential",
        )

    def get_euclidean_params(self):
        """Get configured EuclideanGasParams for comparison.

        Returns:
            EuclideanGasParams: Backbone parameters without adaptive mechanisms.

        """
        return self.euclidean_selector.get_params()

    def panel(self):
        """Create and return the Panel dashboard.

        Returns:
            pn.Column: Panel column containing all dashboard components.

        Example:
            >>> selector = AdaptiveGasParamSelector()
            >>> dashboard = selector.panel()
            >>> dashboard.servable()  # In a Panel app
            >>> # Or in a notebook:
            >>> dashboard  # Display inline

        """
        return pn.Column(
            self.header,
            self.mode_section,
            pn.pane.Markdown("---"),
            pn.pane.Markdown("### Euclidean Gas Backbone Parameters"),
            self.euclidean_selector.panel(),
            pn.pane.Markdown("---"),
            pn.pane.Markdown("### Adaptive Mechanism Parameters"),
            self.adaptive_force_section,
            self.viscous_force_section,
            self.adaptive_diffusion_section,
            self.fitness_section,
            pn.pane.Markdown("---"),
            self.summary_pane,
            width=800,
        )

    def __panel__(self):
        """Support for direct Panel rendering."""
        return self.panel()


# Convenience function for quick usage
def create_adaptive_param_selector(**kwargs) -> AdaptiveGasParamSelector:
    """Create an adaptive parameter selector with custom defaults.

    Args:
        **kwargs: Parameter values to override defaults.

    Returns:
        AdaptiveGasParamSelector: Configured parameter selector instance.

    Example:
        >>> selector = create_adaptive_param_selector(epsilon_F=0.2, nu=0.1)
        >>> selector.panel()

    """
    return AdaptiveGasParamSelector(**kwargs)
