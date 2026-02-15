import panel as pn
import param
import torch
from torch import Tensor

from fragile.physics.fractal_gas.panel_model import INPUT_WIDTH, PanelModel


def logistic_rescale(z: Tensor, A: float = 1.0) -> Tensor:
    """Logistic rescale function mapping R -> [0, A].

    Implements g_A(z) = A / (1 + exp(-z)), a smooth, bounded, monotone increasing
    function used in the fitness potential V_fit[f, ρ](x) = g_A(Z_ρ[f, d, x]).

    Reference: Definition def-localized-mean-field-fitness in 11_geometric_gas.md

    Args:
        z: Input tensor (typically Z-scores)
        A: Upper bound of the output range (default: 1.0)

    Returns:
        Tensor with values in [0, A]
    """
    z_safe = torch.nan_to_num(z, nan=0.0, posinf=50.0, neginf=-50.0)
    z_safe = torch.clamp(z_safe, min=-50.0, max=50.0)
    return A * torch.sigmoid(z_safe)


def global_stats(values_tensor: Tensor, sigma_min: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Compute global mean and regularized std. All walkers are assumed alive."""
    mu = values_tensor.mean()
    sigma_reg = torch.sqrt(values_tensor.var() + sigma_min**2)
    return mu, sigma_reg

def patched_standardization(
    values: Tensor,
    sigma_min: float = 1e-8,
    detach_stats: bool = False,
    return_statistics: bool = False,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Compute Z-scores over all walkers (all walkers assumed alive).

    Z_i = (v_i - mu) / sigma', where mu and sigma are global statistics
    and sigma' = sqrt(sigma^2 + sigma_min^2) ensures numerical stability.

    Args:
        values: Tensor [N] of measurement values.
        sigma_min: Regularization constant ensuring sigma' >= sigma_min > 0.
        detach_stats: If True, detach mu and sigma from the computation graph.
        return_statistics: If True, return (z_scores, mu, sigma) tuple.

    Returns:
        If return_statistics=False: Z-scores tensor [N].
        If return_statistics=True: (z_scores [N], mu [scalar], sigma [scalar]).
    """
    mu, sigma_reg = global_stats(values, sigma_min=sigma_min)
    if detach_stats:
        mu = mu.detach()
        sigma_reg = sigma_reg.detach()

    # Compute Z-scores for all walkers
    z_scores = (values - mu) / sigma_reg

    if return_statistics:
        return z_scores, mu, sigma_reg
    return z_scores


def compute_fitness(
    positions: Tensor,
    rewards: Tensor,
    companions: Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    eta: float = 0.,
    sigma_min: float = 1e-8,
    A: float = 2.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute fitness potential for the Euclidean Gas. All walkers assumed alive.

    Pipeline:
    1. Compute distances to companions: d_i = ||x_i - x_{c(i)}||
    2. Standardize rewards and distances (global Z-scores)
    3. Logistic rescale: g_A(z) = A / (1 + exp(-z))
    4. Add positivity floor eta
    5. Combine: V_i = (d'_i)^beta * (r'_i)^alpha

    Args:
        positions: Walker positions [N, d].
        rewards: Raw reward values [N].
        companions: Companion indices [N] (pre-selected, not computed here).
        alpha: Reward channel exponent.
        beta: Diversity channel exponent.
        eta: Positivity floor added after rescale.
        sigma_min: Regularization for standardization.
        A: Upper bound for logistic rescale.

    Returns:
        fitness: Fitness potential [N].
        info: Dict with intermediate values (distances, z_rewards, z_distances,
            rescaled_rewards, rescaled_distances, mu/sigma for each channel).
    """
    # Step 1: Compute distances to companions
    companion_positions = positions[companions]
    pos_diff = positions - companion_positions
    distances = torch.sqrt((pos_diff**2).sum(dim=-1))

    # Step 2: Standardize both channels (global Z-scores over all walkers)
    z_rewards, mu_rewards, sigma_rewards = patched_standardization(
        values=rewards,
        sigma_min=sigma_min,
        return_statistics=True,
    )
    z_distances, mu_distances, sigma_distances = patched_standardization(
        values=distances,
        sigma_min=sigma_min,
        return_statistics=True,
    )
    # Step 5-6: Logistic rescale + positivity floor
    # r'_i = g_A(z_r,i) + η, d'_i = g_A(z_d,i) + η
    r_prime = logistic_rescale(z_rewards, A=A) + eta
    d_prime = logistic_rescale(z_distances, A=A) + eta

    # Step 7: Combine channels into fitness potential
    # V_i = (d'_i)^β · (r'_i)^α
    fitness = (d_prime**beta) * (r_prime**alpha)

    # All walkers are alive — no masking needed
    pos_squared_differences = (pos_diff**2).sum(dim=-1)
    info = {
        "distances": distances,
        "companions": companions,
        "z_rewards": z_rewards,
        "z_distances": z_distances,
        "pos_squared_differences": pos_squared_differences,
        "vel_squared_differences": torch.zeros_like(pos_squared_differences),
        "rescaled_rewards": r_prime,
        "rescaled_distances": d_prime,
        "mu_rewards": mu_rewards,
        "sigma_rewards": sigma_rewards,
        "mu_distances": mu_distances,
        "sigma_distances": sigma_distances,
    }
    return fitness, info


class FitnessOperator(PanelModel):
    """Fitness operator for the Euclidean Gas. All walkers assumed alive.

    Wraps ``compute_fitness`` with configurable parameters exposed as Panel widgets.
    Companion selection is handled externally by EuclideanGas; this operator only
    computes fitness given pre-selected companions.

    Parameters:
        alpha: Reward channel exponent.
        beta: Diversity channel exponent.
        eta: Positivity floor added after logistic rescale.
        sigma_min: Regularization for standardization.
        A: Upper bound for logistic rescale.
    """

    _n_widget_columns = param.Integer(default=2, bounds=(1, None), doc="Number of widget columns")
    _max_widget_width = param.Integer(default=800, bounds=(0, None), doc="Maximum widget width")

    # Fitness parameters
    alpha = param.Number(
        default=1.0,
        bounds=(0, None),
        softbounds=(0.01, 5.0),
        inclusive_bounds=(False, True),
        doc="Reward channel exponent (α)",
    )
    beta = param.Number(
        default=1.0,
        bounds=(0, None),
        softbounds=(0.01, 5.0),
        inclusive_bounds=(False, True),
        doc="Diversity channel exponent (β)",
    )
    eta = param.Number(
        default=0.0,
        bounds=(0, None),
        softbounds=(0.0, 0.5),
        inclusive_bounds=(True, True),
        doc="Positivity floor parameter (η)",
    )
    sigma_min = param.Number(
        default=0.,
        bounds=(0, None),
        softbounds=(0.0, 1e-3),
        inclusive_bounds=(True, True),
        doc="Regularization for patched standardization (σ_min)",
    )
    A = param.Number(
        default=2.0,
        bounds=(0, None),
        softbounds=(1.0, 5.0),
        inclusive_bounds=(False, True),
        doc="Upper bound for logistic rescale",
    )

    @property
    def widgets(self) -> dict[str, dict]:
        """Widget configurations for fitness parameters."""
        return {
            "alpha": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "α (reward exponent)",
                "start": 0.1,
                "end": 5.0,
                "step": 0.05,
            },
            "beta": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "β (diversity exponent)",
                "start": 0.5,
                "end": 5.0,
                "step": 0.1,
            },
            "eta": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "η (positivity floor)",
                "start": 0.0,
                "end": 0.1,
                "step": 0.001,
            },
            "sigma_min": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "σ_min (standardization reg)",
                "start": 1e-9,
                "end": 1e-3,
                "step": 1e-9,
                "format": "%.1e",
            },
            "A": {
                "type": pn.widgets.EditableFloatSlider,
                "width": INPUT_WIDTH,
                "name": "A (rescale bound)",
                "start": 1.0,
                "end": 5.0,
                "step": 0.1,
            },
        }

    @property
    def widget_parameters(self) -> list[str]:
        """Parameters to display in UI."""
        return ["alpha", "beta", "eta", "sigma_min", "A"]

    def __call__(
        self,
        positions: Tensor,
        rewards: Tensor,
        companions: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute fitness potential. All walkers assumed alive.

        Args:
            positions: Walker positions [N, d].
            rewards: Raw reward values [N].
            companions: Companion indices [N] (pre-selected by EuclideanGas).

        Returns:
            fitness: Fitness potential [N].
            info: Dict with intermediate values for diagnostics.
        """
        with torch.no_grad():
            return compute_fitness(
                positions=positions,
                rewards=rewards,
                companions=companions,
                alpha=self.alpha,
                beta=self.beta,
                eta=self.eta,
                sigma_min=self.sigma_min,
                A=self.A,
            )
