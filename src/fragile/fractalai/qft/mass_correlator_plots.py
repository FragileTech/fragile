"""
Mass correlator plotting module for lattice QFT channel analysis.

Computes and visualizes two-point correlators across standard lattice QFT channels
using HoloViews with Bokeh backend.

Standard Channels (Lattice QFT):
    - scalar (σ): Identity/trace - mass from C(t) ~ e^{-mσ·t}
    - pseudoscalar (π, pion): γ₅ projection - lightest meson
    - vector (ρ, rho): γ_μ projection - J=1 meson
    - axial_vector (a₁): γ₅γ_μ projection - J=1⁺ meson
    - tensor: σ_μν projection - J=2 states
    - nucleon: Baryon channel - 3-quark correlator

Usage:
    from fragile.fractalai.qft.mass_correlator_plots import (
        MassCorrelatorComputer,
        MassCorrelatorPlotter,
        compute_all_channel_correlators,
        build_mass_correlator_dashboard,
    )

    # From RunHistory
    computer = MassCorrelatorComputer(history)
    correlators = computer.compute_all_channels()
    plotter = MassCorrelatorPlotter(correlators)
    dashboard = plotter.build_dashboard()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import holoviews as hv
import numpy as np
import torch
from torch import Tensor

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.qft.particle_observables import (
    compute_color_state,
    compute_companion_distance,
    compute_effective_mass,
    compute_knn_indices,
    compute_meson_operator_knn,
    compute_baryon_operator_knn,
    compute_time_correlator,
    fit_mass_exponential,
)


hv.extension("bokeh")


# =============================================================================
# Channel Definitions (Standard Lattice QFT)
# =============================================================================

@dataclass
class ChannelDefinition:
    """Definition of a particle channel for correlator analysis."""

    name: str
    display_name: str
    quantum_numbers: dict[str, Any]
    color: str  # Bokeh color for plotting
    description: str
    operator_type: str  # 'bilinear', 'trilinear', 'gauge'


# Standard lattice QFT channels with their quantum numbers
STANDARD_CHANNELS: dict[str, ChannelDefinition] = {
    "scalar": ChannelDefinition(
        name="scalar",
        display_name="Scalar (σ)",
        quantum_numbers={"J": 0, "P": "+", "C": "+"},
        color="#1f77b4",  # Blue
        description="Scalar meson channel (σ/f₀)",
        operator_type="bilinear",
    ),
    "pseudoscalar": ChannelDefinition(
        name="pseudoscalar",
        display_name="Pseudoscalar (π)",
        quantum_numbers={"J": 0, "P": "-", "C": "+"},
        color="#ff7f0e",  # Orange
        description="Pseudoscalar meson channel (pion)",
        operator_type="bilinear",
    ),
    "vector": ChannelDefinition(
        name="vector",
        display_name="Vector (ρ)",
        quantum_numbers={"J": 1, "P": "-", "C": "-"},
        color="#2ca02c",  # Green
        description="Vector meson channel (rho)",
        operator_type="bilinear",
    ),
    "axial_vector": ChannelDefinition(
        name="axial_vector",
        display_name="Axial Vector (a₁)",
        quantum_numbers={"J": 1, "P": "+", "C": "-"},
        color="#d62728",  # Red
        description="Axial vector meson channel (a₁)",
        operator_type="bilinear",
    ),
    "tensor": ChannelDefinition(
        name="tensor",
        display_name="Tensor (f₂)",
        quantum_numbers={"J": 2, "P": "+", "C": "+"},
        color="#9467bd",  # Purple
        description="Tensor meson channel",
        operator_type="bilinear",
    ),
    "nucleon": ChannelDefinition(
        name="nucleon",
        display_name="Nucleon (N)",
        quantum_numbers={"J": 0.5, "P": "+", "B": 1},
        color="#8c564b",  # Brown
        description="Baryon channel (proton/neutron)",
        operator_type="trilinear",
    ),
    "glueball": ChannelDefinition(
        name="glueball",
        display_name="Glueball (0⁺⁺)",
        quantum_numbers={"J": 0, "P": "+", "C": "+", "glue": True},
        color="#e377c2",  # Pink
        description="Scalar glueball channel",
        operator_type="gauge",
    ),
}


@dataclass
class ChannelCorrelatorResult:
    """Result of computing correlator for a single channel."""

    channel: ChannelDefinition
    lags: np.ndarray  # Time separations (lattice units)
    lag_times: np.ndarray  # Time separations (physical units)
    correlator: np.ndarray  # C(t) values
    correlator_err: np.ndarray | None  # Jackknife/bootstrap errors
    effective_mass: np.ndarray  # m_eff(t) = -d/dt log C(t)
    mass_fit: dict[str, float]  # Fitted mass parameters
    n_samples: int  # Number of time samples used
    series: np.ndarray | None = None  # Raw operator time series


@dataclass
class MassCorrelatorConfig:
    """Configuration for mass correlator computation."""

    # Time parameters
    warmup_fraction: float = 0.1
    max_lag: int | None = 80
    dt: float | None = None  # Override delta_t if needed

    # Color state parameters
    h_eff: float = 1.0
    mass: float = 1.0
    ell0: float | None = None

    # Neighbor selection
    neighbor_method: str = "knn"  # 'companion', 'knn', 'voronoi'
    knn_k: int = 4
    knn_sample: int | None = 512

    # Fitting parameters
    use_connected: bool = True
    fit_start: int = 5
    fit_stop: int = 20

    # Channels to compute
    channels: tuple[str, ...] = (
        "scalar",
        "pseudoscalar",
        "vector",
        "nucleon",
    )


# =============================================================================
# Channel Projector Operators
# =============================================================================


class ChannelProjector:
    """
    Projects walker states onto particle channel operators.

    Implements standard lattice QFT gamma matrix structure for
    different meson/baryon channels.
    """

    def __init__(self, dim: int, device: torch.device | str = "cpu"):
        """
        Initialize projector for given dimension.

        Args:
            dim: Spatial dimension (typically 3).
            device: Compute device.
        """
        self.dim = dim
        self.device = torch.device(device)
        self._build_gamma_matrices()

    def _build_gamma_matrices(self) -> None:
        """Build gamma matrices for channel projections."""
        d = self.dim
        device = self.device
        dtype = torch.complex128

        self.gamma: dict[str, Tensor] = {}

        # Identity (scalar channel)
        self.gamma["1"] = torch.eye(d, device=device, dtype=dtype)

        # γ₅ (pseudoscalar) - alternating signs for d-dimensional generalization
        gamma5_diag = torch.tensor(
            [(-1.0) ** i for i in range(d)],
            device=device,
            dtype=dtype,
        )
        self.gamma["5"] = torch.diag(gamma5_diag)

        # γ_μ (vector) - direction projections
        for mu in range(d):
            gamma_mu = torch.zeros(d, d, device=device, dtype=dtype)
            gamma_mu[mu, mu] = 1.0
            # Off-diagonal terms for proper vector transformation
            if mu > 0:
                gamma_mu[mu, 0] = 0.5j
                gamma_mu[0, mu] = -0.5j
            self.gamma[f"mu{mu}"] = gamma_mu

        # γ₅γ_μ (axial vector)
        for mu in range(d):
            self.gamma[f"5mu{mu}"] = self.gamma["5"] @ self.gamma[f"mu{mu}"]

        # σ_μν (tensor) - antisymmetric combinations
        for mu in range(d):
            for nu in range(mu + 1, d):
                sigma = torch.zeros(d, d, device=device, dtype=dtype)
                sigma[mu, nu] = 1.0j
                sigma[nu, mu] = -1.0j
                self.gamma[f"sig{mu}{nu}"] = sigma

    def project_bilinear(
        self,
        color_i: Tensor,
        color_j: Tensor,
        channel: str,
    ) -> Tensor:
        """
        Compute bilinear operator ψ̄_i Γ ψ_j for given channel.

        Args:
            color_i: Color states for site i [N, d].
            color_j: Color states for site j [N, d].
            channel: Channel name ('scalar', 'pseudoscalar', etc.).

        Returns:
            Operator values [N].
        """
        if channel == "scalar":
            gamma = self.gamma["1"]
        elif channel == "pseudoscalar":
            gamma = self.gamma["5"]
        elif channel == "vector":
            # Sum over spatial directions
            result = torch.zeros(
                color_i.shape[0],
                device=color_i.device,
                dtype=color_i.dtype,
            )
            for mu in range(self.dim):
                gamma_mu = self.gamma[f"mu{mu}"].to(color_i.device, dtype=color_i.dtype)
                result += torch.einsum(
                    "ni,ij,nj->n",
                    color_i.conj(),
                    gamma_mu,
                    color_j,
                )
            return result / self.dim
        elif channel == "axial_vector":
            result = torch.zeros(
                color_i.shape[0],
                device=color_i.device,
                dtype=color_i.dtype,
            )
            for mu in range(self.dim):
                gamma_5mu = self.gamma[f"5mu{mu}"].to(color_i.device, dtype=color_i.dtype)
                result += torch.einsum(
                    "ni,ij,nj->n",
                    color_i.conj(),
                    gamma_5mu,
                    color_j,
                )
            return result / self.dim
        elif channel == "tensor":
            # Sum over antisymmetric pairs
            result = torch.zeros(
                color_i.shape[0],
                device=color_i.device,
                dtype=color_i.dtype,
            )
            count = 0
            for mu in range(self.dim):
                for nu in range(mu + 1, self.dim):
                    sigma = self.gamma[f"sig{mu}{nu}"].to(color_i.device, dtype=color_i.dtype)
                    result += torch.einsum(
                        "ni,ij,nj->n",
                        color_i.conj(),
                        sigma,
                        color_j,
                    )
                    count += 1
            return result / max(count, 1)
        else:
            gamma = self.gamma.get("1", torch.eye(self.dim, device=color_i.device))

        gamma = gamma.to(color_i.device, dtype=color_i.dtype)
        return torch.einsum("ni,ij,nj->n", color_i.conj(), gamma, color_j)


# =============================================================================
# Mass Correlator Computer
# =============================================================================


class MassCorrelatorComputer:
    """
    Computes mass correlators from RunHistory for all channels.

    This class extracts particle operator time series from Fractal Gas
    run data and computes two-point correlators for mass extraction.
    """

    def __init__(
        self,
        history: RunHistory,
        config: MassCorrelatorConfig | None = None,
    ):
        """
        Initialize correlator computer.

        Args:
            history: Fractal Gas run history.
            config: Configuration parameters.
        """
        self.history = history
        self.config = config or MassCorrelatorConfig()
        self.projector = ChannelProjector(history.d, history.x_final.device)
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration against history."""
        if self.config.ell0 is None:
            # Estimate from companion distances
            self._estimate_ell0()

        if self.config.dt is None:
            self.config.dt = float(self.history.delta_t * self.history.record_every)

    def _estimate_ell0(self) -> None:
        """Estimate ell0 from median companion distance."""
        mid_idx = self.history.n_recorded // 2
        if mid_idx == 0:
            self.config.ell0 = 1.0
            return

        x_pre = self.history.x_before_clone[mid_idx]
        comp_dist = self.history.companions_distance[mid_idx - 1]
        alive = self.history.alive_mask[mid_idx - 1]

        dist = compute_companion_distance(
            x_pre, comp_dist, self.history.pbc, self.history.bounds
        )
        if dist.numel() > 0 and alive.any():
            self.config.ell0 = float(dist[alive].median().item())
        else:
            self.config.ell0 = 1.0

    def _compute_color_states(self, t_idx: int) -> tuple[Tensor, Tensor]:
        """
        Compute color states at given time index.

        Args:
            t_idx: Time index into history.

        Returns:
            (color, valid) tensors.
        """
        info_idx = t_idx - 1
        v_pre = self.history.v_before_clone[t_idx]
        force_visc = self.history.force_viscous[info_idx]

        color, valid = compute_color_state(
            force_visc,
            v_pre,
            self.config.h_eff,
            self.config.mass,
            self.config.ell0,
        )
        return color, valid

    def _get_neighbors(
        self,
        t_idx: int,
        sample_indices: Tensor,
    ) -> Tensor | None:
        """
        Get neighbor indices for operator computation.

        Args:
            t_idx: Time index.
            sample_indices: Walker indices to compute neighbors for.

        Returns:
            Neighbor indices [n_samples, k] or None.
        """
        info_idx = t_idx - 1
        alive = self.history.alive_mask[info_idx]
        x_pre = self.history.x_before_clone[t_idx]

        if self.config.neighbor_method == "knn":
            try:
                return compute_knn_indices(
                    x_pre,
                    alive,
                    self.config.knn_k,
                    self.history.pbc,
                    self.history.bounds,
                    sample_indices=sample_indices,
                )
            except ValueError:
                return None
        else:
            # Use companion indices
            return self.history.companions_distance[info_idx].unsqueeze(1)

    def compute_channel_series(
        self,
        channel: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute operator time series for given channel.

        Args:
            channel: Channel name.

        Returns:
            (time_index, time_tau, series) arrays.
        """
        channel_def = STANDARD_CHANNELS.get(channel)
        if channel_def is None:
            raise ValueError(f"Unknown channel: {channel}")

        start_idx = max(1, int(self.history.n_recorded * self.config.warmup_fraction))
        time_index = []
        time_tau = []
        series = []

        for t_idx in range(start_idx, self.history.n_recorded):
            info_idx = t_idx - 1
            alive = self.history.alive_mask[info_idx]

            if not alive.any():
                continue

            # Get color states
            color, color_valid = self._compute_color_states(t_idx)

            # Sample indices
            sample_indices = torch.where(alive)[0]
            if self.config.knn_sample is not None:
                sample_indices = sample_indices[: self.config.knn_sample]

            if sample_indices.numel() == 0:
                continue

            # Compute operator based on channel type
            if channel_def.operator_type == "trilinear":
                # Nucleon/baryon operator
                if self.history.d != 3:
                    continue  # Baryon requires d=3

                neighbors = self._get_neighbors(t_idx, sample_indices)
                if neighbors is None or neighbors.shape[1] < 2:
                    continue

                try:
                    op_values, valid = compute_baryon_operator_knn(
                        color,
                        sample_indices,
                        neighbors,
                        alive,
                        color_valid,
                        max_pairs=100,
                    )
                except ValueError:
                    continue
            else:
                # Bilinear meson operator
                neighbors = self._get_neighbors(t_idx, sample_indices)
                if neighbors is None or neighbors.shape[1] < 1:
                    continue

                # Compute channel-projected bilinear
                color_i = color[sample_indices]
                color_j = color[neighbors[:, 0]]
                valid_i = alive[sample_indices] & color_valid[sample_indices]
                valid_j = alive[neighbors[:, 0]] & color_valid[neighbors[:, 0]]
                valid = valid_i & valid_j

                op_values = self.projector.project_bilinear(
                    color_i, color_j, channel
                )

            # Average over valid samples
            if valid.any():
                op_mean = op_values[valid].mean().item()
            else:
                op_mean = 0.0 + 0.0j

            time_index.append(t_idx)
            time_tau.append(self.history.recorded_steps[t_idx] * self.history.delta_t)
            series.append(op_mean)

        return (
            np.array(time_index, dtype=np.int64),
            np.array(time_tau, dtype=np.float64),
            np.array(series, dtype=np.complex128),
        )

    def compute_channel_correlator(
        self,
        channel: str,
    ) -> ChannelCorrelatorResult:
        """
        Compute full correlator analysis for a channel.

        Args:
            channel: Channel name.

        Returns:
            ChannelCorrelatorResult with all computed quantities.
        """
        channel_def = STANDARD_CHANNELS.get(channel)
        if channel_def is None:
            raise ValueError(f"Unknown channel: {channel}")

        # Get operator series
        time_index, time_tau, series = self.compute_channel_series(channel)

        if series.size == 0:
            return ChannelCorrelatorResult(
                channel=channel_def,
                lags=np.array([]),
                lag_times=np.array([]),
                correlator=np.array([]),
                correlator_err=None,
                effective_mass=np.array([]),
                mass_fit={"mass": 0.0, "amplitude": 0.0, "r_squared": 0.0},
                n_samples=0,
                series=series,
            )

        # Compute time correlator
        lags, corr = compute_time_correlator(
            series,
            max_lag=self.config.max_lag,
            use_connected=self.config.use_connected,
        )

        # Convert lags to physical time
        lag_times = lags.astype(np.float64) * self.config.dt

        # Compute effective mass
        eff_mass = compute_effective_mass(corr, self.config.dt)

        # Fit mass from correlator
        mass_fit = fit_mass_exponential(
            lag_times,
            corr,
            fit_start=self.config.fit_start,
            fit_stop=self.config.fit_stop,
        )

        return ChannelCorrelatorResult(
            channel=channel_def,
            lags=lags,
            lag_times=lag_times,
            correlator=np.real(corr),
            correlator_err=None,  # Could add jackknife here
            effective_mass=eff_mass,
            mass_fit=mass_fit,
            n_samples=series.size,
            series=series,
        )

    def compute_all_channels(self) -> dict[str, ChannelCorrelatorResult]:
        """
        Compute correlators for all configured channels.

        Returns:
            Dictionary mapping channel names to results.
        """
        results = {}
        for channel in self.config.channels:
            try:
                results[channel] = self.compute_channel_correlator(channel)
            except Exception as e:
                # Create empty result on error
                channel_def = STANDARD_CHANNELS.get(
                    channel,
                    ChannelDefinition(
                        name=channel,
                        display_name=channel,
                        quantum_numbers={},
                        color="#666666",
                        description=f"Error: {e}",
                        operator_type="unknown",
                    ),
                )
                results[channel] = ChannelCorrelatorResult(
                    channel=channel_def,
                    lags=np.array([]),
                    lag_times=np.array([]),
                    correlator=np.array([]),
                    correlator_err=None,
                    effective_mass=np.array([]),
                    mass_fit={"mass": 0.0, "error": str(e)},
                    n_samples=0,
                )
        return results


# =============================================================================
# HoloViews Plotting
# =============================================================================


class MassCorrelatorPlotter:
    """
    Creates HoloViews plots for mass correlator analysis.

    Uses Bokeh backend for interactive visualization.
    """

    def __init__(self, results: dict[str, ChannelCorrelatorResult]):
        """
        Initialize plotter with computed results.

        Args:
            results: Dictionary of channel correlator results.
        """
        self.results = results

    def build_correlator_plot(
        self,
        channel: str,
        logy: bool = True,
        width: int = 500,
        height: int = 350,
    ) -> hv.Overlay | None:
        """
        Build correlator C(t) plot for a single channel.

        Args:
            channel: Channel name.
            logy: Use log scale for y-axis.
            width: Plot width.
            height: Plot height.

        Returns:
            HoloViews Overlay or None if no data.
        """
        result = self.results.get(channel)
        if result is None or result.correlator.size == 0:
            return None

        # Filter positive values for log plot
        mask = result.correlator > 0
        if not mask.any():
            return None

        t = result.lag_times[mask]
        c = result.correlator[mask]
        color = result.channel.color

        # Data points
        scatter = hv.Scatter(
            (t, c),
            kdims=["t"],
            vdims=["C(t)"],
        ).opts(
            color=color,
            size=6,
            alpha=0.8,
        )

        elements = [scatter]

        # Add fit curve if available
        fit = result.mass_fit
        if fit.get("mass", 0) > 0 and fit.get("amplitude", 0) > 0:
            t_fit = np.linspace(t.min(), t.max(), 100)
            c_fit = fit["amplitude"] * np.exp(-fit["mass"] * t_fit)
            curve = hv.Curve(
                (t_fit, c_fit),
                kdims=["t"],
                vdims=["C(t)"],
            ).opts(
                color=color,
                line_dash="dashed",
                line_width=2,
            )
            elements.append(curve)

        overlay = hv.Overlay(elements).opts(
            xlabel="t",
            ylabel="C(t)",
            title=f"{result.channel.display_name} Correlator",
            logy=logy,
            width=width,
            height=height,
            legend_position="top_right",
        )

        return overlay

    def build_effective_mass_plot(
        self,
        channel: str,
        width: int = 500,
        height: int = 350,
    ) -> hv.Overlay | None:
        """
        Build effective mass m_eff(t) plot for a single channel.

        Args:
            channel: Channel name.
            width: Plot width.
            height: Plot height.

        Returns:
            HoloViews Overlay or None if no data.
        """
        result = self.results.get(channel)
        if result is None or result.effective_mass.size == 0:
            return None

        # Filter finite values
        mask = np.isfinite(result.effective_mass) & (result.effective_mass > 0)
        if not mask.any():
            return None

        # Effective mass is computed between points, so use midpoint times
        t_mid = (result.lag_times[:-1] + result.lag_times[1:]) / 2
        t = t_mid[mask]
        m_eff = result.effective_mass[mask]
        color = result.channel.color

        # Data points with error bars (if available)
        scatter = hv.Scatter(
            (t, m_eff),
            kdims=["t"],
            vdims=["m_eff"],
        ).opts(
            color=color,
            size=6,
            alpha=0.8,
        )

        elements = [scatter]

        # Add fitted mass as horizontal line
        fit = result.mass_fit
        if fit.get("mass", 0) > 0:
            mass_line = hv.HLine(fit["mass"]).opts(
                color=color,
                line_dash="dashed",
                line_width=2,
            )
            elements.append(mass_line)

            # Add mass label
            label = hv.Text(
                t.max() * 0.7,
                fit["mass"] * 1.1,
                f'm = {fit["mass"]:.4f}',
            ).opts(
                color=color,
                text_font_size="10pt",
            )
            elements.append(label)

        overlay = hv.Overlay(elements).opts(
            xlabel="t",
            ylabel="m_eff(t)",
            title=f"{result.channel.display_name} Effective Mass",
            width=width,
            height=height,
        )

        return overlay

    def build_all_correlators_overlay(
        self,
        logy: bool = True,
        width: int = 700,
        height: int = 450,
    ) -> hv.Overlay | None:
        """
        Build overlay of all channel correlators on one plot.

        Args:
            logy: Use log scale.
            width: Plot width.
            height: Plot height.

        Returns:
            HoloViews Overlay.
        """
        curves = []

        for channel, result in self.results.items():
            if result.correlator.size == 0:
                continue

            mask = result.correlator > 0
            if not mask.any():
                continue

            t = result.lag_times[mask]
            c = result.correlator[mask]

            curve = hv.Curve(
                (t, c),
                kdims=["t"],
                vdims=["C(t)"],
                label=result.channel.display_name,
            ).opts(
                color=result.channel.color,
                line_width=2,
            )
            curves.append(curve)

        if not curves:
            return None

        return hv.Overlay(curves).opts(
            xlabel="t",
            ylabel="C(t)",
            title="Mass Correlators by Channel",
            logy=logy,
            width=width,
            height=height,
            legend_position="top_right",
            show_legend=True,
        )

    def build_all_effective_masses_overlay(
        self,
        width: int = 700,
        height: int = 450,
    ) -> hv.Overlay | None:
        """
        Build overlay of all channel effective masses on one plot.

        Args:
            width: Plot width.
            height: Plot height.

        Returns:
            HoloViews Overlay.
        """
        curves = []

        for channel, result in self.results.items():
            if result.effective_mass.size == 0:
                continue

            mask = np.isfinite(result.effective_mass) & (result.effective_mass > 0)
            if not mask.any():
                continue

            t_mid = (result.lag_times[:-1] + result.lag_times[1:]) / 2
            t = t_mid[mask]
            m_eff = result.effective_mass[mask]

            curve = hv.Curve(
                (t, m_eff),
                kdims=["t"],
                vdims=["m_eff"],
                label=result.channel.display_name,
            ).opts(
                color=result.channel.color,
                line_width=2,
            )
            curves.append(curve)

            # Add horizontal line for fitted mass
            fit = result.mass_fit
            if fit.get("mass", 0) > 0:
                hline = hv.HLine(fit["mass"]).opts(
                    color=result.channel.color,
                    line_dash="dashed",
                    alpha=0.6,
                )
                curves.append(hline)

        if not curves:
            return None

        return hv.Overlay(curves).opts(
            xlabel="t",
            ylabel="m_eff(t)",
            title="Effective Masses by Channel",
            width=width,
            height=height,
            legend_position="top_right",
            show_legend=True,
        )

    def build_mass_spectrum_bar(
        self,
        width: int = 500,
        height: int = 400,
    ) -> hv.Bars | None:
        """
        Build bar chart of extracted masses.

        Args:
            width: Plot width.
            height: Plot height.

        Returns:
            HoloViews Bars.
        """
        data = []
        colors = []

        for channel, result in self.results.items():
            mass = result.mass_fit.get("mass", 0)
            if mass > 0:
                data.append((result.channel.display_name, mass))
                colors.append(result.channel.color)

        if not data:
            return None

        bars = hv.Bars(
            data,
            kdims=["Channel"],
            vdims=["Mass"],
        ).opts(
            color=hv.Cycle(colors),
            xlabel="Channel",
            ylabel="Mass (lattice units)",
            title="Extracted Mass Spectrum",
            width=width,
            height=height,
            xrotation=45,
        )

        return bars

    def build_dashboard(
        self,
        width: int = 500,
        height: int = 350,
    ) -> hv.Layout:
        """
        Build full dashboard with all plots.

        Args:
            width: Individual plot width.
            height: Individual plot height.

        Returns:
            HoloViews Layout.
        """
        plots = []

        # Combined overlays
        all_corr = self.build_all_correlators_overlay(width=700, height=450)
        if all_corr is not None:
            plots.append(all_corr)

        all_meff = self.build_all_effective_masses_overlay(width=700, height=450)
        if all_meff is not None:
            plots.append(all_meff)

        # Mass spectrum
        spectrum = self.build_mass_spectrum_bar(width=500, height=400)
        if spectrum is not None:
            plots.append(spectrum)

        # Individual channel plots
        for channel in self.results:
            corr_plot = self.build_correlator_plot(channel, width=width, height=height)
            if corr_plot is not None:
                plots.append(corr_plot)

            meff_plot = self.build_effective_mass_plot(channel, width=width, height=height)
            if meff_plot is not None:
                plots.append(meff_plot)

        if not plots:
            # Return empty layout
            return hv.Text(0.5, 0.5, "No data available").opts(
                width=400, height=200
            )

        return hv.Layout(plots).cols(2)

    def save_dashboard(
        self,
        output_path: Path | str,
        backend: str = "bokeh",
    ) -> Path:
        """
        Save dashboard to HTML file.

        Args:
            output_path: Output file path.
            backend: HoloViews backend.

        Returns:
            Path to saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() != ".html":
            output_path = output_path.with_suffix(".html")

        dashboard = self.build_dashboard()
        hv.save(dashboard, str(output_path), backend=backend)

        return output_path


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_all_channel_correlators(
    history: RunHistory,
    channels: tuple[str, ...] | None = None,
    config: MassCorrelatorConfig | None = None,
) -> dict[str, ChannelCorrelatorResult]:
    """
    Convenience function to compute correlators for all channels.

    Args:
        history: Fractal Gas run history.
        channels: Channels to compute (default: standard set).
        config: Configuration (will be updated with channels if provided).

    Returns:
        Dictionary of channel results.
    """
    if config is None:
        config = MassCorrelatorConfig()

    if channels is not None:
        config.channels = channels

    computer = MassCorrelatorComputer(history, config)
    return computer.compute_all_channels()


def build_mass_correlator_dashboard(
    history: RunHistory,
    channels: tuple[str, ...] | None = None,
    config: MassCorrelatorConfig | None = None,
) -> hv.Layout:
    """
    Convenience function to build full mass correlator dashboard.

    Args:
        history: Fractal Gas run history.
        channels: Channels to compute.
        config: Configuration.

    Returns:
        HoloViews Layout.
    """
    results = compute_all_channel_correlators(history, channels, config)
    plotter = MassCorrelatorPlotter(results)
    return plotter.build_dashboard()


def save_mass_correlator_plots(
    history: RunHistory,
    output_dir: Path | str,
    channels: tuple[str, ...] | None = None,
    config: MassCorrelatorConfig | None = None,
) -> dict[str, Path]:
    """
    Compute correlators and save all plots to files.

    Args:
        history: Fractal Gas run history.
        output_dir: Output directory.
        channels: Channels to compute.
        config: Configuration.

    Returns:
        Dictionary mapping plot names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = compute_all_channel_correlators(history, channels, config)
    plotter = MassCorrelatorPlotter(results)

    saved_paths = {}

    # Save dashboard
    dashboard_path = plotter.save_dashboard(output_dir / "mass_correlator_dashboard.html")
    saved_paths["dashboard"] = dashboard_path

    # Save individual plots
    for channel in results:
        corr_plot = plotter.build_correlator_plot(channel)
        if corr_plot is not None:
            path = output_dir / f"{channel}_correlator.html"
            hv.save(corr_plot, str(path), backend="bokeh")
            saved_paths[f"{channel}_correlator"] = path

        meff_plot = plotter.build_effective_mass_plot(channel)
        if meff_plot is not None:
            path = output_dir / f"{channel}_effective_mass.html"
            hv.save(meff_plot, str(path), backend="bokeh")
            saved_paths[f"{channel}_effective_mass"] = path

    # Save combined plots
    all_corr = plotter.build_all_correlators_overlay()
    if all_corr is not None:
        path = output_dir / "all_correlators.html"
        hv.save(all_corr, str(path), backend="bokeh")
        saved_paths["all_correlators"] = path

    all_meff = plotter.build_all_effective_masses_overlay()
    if all_meff is not None:
        path = output_dir / "all_effective_masses.html"
        hv.save(all_meff, str(path), backend="bokeh")
        saved_paths["all_effective_masses"] = path

    spectrum = plotter.build_mass_spectrum_bar()
    if spectrum is not None:
        path = output_dir / "mass_spectrum.html"
        hv.save(spectrum, str(path), backend="bokeh")
        saved_paths["mass_spectrum"] = path

    return saved_paths
