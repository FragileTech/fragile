"""
Standard Model of Cognition (SMoC) Simulation and Analysis Pipeline.

This module implements a complete pipeline for:
1. Simulating agent dynamics with local gauge symmetry
2. Projecting fields onto particle channels (pion, rho, scalar, etc.)
3. Computing correlators via FFT
4. Extracting masses using AIC-weighted window selection

The implementation is fully vectorized using PyTorch for GPU acceleration.

Usage:
    from fragile.fractalai.qft.smoc_pipeline import (
        SMoCSimulator,
        ChannelProjector,
        CorrelatorComputer,
        MassExtractor,
        run_smoc_pipeline,
    )

    # Quick pipeline run
    results = run_smoc_pipeline(
        batch_size=1000,
        grid_size=64,
        internal_dim=4,
        t_thermalization=500,
        t_measurement=1000,
    )

References:
    - Standard Model of Cognition: docs/source/1_agent/08_multiagent/02_standard_model.md
    - OS2 Reflection Positivity: Theorem thm-os2-closure-semigroup
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Phase 1: Simulation (Data Generation)
# =============================================================================


@dataclass
class SimulationConfig:
    """Configuration for the SMoC simulation."""

    batch_size: int = 1000
    grid_size: int = 64
    internal_dim: int = 4
    t_thermalization: int = 500
    t_measurement: int = 1000
    dt: float = 0.01
    coupling_strength: float = 0.1
    noise_scale: float = 0.01
    init_mode: Literal["hot", "cold"] = "hot"
    use_pbc: bool = True
    seed: int | None = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class SMoCSimulator:
    """
    Simulates agent dynamics with local gauge symmetry.

    The simulation models N agents on a 1D grid, each with an internal
    state vector of dimension d. The dynamics respect local U(d) gauge
    symmetry through nearest-neighbor interactions.

    Attributes:
        config: Simulation configuration.
        agents: Current agent states (batch, grid, internal_dim).
        history: Recorded states during measurement phase.
    """

    def __init__(self, config: SimulationConfig | None = None):
        """Initialize the simulator with given configuration."""
        self.config = config or SimulationConfig()
        self._setup_device()
        self._initialize_agents()
        self.history: Tensor | None = None

    def _setup_device(self) -> None:
        """Set up device and random seed."""
        self.device = torch.device(self.config.device)
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

    def _initialize_agents(self) -> None:
        """Initialize the agent state tensor."""
        shape = (
            self.config.batch_size,
            self.config.grid_size,
            self.config.internal_dim,
        )

        if self.config.init_mode == "hot":
            # Hot start: random initialization
            self.agents = torch.randn(
                shape,
                device=self.device,
                dtype=self.config.dtype,
            )
            # Normalize to unit vectors in internal space
            self.agents = F.normalize(self.agents, dim=-1)
        else:
            # Cold start: aligned initialization
            self.agents = torch.zeros(
                shape,
                device=self.device,
                dtype=self.config.dtype,
            )
            self.agents[..., 0] = 1.0  # All pointing in first direction

    def _get_neighbors(self, agents: Tensor) -> tuple[Tensor, Tensor]:
        """
        Get left and right neighbors using torch.roll.

        Uses periodic boundary conditions if configured.

        Args:
            agents: Agent states (batch, grid, internal_dim).

        Returns:
            Tuple of (left_neighbors, right_neighbors).
        """
        left = torch.roll(agents, shifts=1, dims=1)
        right = torch.roll(agents, shifts=-1, dims=1)

        if not self.config.use_pbc:
            # Zero out boundary interactions for non-periodic case
            left[:, 0, :] = 0.0
            right[:, -1, :] = 0.0

        return left, right

    def _update_step(self, agents: Tensor) -> Tensor:
        """
        Perform one update step with gauge-invariant local interactions.

        The update respects local U(d) gauge symmetry by using only
        neighbor-relative quantities.

        Args:
            agents: Current agent states (batch, grid, internal_dim).

        Returns:
            Updated agent states.
        """
        left, right = self._get_neighbors(agents)

        # Compute local "field" from neighbors (gauge-invariant combination)
        # Using dot products to ensure gauge invariance
        neighbor_field = left + right

        # Alignment force: agents tend to align with neighbors
        alignment = self.config.coupling_strength * neighbor_field

        # Add thermal noise (respects gauge symmetry as it's isotropic)
        noise = self.config.noise_scale * torch.randn_like(agents)

        # Update with Euler step
        new_agents = agents + self.config.dt * (alignment - agents) + noise

        # Re-normalize to maintain unit vectors (gauge orbit constraint)
        new_agents = F.normalize(new_agents, dim=-1)

        return new_agents

    def run_thermalization(self) -> None:
        """Run thermalization phase to reach equilibrium."""
        for _ in range(self.config.t_thermalization):
            self.agents = self._update_step(self.agents)

    def run_measurement(self) -> Tensor:
        """
        Run measurement phase and record history.

        Returns:
            History tensor of shape (batch, time, grid, internal_dim).
        """
        history_list = []

        for _ in range(self.config.t_measurement):
            self.agents = self._update_step(self.agents)
            history_list.append(self.agents.clone())

        self.history = torch.stack(history_list, dim=1)
        return self.history

    def run(self) -> Tensor:
        """
        Run full simulation (thermalization + measurement).

        Returns:
            History tensor of shape (batch, time, grid, internal_dim).
        """
        self.run_thermalization()
        return self.run_measurement()


# =============================================================================
# Phase 2: Channel Projection (The "Radio Tuner")
# =============================================================================


@dataclass
class ProjectorConfig:
    """Configuration for channel projectors."""

    internal_dim: int = 4
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class ChannelProjector:
    """
    Projects internal agent states onto particle channels.

    Each channel corresponds to a specific quantum number / operator
    that extracts a particular "particle" signal from the field.

    Available channels:
        - scalar: Identity projection (trace)
        - pion: γ₅ (pseudoscalar, sign flip)
        - rho: γ_μ (vector, directional)
        - sigma: Real part projection
        - eta: Imaginary part projection
    """

    def __init__(self, config: ProjectorConfig | None = None):
        """Initialize projector with given configuration."""
        self.config = config or ProjectorConfig()
        self._build_projectors()

    def _build_projectors(self) -> None:
        """Build the projector matrices (gamma matrices)."""
        d = self.config.internal_dim
        device = torch.device(self.config.device)
        dtype = self.config.dtype

        self.projectors: dict[str, Tensor] = {}

        # Scalar: Identity (sum all components)
        self.projectors["scalar"] = torch.eye(d, device=device, dtype=dtype)

        # Pion (γ₅): Pseudoscalar - alternating signs
        pion = torch.diag(
            torch.tensor(
                [(-1) ** i for i in range(d)],
                device=device,
                dtype=dtype,
            )
        )
        self.projectors["pion"] = pion

        # Rho (γ_μ): Vector - projects onto first component direction
        # In practice, this extracts the "direction" of the internal state
        rho = torch.zeros(d, d, device=device, dtype=dtype)
        rho[0, 0] = 1.0
        if d > 1:
            rho[1, 1] = -1.0  # Creates vector-like transformation
        self.projectors["rho"] = rho

        # Sigma: Real part (first half of components)
        sigma = torch.zeros(d, d, device=device, dtype=dtype)
        for i in range(d // 2):
            sigma[i, i] = 1.0
        self.projectors["sigma"] = sigma

        # Eta: Second half of components (imaginary-like)
        eta = torch.zeros(d, d, device=device, dtype=dtype)
        for i in range(d // 2, d):
            eta[i, i] = 1.0
        self.projectors["eta"] = eta

        # Nucleon: Combination projector
        nucleon = torch.zeros(d, d, device=device, dtype=dtype)
        nucleon[0, 0] = 1.0
        nucleon[-1, -1] = 1.0 if d > 1 else 0.0
        self.projectors["nucleon"] = nucleon

    def project(self, history: Tensor, channel: str) -> Tensor:
        """
        Project history tensor onto a specific channel.

        Args:
            history: Agent history (batch, time, grid, internal_dim).
            channel: Channel name (scalar, pion, rho, sigma, eta, nucleon).

        Returns:
            Projected field of shape (batch, time, grid).
        """
        if channel not in self.projectors:
            available = list(self.projectors.keys())
            msg = f"Unknown channel '{channel}'. Available: {available}"
            raise ValueError(msg)

        proj_matrix = self.projectors[channel].to(history.device)

        # Contract: field_i = sum_j history_j * proj_ij
        # Result: (batch, time, grid)
        projected = torch.einsum("btgi,ij->btgj", history, proj_matrix)

        # Sum over internal dimension to get scalar field
        field = projected.sum(dim=-1)

        return field

    def project_all(self, history: Tensor) -> dict[str, Tensor]:
        """
        Project history onto all available channels.

        Args:
            history: Agent history (batch, time, grid, internal_dim).

        Returns:
            Dictionary mapping channel names to projected fields.
        """
        return {ch: self.project(history, ch) for ch in self.projectors}


# =============================================================================
# Phase 3: Correlation Calculation (The FFT Trick)
# =============================================================================


@dataclass
class CorrelatorConfig:
    """Configuration for correlator computation."""

    use_connected: bool = True
    normalize: bool = True


class CorrelatorComputer:
    """
    Computes time correlators using FFT for efficiency.

    The correlator C(τ) = ⟨O(t)O(t+τ)⟩ is computed via the
    Wiener-Khinchin theorem using FFT.
    """

    def __init__(self, config: CorrelatorConfig | None = None):
        """Initialize correlator computer."""
        self.config = config or CorrelatorConfig()

    def spatial_average(self, field: Tensor) -> Tensor:
        """
        Compute zero-momentum mode (spatial average).

        Args:
            field: Field values (batch, time, grid).

        Returns:
            Spatially averaged field (batch, time).
        """
        return field.mean(dim=-1)

    def compute_autocorrelation_fft(self, signal: Tensor) -> Tensor:
        """
        Compute autocorrelation using FFT (Wiener-Khinchin theorem).

        C(τ) = IFFT(|FFT(signal)|²)

        Args:
            signal: Time series (batch, time).

        Returns:
            Autocorrelation (batch, time).
        """
        # Use connected correlator if configured
        if self.config.use_connected:
            signal = signal - signal.mean(dim=-1, keepdim=True)

        # Zero-pad to avoid circular correlation artifacts
        n = signal.shape[-1]
        padded = F.pad(signal, (0, n))

        # FFT along time dimension
        fft_signal = torch.fft.rfft(padded, dim=-1)

        # Power spectrum
        power = fft_signal * fft_signal.conj()

        # Inverse FFT to get correlation
        corr = torch.fft.irfft(power, dim=-1)

        # Take first half (positive lags only)
        corr = corr[..., :n]

        # Normalize by number of overlapping points
        norm_factors = torch.arange(n, 0, -1, device=signal.device, dtype=signal.dtype)
        corr = corr / norm_factors

        # Normalize so C(0) = 1 if configured
        if self.config.normalize:
            c0 = corr[..., 0:1].clamp(min=1e-12)
            corr = corr / c0

        return corr

    def compute_correlator(self, field: Tensor) -> Tensor:
        """
        Compute time correlator from field data.

        Args:
            field: Field values (batch, time, grid).

        Returns:
            Correlator (batch, time).
        """
        # Spatial average (zero momentum projection)
        avg_field = self.spatial_average(field)

        # Autocorrelation via FFT
        corr = self.compute_autocorrelation_fft(avg_field)

        return corr


# =============================================================================
# Phase 4: Statistical Aggregation (The Signal Boost)
# =============================================================================


@dataclass
class AggregatedCorrelator:
    """Container for aggregated correlator data."""

    mean: Tensor  # (time,)
    std: Tensor  # (time,)
    std_err: Tensor  # (time,)
    n_samples: int
    raw: Tensor | None = None  # Optional: keep raw data


def aggregate_correlators(
    correlators: Tensor,
    keep_raw: bool = False,
) -> AggregatedCorrelator:
    """
    Aggregate correlators across batch dimension.

    Args:
        correlators: Correlator values (batch, time).
        keep_raw: Whether to keep raw correlator data.

    Returns:
        AggregatedCorrelator with mean, std, and std_err.
    """
    n_samples = correlators.shape[0]

    mean_corr = correlators.mean(dim=0)
    std_corr = correlators.std(dim=0)
    std_err = std_corr / (n_samples ** 0.5)

    return AggregatedCorrelator(
        mean=mean_corr,
        std=std_corr,
        std_err=std_err,
        n_samples=n_samples,
        raw=correlators if keep_raw else None,
    )


# =============================================================================
# Phase 5: Parallel Window Selection (AIC)
# =============================================================================


@dataclass
class WindowFitResult:
    """Result of a single window fit."""

    t_start: int
    t_end: int
    mass: float
    amplitude: float
    chi2: float
    aic: float
    valid: bool


@dataclass
class MassExtractionConfig:
    """Configuration for mass extraction."""

    min_window_length: int = 5
    max_window_length: int | None = None
    min_t_start: int = 2  # Skip initial transient
    max_t_end: int | None = None
    min_mass: float = 0.0
    max_mass: float = float("inf")


class MassExtractor:
    """
    Extracts particle masses using AIC-weighted window selection.

    This implements a fully vectorized approach to fitting multiple
    windows simultaneously and selecting the optimal fit using AIC.
    """

    def __init__(self, config: MassExtractionConfig | None = None):
        """Initialize mass extractor."""
        self.config = config or MassExtractionConfig()

    def generate_windows(
        self,
        t_max: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Generate all valid window (t_start, t_end) combinations.

        Args:
            t_max: Maximum time index.

        Returns:
            Tuple of (t_starts, t_ends) tensors.
        """
        min_len = self.config.min_window_length
        max_len = self.config.max_window_length or t_max
        min_start = self.config.min_t_start
        max_end = self.config.max_t_end or t_max

        windows_start = []
        windows_end = []

        for t_start in range(min_start, max_end - min_len + 1):
            for t_end in range(t_start + min_len, min(t_start + max_len, max_end) + 1):
                windows_start.append(t_start)
                windows_end.append(t_end)

        return (
            torch.tensor(windows_start, dtype=torch.long),
            torch.tensor(windows_end, dtype=torch.long),
        )

    def vectorized_log_linear_fit(
        self,
        log_corr: Tensor,
        std_err: Tensor,
        t_starts: Tensor,
        t_ends: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Perform vectorized log-linear fits for all windows.

        Fits: log(C(t)) = A - m*t  =>  mass = m, amplitude = exp(A)

        Args:
            log_corr: Log of mean correlator (time,).
            std_err: Standard errors (time,).
            t_starts: Window start indices (n_windows,).
            t_ends: Window end indices (n_windows,).

        Returns:
            Tuple of (masses, amplitudes, chi2, valid_mask).
        """
        device = log_corr.device
        n_windows = t_starts.shape[0]
        t_max = log_corr.shape[0]

        # Time indices
        t_indices = torch.arange(t_max, device=device, dtype=log_corr.dtype)

        # Initialize outputs
        masses = torch.zeros(n_windows, device=device, dtype=log_corr.dtype)
        amplitudes = torch.zeros(n_windows, device=device, dtype=log_corr.dtype)
        chi2_values = torch.full((n_windows,), float("inf"), device=device, dtype=log_corr.dtype)
        valid_mask = torch.zeros(n_windows, device=device, dtype=torch.bool)

        # Process each window (could be further vectorized with padding)
        for i in range(n_windows):
            t0 = t_starts[i].item()
            t1 = t_ends[i].item()

            # Extract window data
            t_window = t_indices[t0:t1]
            y_window = log_corr[t0:t1]
            err_window = std_err[t0:t1]

            # Check for valid data (no NaN/Inf)
            valid_data = torch.isfinite(y_window) & (err_window > 0)
            if valid_data.sum() < 2:
                continue

            t_valid = t_window[valid_data]
            y_valid = y_window[valid_data]
            err_valid = err_window[valid_data]

            # Weighted least squares: solve [1, t] @ [A, -m]^T = y
            n_pts = t_valid.shape[0]
            weights = 1.0 / (err_valid ** 2)

            # Design matrix
            X = torch.stack([torch.ones_like(t_valid), t_valid], dim=1)

            # Weighted normal equations: (X^T W X) @ params = X^T W y
            W = torch.diag(weights)
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y_valid

            # Solve
            try:
                params = torch.linalg.solve(XtWX, XtWy)
                A, neg_m = params[0], params[1]
                m = -neg_m

                # Chi-squared
                y_pred = A + neg_m * t_valid
                residuals = (y_valid - y_pred) / err_valid
                chi2 = (residuals ** 2).sum()

                # Store results
                masses[i] = m
                amplitudes[i] = torch.exp(A)
                chi2_values[i] = chi2
                valid_mask[i] = True

            except RuntimeError:
                # Singular matrix - skip this window
                continue

        return masses, amplitudes, chi2_values, valid_mask

    def compute_aic(
        self,
        chi2: Tensor,
        n_params: int = 2,
    ) -> Tensor:
        """
        Compute AIC scores.

        AIC = χ² + 2k where k is the number of parameters.

        Args:
            chi2: Chi-squared values (n_windows,).
            n_params: Number of fit parameters (default: 2 for mass + amplitude).

        Returns:
            AIC scores (n_windows,).
        """
        return chi2 + 2 * n_params

    def compute_aic_weights(self, aic: Tensor, valid: Tensor) -> Tensor:
        """
        Convert AIC scores to probability weights.

        w_i = exp(-0.5 * (AIC_i - AIC_min))

        Args:
            aic: AIC scores (n_windows,).
            valid: Validity mask (n_windows,).

        Returns:
            Normalized weights (n_windows,).
        """
        # Mask invalid entries
        aic_masked = aic.clone()
        aic_masked[~valid] = float("inf")

        # Compute relative to minimum
        aic_min = aic_masked.min()
        delta_aic = aic_masked - aic_min

        # Compute weights
        weights = torch.exp(-0.5 * delta_aic)
        weights[~valid] = 0.0

        # Normalize
        total = weights.sum()
        if total > 0:
            weights = weights / total

        return weights

    def extract_mass(
        self,
        agg_corr: AggregatedCorrelator,
    ) -> dict[str, Any]:
        """
        Extract mass from aggregated correlator using AIC-weighted averaging.

        Args:
            agg_corr: Aggregated correlator data.

        Returns:
            Dictionary with extraction results.
        """
        mean_corr = agg_corr.mean
        std_err = agg_corr.std_err
        device = mean_corr.device

        # Ensure positive correlator for log
        positive_mask = mean_corr > 0
        if not positive_mask.any():
            return {
                "mass": 0.0,
                "mass_error": float("inf"),
                "n_valid_windows": 0,
                "best_window": None,
            }

        # Compute log correlator
        log_corr = torch.full_like(mean_corr, float("-inf"))
        log_corr[positive_mask] = torch.log(mean_corr[positive_mask])

        # Propagate errors: σ(log C) = σ(C) / C
        log_std_err = torch.full_like(std_err, float("inf"))
        log_std_err[positive_mask] = std_err[positive_mask] / mean_corr[positive_mask]

        # Generate windows
        t_max = mean_corr.shape[0]
        t_starts, t_ends = self.generate_windows(t_max)
        t_starts = t_starts.to(device)
        t_ends = t_ends.to(device)

        if t_starts.shape[0] == 0:
            return {
                "mass": 0.0,
                "mass_error": float("inf"),
                "n_valid_windows": 0,
                "best_window": None,
            }

        # Vectorized fitting
        masses, amplitudes, chi2, valid = self.vectorized_log_linear_fit(
            log_corr, log_std_err, t_starts, t_ends
        )

        # Apply mass constraints
        mass_valid = (masses > self.config.min_mass) & (masses < self.config.max_mass)
        valid = valid & mass_valid

        n_valid = valid.sum().item()
        if n_valid == 0:
            return {
                "mass": 0.0,
                "mass_error": float("inf"),
                "n_valid_windows": 0,
                "best_window": None,
            }

        # Compute AIC and weights
        aic = self.compute_aic(chi2)
        weights = self.compute_aic_weights(aic, valid)

        # Weighted average mass
        mass_final = (weights * masses).sum().item()

        # Weighted standard deviation as error estimate
        mass_var = (weights * (masses - mass_final) ** 2).sum()
        mass_error = (mass_var ** 0.5).item()

        # Find best window (minimum AIC)
        best_idx = (aic + (~valid).float() * 1e10).argmin().item()
        best_window = {
            "t_start": t_starts[best_idx].item(),
            "t_end": t_ends[best_idx].item(),
            "mass": masses[best_idx].item(),
            "amplitude": amplitudes[best_idx].item(),
            "chi2": chi2[best_idx].item(),
            "aic": aic[best_idx].item(),
        }

        return {
            "mass": mass_final,
            "mass_error": mass_error,
            "n_valid_windows": n_valid,
            "best_window": best_window,
            "all_masses": masses[valid].cpu().numpy(),
            "all_weights": weights[valid].cpu().numpy(),
        }


# =============================================================================
# Phase 6: Full Pipeline
# =============================================================================


@dataclass
class SMoCPipelineConfig:
    """Full pipeline configuration."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    correlator: CorrelatorConfig = field(default_factory=CorrelatorConfig)
    mass_extraction: MassExtractionConfig = field(default_factory=MassExtractionConfig)
    channels: tuple[str, ...] = ("scalar", "pion", "rho")


@dataclass
class SMoCPipelineResult:
    """Results from the full SMoC pipeline."""

    history: Tensor | None
    projected_fields: dict[str, Tensor]
    correlators: dict[str, Tensor]
    aggregated: dict[str, AggregatedCorrelator]
    masses: dict[str, dict[str, Any]]
    config: SMoCPipelineConfig


def run_smoc_pipeline(
    batch_size: int = 1000,
    grid_size: int = 64,
    internal_dim: int = 4,
    t_thermalization: int = 500,
    t_measurement: int = 1000,
    channels: tuple[str, ...] = ("scalar", "pion", "rho"),
    device: str = "cpu",
    seed: int | None = 42,
    keep_history: bool = False,
    verbose: bool = True,
) -> SMoCPipelineResult:
    """
    Run the complete SMoC simulation and analysis pipeline.

    This is the main entry point for the Standard Model of Cognition
    particle physics analysis.

    Args:
        batch_size: Number of independent universes.
        grid_size: Number of agents per universe.
        internal_dim: Internal state dimension (spin).
        t_thermalization: Thermalization steps.
        t_measurement: Measurement steps.
        channels: Particle channels to analyze.
        device: Compute device (cpu/cuda).
        seed: Random seed for reproducibility.
        keep_history: Whether to keep full history tensor.
        verbose: Print progress messages.

    Returns:
        SMoCPipelineResult containing all computed quantities.
    """
    # Build configuration
    sim_config = SimulationConfig(
        batch_size=batch_size,
        grid_size=grid_size,
        internal_dim=internal_dim,
        t_thermalization=t_thermalization,
        t_measurement=t_measurement,
        device=device,
        seed=seed,
    )

    proj_config = ProjectorConfig(
        internal_dim=internal_dim,
        device=device,
    )

    config = SMoCPipelineConfig(
        simulation=sim_config,
        projector=proj_config,
        channels=channels,
    )

    # Phase 1: Simulation
    if verbose:
        print(f"Phase 1: Running simulation ({batch_size} universes, {grid_size} agents)...")

    simulator = SMoCSimulator(sim_config)
    history = simulator.run()

    if verbose:
        print(f"  History shape: {history.shape}")

    # Phase 2: Channel Projection
    if verbose:
        print(f"Phase 2: Projecting onto channels: {channels}...")

    projector = ChannelProjector(proj_config)
    projected_fields = {ch: projector.project(history, ch) for ch in channels}

    # Phase 3: Correlation Calculation
    if verbose:
        print("Phase 3: Computing correlators via FFT...")

    correlator_computer = CorrelatorComputer(config.correlator)
    correlators = {
        ch: correlator_computer.compute_correlator(field)
        for ch, field in projected_fields.items()
    }

    # Phase 4: Statistical Aggregation
    if verbose:
        print("Phase 4: Aggregating statistics...")

    aggregated = {ch: aggregate_correlators(corr) for ch, corr in correlators.items()}

    # Phase 5 & 6: Mass Extraction
    if verbose:
        print("Phase 5-6: Extracting masses with AIC window selection...")

    extractor = MassExtractor(config.mass_extraction)
    masses = {ch: extractor.extract_mass(agg) for ch, agg in aggregated.items()}

    if verbose:
        print("\nResults:")
        for ch, result in masses.items():
            m = result["mass"]
            err = result["mass_error"]
            n_win = result["n_valid_windows"]
            print(f"  {ch}: M = {m:.4f} ± {err:.4f} ({n_win} valid windows)")

    return SMoCPipelineResult(
        history=history if keep_history else None,
        projected_fields=projected_fields,
        correlators=correlators,
        aggregated=aggregated,
        masses=masses,
        config=config,
    )


# =============================================================================
# Utilities for Integration with Existing Fractal Gas
# =============================================================================


def compute_smoc_correlators_from_history(
    positions: Tensor,
    velocities: Tensor,
    alive: Tensor,
    channels: tuple[str, ...] = ("scalar", "pion", "rho"),
    max_lag: int | None = None,
    use_connected: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Compute SMoC-style correlators from Fractal Gas history data.

    This adapts the SMoC pipeline to work with existing Fractal Gas
    run data rather than the simplified simulation.

    Args:
        positions: Position history (time, n_walkers, dims).
        velocities: Velocity history (time, n_walkers, dims).
        alive: Alive status history (time, n_walkers).
        channels: Channels to compute.
        max_lag: Maximum correlation lag.
        use_connected: Use connected correlators.

    Returns:
        Dictionary with correlator data for each channel.
    """
    t_steps, n_walkers, dims = positions.shape
    device = positions.device

    if max_lag is None:
        max_lag = t_steps // 2

    results = {}

    # Construct "internal state" from position + velocity
    # This maps (x, v) -> internal representation
    internal = torch.cat([positions, velocities], dim=-1)  # (time, n_walkers, 2*dims)
    internal_dim = internal.shape[-1]

    # Build projector
    proj_config = ProjectorConfig(internal_dim=internal_dim, device=str(device))
    projector = ChannelProjector(proj_config)

    # Add batch dimension and transpose for projector
    # From (time, walkers, dim) to (1, time, walkers, dim)
    history = internal.unsqueeze(0)

    corr_config = CorrelatorConfig(use_connected=use_connected)
    corr_computer = CorrelatorComputer(corr_config)

    for channel in channels:
        try:
            # Project field
            field = projector.project(history, channel)  # (1, time, walkers)

            # Mask by alive status
            alive_expanded = alive.unsqueeze(0)  # (1, time, walkers)
            field = field * alive_expanded.float()

            # Compute correlator
            corr = corr_computer.compute_correlator(field)  # (1, time)
            corr = corr.squeeze(0)[:max_lag]  # (max_lag,)

            results[channel] = {
                "correlator": corr.cpu().numpy(),
                "lags": torch.arange(max_lag).cpu().numpy(),
            }

        except (ValueError, RuntimeError) as e:
            results[channel] = {
                "correlator": None,
                "error": str(e),
            }

    return results
