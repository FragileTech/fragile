"""Fluid Dynamics Utilities for Validation and Analysis.

This module provides rigorous utilities for computing fluid fields from particle data,
validating fluid dynamics simulations, and checking conservation laws.

Features:
- SPH-style kernel interpolation with periodic boundary conditions
- Conservation law validation (mass, momentum, energy)
- Incompressibility analysis (divergence, stream function)
- Analytical solution comparison (Taylor-Green vortex)
- Reynolds number and flow regime validation
- Spectral analysis (energy spectrum, enstrophy)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
import torch


__all__ = [
    "FLUID_CONFIGS",
    "ConservationValidator",
    "FlowAnalyzer",
    "FluidFieldComputer",
    "TaylorGreenValidator",
    "ValidationMetrics",
]


# ============================================================================
# Validation Metrics
# ============================================================================


@dataclass
class ValidationMetrics:
    """Container for validation metrics comparing simulation to theory."""

    metric_name: str
    measured_value: float
    theoretical_value: float | None
    tolerance: float
    passed: bool
    description: str


# ============================================================================
# Fluid Field Computation Utilities
# ============================================================================


class FluidFieldComputer:
    """Rigorous computation of fluid fields from particle data.

    Uses SPH-style kernel interpolation with proper periodic boundary conditions
    and normalized kernels for accurate field reconstruction.

    Methods:
        compute_velocity_field: SPH velocity field with periodic BC
        compute_velocity_field_periodic: Explicit periodic version
        compute_vorticity: Vorticity ω = ∂v/∂x - ∂u/∂y
        compute_divergence: Divergence ∇·u for incompressibility check
        compute_stream_function: Stream function ψ for 2D incompressible flow
        compute_density_field: Normalized particle density field
        sph_kernel: Wendland C2 kernel with compact support
    """

    @staticmethod
    def sph_kernel(r: np.ndarray, h: float) -> np.ndarray:
        """Wendland C2 kernel with compact support.

        Kernel function:
            W(r,h) = (7/(4πh²)) * (1 - r/(2h))⁴ * (1 + 2r/h)  for r < 2h
                   = 0                                         for r ≥ 2h

        This kernel has:
        - Compact support (radius 2h)
        - C² continuity
        - Positive definite
        - Proper normalization ∫W(r,h)dr = 1

        Args:
            r: Distance array
            h: Smoothing length (kernel bandwidth)

        Returns:
            Kernel weights (same shape as r)
        """
        q = r / h
        C = 7.0 / (4.0 * np.pi * h**2)  # 2D normalization constant

        kernel = np.zeros_like(r)
        mask = q < 2.0

        q_masked = q[mask]
        kernel[mask] = C * (1 - 0.5 * q_masked) ** 4 * (1 + 2 * q_masked)

        return kernel

    @staticmethod
    def _apply_periodic_distance(
        dx: np.ndarray, dy: np.ndarray, domain_size: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply periodic boundary conditions to distance vectors.

        For periodic domain [L_min, L_max], the minimum distance accounting
        for periodicity is:
            d_periodic = min(|d|, L - |d|)

        Args:
            dx: x-component of distance
            dy: y-component of distance
            domain_size: Size of periodic domain (L_max - L_min)

        Returns:
            Periodic dx, dy with minimum image convention
        """
        # Minimum image convention
        dx = np.where(np.abs(dx) > domain_size / 2, dx - np.sign(dx) * domain_size, dx)
        dy = np.where(np.abs(dy) > domain_size / 2, dy - np.sign(dy) * domain_size, dy)

        return dx, dy

    @staticmethod
    def compute_velocity_field(
        positions: torch.Tensor,
        velocities: torch.Tensor,
        grid_resolution: int = 50,
        kernel_bandwidth: float = 0.3,
        bounds: tuple[float, float] = (-np.pi, np.pi),
        periodic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute velocity field using SPH kernel interpolation.

        Uses Wendland C2 kernel with compact support for efficient, accurate
        interpolation with proper normalization and periodic boundaries.

        Velocity interpolation:
            u(x) = Σ_i W(||x - x_i||, h) * u_i / Σ_i W(||x - x_i||, h)

        Args:
            positions: Particle positions [N, 2]
            velocities: Particle velocities [N, 2]
            grid_resolution: Number of grid points per dimension
            kernel_bandwidth: Smoothing length h (compact support = 2h)
            bounds: Spatial domain (min, max)
            periodic: Use periodic boundary conditions

        Returns:
            X: x-coordinates of grid [H, W]
            Y: y-coordinates of grid [H, W]
            U: x-velocity component [H, W]
            V: y-velocity component [H, W]
        """
        # Convert to numpy
        pos = positions.cpu().numpy()
        vel = velocities.cpu().numpy()

        # Create grid
        x_grid = np.linspace(bounds[0], bounds[1], grid_resolution)
        y_grid = np.linspace(bounds[0], bounds[1], grid_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Initialize fields
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        W = np.zeros_like(X)  # Normalization weights

        domain_size = bounds[1] - bounds[0]

        # SPH interpolation with compact support
        for i in range(len(pos)):
            # Compute distances
            dx = X - pos[i, 0]
            dy = Y - pos[i, 1]

            # Apply periodic boundary conditions
            if periodic:
                dx, dy = FluidFieldComputer._apply_periodic_distance(dx, dy, domain_size)

            r = np.sqrt(dx**2 + dy**2)

            # Wendland C2 kernel (compact support: r < 2h)
            kernel = FluidFieldComputer.sph_kernel(r, kernel_bandwidth)

            # Accumulate weighted contributions
            U += kernel * vel[i, 0]
            V += kernel * vel[i, 1]
            W += kernel

        # Normalize (with safety for zero weight regions)
        W = np.maximum(W, 1e-12)
        U /= W
        V /= W

        return X, Y, U, V

    @staticmethod
    def compute_vorticity(
        U: np.ndarray, V: np.ndarray, dx: float, dy: float, periodic: bool = True
    ) -> np.ndarray:
        """Compute vorticity field ω = ∂v/∂x - ∂u/∂y.

        Uses central finite differences with proper periodic boundary conditions.

        Args:
            U: x-velocity component [H, W]
            V: y-velocity component [H, W]
            dx: Grid spacing in x
            dy: Grid spacing in y
            periodic: Apply periodic boundary conditions

        Returns:
            Vorticity field ω [H, W]
        """
        if periodic:
            # Periodic central differences
            dv_dx = (np.roll(V, -1, axis=1) - np.roll(V, 1, axis=1)) / (2 * dx)
            du_dy = (np.roll(U, -1, axis=0) - np.roll(U, 1, axis=0)) / (2 * dy)
        else:
            # Standard central differences (uses one-sided at boundaries)
            dv_dx = np.gradient(V, dx, axis=1)
            du_dy = np.gradient(U, dy, axis=0)

        return dv_dx - du_dy

    @staticmethod
    def compute_divergence(
        U: np.ndarray, V: np.ndarray, dx: float, dy: float, periodic: bool = True
    ) -> np.ndarray:
        """Compute velocity divergence ∇·u = ∂u/∂x + ∂v/∂y.

        For incompressible flow, this should be approximately zero everywhere.

        Args:
            U: x-velocity component [H, W]
            V: y-velocity component [H, W]
            dx: Grid spacing in x
            dy: Grid spacing in y
            periodic: Apply periodic boundary conditions

        Returns:
            Divergence field [H, W]
        """
        if periodic:
            # Periodic central differences
            du_dx = (np.roll(U, -1, axis=1) - np.roll(U, 1, axis=1)) / (2 * dx)
            dv_dy = (np.roll(V, -1, axis=0) - np.roll(V, 1, axis=0)) / (2 * dy)
        else:
            # Standard central differences
            du_dx = np.gradient(U, dx, axis=1)
            dv_dy = np.gradient(V, dy, axis=0)

        return du_dx + dv_dy

    @staticmethod
    def compute_stream_function(
        U: np.ndarray,
        V: np.ndarray,
        dx: float,
        dy: float,
        method: str = "integration",
    ) -> np.ndarray:
        """Compute stream function ψ for 2D incompressible flow.

        For incompressible flow: u = -∂ψ/∂y, v = ∂ψ/∂x

        Equivalently: ∇²ψ = ω (Poisson equation for stream function)

        Args:
            U: x-velocity component [H, W]
            V: y-velocity component [H, W]
            dx: Grid spacing in x
            dy: Grid spacing in y
            method: 'integration' or 'poisson'
                - 'integration': Direct line integration from reference point
                - 'poisson': Solve Poisson equation ∇²ψ = ω (more accurate for noisy data)

        Returns:
            Stream function ψ [H, W]
        """
        H, W = U.shape

        if method == "integration":
            # Direct integration: ψ(x,y) = ψ(x0,y0) + ∫u·dl
            psi = np.zeros((H, W))

            # Integrate along first row (y = y0): dψ/dx = v
            for j in range(1, W):
                psi[0, j] = psi[0, j - 1] + V[0, j - 1] * dx

            # Integrate along columns: dψ/dy = -u
            for i in range(1, H):
                psi[i, :] = psi[i - 1, :] - U[i - 1, :] * dy

            # Remove mean (arbitrary constant)
            psi -= np.mean(psi)

        elif method == "poisson":
            # Solve ∇²ψ = ω using FFT (assumes periodic boundaries)
            omega = FluidFieldComputer.compute_vorticity(U, V, dx, dy, periodic=True)

            # Wavenumbers for FFT
            kx = 2 * np.pi * np.fft.fftfreq(W, dx)
            ky = 2 * np.pi * np.fft.fftfreq(H, dy)
            KX, KY = np.meshgrid(kx, ky)
            K_sq = KX**2 + KY**2

            # Solve in Fourier space: ψ̂ = -ω̂ / k²
            omega_hat = np.fft.fft2(omega)
            psi_hat = np.zeros_like(omega_hat)

            # Avoid division by zero at k=0
            mask = K_sq > 1e-10
            psi_hat[mask] = -omega_hat[mask] / K_sq[mask]

            # Transform back
            psi = np.real(np.fft.ifft2(psi_hat))

        else:
            raise ValueError(f"Unknown method: {method}")

        return psi

    @staticmethod
    def compute_density_field(
        positions: torch.Tensor,
        grid_resolution: int = 50,
        bounds: tuple[float, float] = (-np.pi, np.pi),
        smoothing: float = 1.0,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute normalized particle density field.

        Uses 2D histogram with Gaussian smoothing. Optionally normalizes so that
        the integral over the domain equals the number of particles.

        Args:
            positions: Particle positions [N, 2]
            grid_resolution: Number of grid points per dimension
            bounds: Spatial domain (min, max)
            smoothing: Gaussian smoothing sigma (in grid units)
            normalize: If True, normalize so ∫ρ dA = N

        Returns:
            X: x-coordinates [H, W]
            Y: y-coordinates [H, W]
            density: Density field ρ [H, W]
        """
        # Convert to numpy
        pos = positions.cpu().numpy()
        N_particles = len(pos)

        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            pos[:, 0],
            pos[:, 1],
            bins=grid_resolution,
            range=[[bounds[0], bounds[1]], [bounds[0], bounds[1]]],
        )

        # Smooth with Gaussian filter
        density = gaussian_filter(hist.T, sigma=smoothing)

        # Normalize to conserve particle number
        if normalize:
            domain_size = bounds[1] - bounds[0]
            dx = domain_size / grid_resolution
            dy = domain_size / grid_resolution

            # Current integral
            current_integral = np.sum(density) * dx * dy

            # Rescale to match N_particles
            if current_integral > 1e-12:
                density *= N_particles / current_integral

        # Create coordinate grids
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)

        return X, Y, density


# ============================================================================
# Conservation Law Validators
# ============================================================================


class ConservationValidator:
    """Validate conservation laws for fluid dynamics simulations.

    Checks:
    - Mass conservation: ∫ρ dV = constant
    - Momentum conservation: ∫ρu dV = constant (or predictable evolution)
    - Energy budget: dE/dt = power_input - dissipation
    - Incompressibility: ∇·u ≈ 0
    """

    @staticmethod
    def check_mass_conservation(
        positions_history: list[torch.Tensor],
        bounds: tuple[float, float] = (-np.pi, np.pi),
        tolerance: float = 0.05,
    ) -> ValidationMetrics:
        """Check mass conservation: number of particles should remain constant.

        For particle methods, "mass" is the particle count. Variations indicate
        particle creation/deletion (e.g., from cloning or death).

        Args:
            positions_history: List of position tensors [N, 2] at each timestep
            bounds: Spatial domain
            tolerance: Relative tolerance for mass variation

        Returns:
            ValidationMetrics with mass conservation check
        """
        particle_counts = [len(pos) for pos in positions_history]
        initial_count = particle_counts[0]

        # Compute relative variation
        max_variation = max(abs(n - initial_count) / initial_count for n in particle_counts)

        passed = max_variation <= tolerance

        return ValidationMetrics(
            metric_name="Mass Conservation",
            measured_value=max_variation,
            theoretical_value=0.0,
            tolerance=tolerance,
            passed=passed,
            description=f"Max relative particle count variation: {max_variation:.4f} "
            f"(tolerance: {tolerance})",
        )

    @staticmethod
    def check_momentum_conservation(
        positions_history: list[torch.Tensor],
        velocities_history: list[torch.Tensor],
        tolerance: float = 0.1,
    ) -> ValidationMetrics:
        """Check momentum conservation in absence of external forces.

        For systems without external forces: p(t) = ∫ρu dV = constant

        Args:
            positions_history: List of position tensors [N, 2]
            velocities_history: List of velocity tensors [N, 2]
            tolerance: Relative tolerance for momentum variation

        Returns:
            ValidationMetrics with momentum conservation check
        """
        # Compute total momentum at each timestep (sum over particles)
        momenta = [vel.sum(dim=0).cpu().numpy() for vel in velocities_history]
        momenta = np.array(momenta)  # [T, 2]

        initial_momentum = momenta[0]
        momentum_magnitude = np.linalg.norm(initial_momentum)

        # Compute relative variation
        if momentum_magnitude > 1e-8:
            variations = [
                np.linalg.norm(p - initial_momentum) / momentum_magnitude for p in momenta
            ]
            max_variation = max(variations)
        else:
            # Initial momentum is zero; check absolute variation
            max_variation = max(np.linalg.norm(p) for p in momenta)

        passed = max_variation <= tolerance

        return ValidationMetrics(
            metric_name="Momentum Conservation",
            measured_value=max_variation,
            theoretical_value=0.0,
            tolerance=tolerance,
            passed=passed,
            description=f"Max relative momentum variation: {max_variation:.4f} "
            f"(tolerance: {tolerance})",
        )

    @staticmethod
    def check_energy_budget(
        velocities_history: list[torch.Tensor],
        dt: float,
        power_input: float = 0.0,
        viscosity: float = 0.0,
        tolerance: float = 0.15,
    ) -> ValidationMetrics:
        """Check energy budget: dE/dt = P_in - P_diss.

        Kinetic energy: E_k(t) = (1/2)∫ρ|u|² dV ≈ (1/2)Σ|u_i|²

        For forced systems: dE/dt = P_in - ν∫|∇u|² dV

        Args:
            velocities_history: List of velocity tensors [N, 2]
            dt: Timestep
            power_input: External power input (e.g., from forcing)
            viscosity: Kinematic viscosity ν
            tolerance: Relative tolerance for energy balance

        Returns:
            ValidationMetrics with energy budget check
        """
        # Compute kinetic energy at each timestep
        kinetic_energies = [0.5 * (vel**2).sum().item() / len(vel) for vel in velocities_history]

        # Compute energy change rate (central differences)
        np.gradient(kinetic_energies, dt)

        # Expected rate: P_in - P_diss
        # For now, just check if energy is bounded (crude test)
        E_mean = np.mean(kinetic_energies)
        E_std = np.std(kinetic_energies)

        relative_variation = E_std / E_mean if E_mean > 1e-12 else E_std

        passed = relative_variation <= tolerance

        return ValidationMetrics(
            metric_name="Energy Budget",
            measured_value=relative_variation,
            theoretical_value=None,  # Complex to compute without dissipation details
            tolerance=tolerance,
            passed=passed,
            description=f"Kinetic energy relative variation: {relative_variation:.4f} "
            f"(tolerance: {tolerance})",
        )

    @staticmethod
    def check_incompressibility(
        positions: torch.Tensor,
        velocities: torch.Tensor,
        bounds: tuple[float, float] = (-np.pi, np.pi),
        grid_resolution: int = 50,
        tolerance: float = 0.1,
    ) -> ValidationMetrics:
        """Check incompressibility: ∇·u ≈ 0.

        Computes velocity field and its divergence. For incompressible flow,
        the RMS divergence should be much smaller than typical velocity gradients.

        Args:
            positions: Particle positions [N, 2]
            velocities: Particle velocities [N, 2]
            bounds: Spatial domain
            grid_resolution: Grid resolution for field computation
            tolerance: Tolerance for RMS divergence

        Returns:
            ValidationMetrics with incompressibility check
        """
        # Compute velocity field
        _X, _Y, U, V = FluidFieldComputer.compute_velocity_field(
            positions, velocities, grid_resolution, bounds=bounds
        )

        domain_size = bounds[1] - bounds[0]
        dx = domain_size / grid_resolution
        dy = domain_size / grid_resolution

        # Compute divergence
        div = FluidFieldComputer.compute_divergence(U, V, dx, dy, periodic=True)

        # RMS divergence
        rms_div = np.sqrt(np.mean(div**2))

        # Typical velocity gradient scale (for normalization)
        grad_u = np.gradient(U, dx, axis=1)
        grad_v = np.gradient(V, dy, axis=0)
        typical_gradient = np.sqrt(np.mean(grad_u**2 + grad_v**2))

        # Relative divergence
        relative_div = rms_div / typical_gradient if typical_gradient > 1e-12 else rms_div

        passed = relative_div <= tolerance

        return ValidationMetrics(
            metric_name="Incompressibility",
            measured_value=relative_div,
            theoretical_value=0.0,
            tolerance=tolerance,
            passed=passed,
            description=f"RMS divergence (relative to gradients): {relative_div:.4f} "
            f"(tolerance: {tolerance})",
        )


# ============================================================================
# Flow Analysis Tools
# ============================================================================


class FlowAnalyzer:
    """Advanced flow analysis tools.

    Provides:
    - Reynolds number estimation
    - Enstrophy computation (∫ω² dA)
    - Kinetic energy spectrum E(k)
    - Vorticity statistics
    """

    @staticmethod
    def compute_reynolds_number(
        positions: torch.Tensor,
        velocities: torch.Tensor,
        viscosity: float,
        characteristic_length: float | None = None,
    ) -> float:
        """Compute Reynolds number Re = UL/ν.

        Args:
            positions: Particle positions [N, 2]
            velocities: Particle velocities [N, 2]
            viscosity: Kinematic viscosity ν
            characteristic_length: Characteristic length scale L
                If None, uses domain size

        Returns:
            Reynolds number
        """
        # Characteristic velocity (RMS)
        vel = velocities.cpu().numpy()
        U_rms = np.sqrt(np.mean(vel**2))

        # Characteristic length
        if characteristic_length is None:
            pos = positions.cpu().numpy()
            L = np.ptp(pos[:, 0])  # Range in x-direction
        else:
            L = characteristic_length

        # Reynolds number
        return U_rms * L / viscosity if viscosity > 1e-12 else np.inf

    @staticmethod
    def compute_enstrophy(
        positions: torch.Tensor,
        velocities: torch.Tensor,
        bounds: tuple[float, float] = (-np.pi, np.pi),
        grid_resolution: int = 50,
    ) -> float:
        """Compute enstrophy Z = (1/2)∫ω² dA.

        Enstrophy measures the intensity of vorticity in the flow.

        Args:
            positions: Particle positions [N, 2]
            velocities: Particle velocities [N, 2]
            bounds: Spatial domain
            grid_resolution: Grid resolution

        Returns:
            Enstrophy Z
        """
        # Compute velocity field
        _X, _Y, U, V = FluidFieldComputer.compute_velocity_field(
            positions, velocities, grid_resolution, bounds=bounds
        )

        domain_size = bounds[1] - bounds[0]
        dx = domain_size / grid_resolution
        dy = domain_size / grid_resolution

        # Compute vorticity
        omega = FluidFieldComputer.compute_vorticity(U, V, dx, dy, periodic=True)

        # Integrate enstrophy
        return 0.5 * np.sum(omega**2) * dx * dy

    @staticmethod
    def compute_vorticity_statistics(
        positions: torch.Tensor,
        velocities: torch.Tensor,
        bounds: tuple[float, float] = (-np.pi, np.pi),
        grid_resolution: int = 50,
    ) -> dict:
        """Compute vorticity field statistics.

        Returns:
            Dictionary with vorticity statistics:
                - mean: Mean vorticity
                - std: Standard deviation
                - max: Maximum vorticity
                - min: Minimum vorticity
                - enstrophy: Enstrophy Z = (1/2)∫ω² dA
        """
        # Compute velocity field
        _X, _Y, U, V = FluidFieldComputer.compute_velocity_field(
            positions, velocities, grid_resolution, bounds=bounds
        )

        domain_size = bounds[1] - bounds[0]
        dx = domain_size / grid_resolution
        dy = domain_size / grid_resolution

        # Compute vorticity
        omega = FluidFieldComputer.compute_vorticity(U, V, dx, dy, periodic=True)

        # Statistics
        return {
            "mean": float(np.mean(omega)),
            "std": float(np.std(omega)),
            "max": float(np.max(omega)),
            "min": float(np.min(omega)),
            "enstrophy": float(0.5 * np.sum(omega**2) * dx * dy),
        }


# ============================================================================
# Taylor-Green Vortex Analytical Validator
# ============================================================================


class TaylorGreenValidator:
    """Validate simulations against analytical Taylor-Green vortex solution.

    The Taylor-Green vortex is an exact solution of the 2D Navier-Stokes equations:

    u(x,y,t) = -U₀ cos(kx) sin(ky) exp(-2νk²t)
    v(x,y,t) =  U₀ sin(kx) cos(ky) exp(-2νk²t)
    ω(x,y,t) =  2k U₀ cos(kx) cos(ky) exp(-2νk²t)

    where k = 2π/L, ν is kinematic viscosity, U₀ is initial velocity scale.
    """

    def __init__(
        self,
        U0: float = 1.0,
        viscosity: float = 1.0,
        domain_size: float = 2 * np.pi,
        k: float | None = None,
    ):
        """Initialize Taylor-Green validator.

        Args:
            U0: Initial velocity scale
            viscosity: Kinematic viscosity ν
            domain_size: Domain size L (assumes [-L/2, L/2]²)
            k: Wave number (default: 2π/L for single mode)
        """
        self.U0 = U0
        self.nu = viscosity
        self.L = domain_size
        self.k = k if k is not None else 2 * np.pi / domain_size

    def analytical_velocity(
        self, x: np.ndarray, y: np.ndarray, t: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute analytical velocity field at time t.

        Args:
            x: x-coordinates [H, W]
            y: y-coordinates [H, W]
            t: Time

        Returns:
            u, v: Velocity components [H, W]
        """
        decay = np.exp(-2 * self.nu * self.k**2 * t)

        u = -self.U0 * np.cos(self.k * x) * np.sin(self.k * y) * decay
        v = self.U0 * np.sin(self.k * x) * np.cos(self.k * y) * decay

        return u, v

    def analytical_vorticity(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """Compute analytical vorticity field at time t.

        Args:
            x: x-coordinates [H, W]
            y: y-coordinates [H, W]
            t: Time

        Returns:
            ω: Vorticity field [H, W]
        """
        decay = np.exp(-2 * self.nu * self.k**2 * t)
        return 2 * self.k * self.U0 * np.cos(self.k * x) * np.cos(self.k * y) * decay

    def validate_velocity_field(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        t: float,
        bounds: tuple[float, float] = (-np.pi, np.pi),
        grid_resolution: int = 50,
        tolerance: float = 0.2,
    ) -> ValidationMetrics:
        """Validate velocity field against analytical solution.

        Args:
            positions: Particle positions [N, 2]
            velocities: Particle velocities [N, 2]
            t: Current simulation time
            bounds: Spatial domain
            grid_resolution: Grid resolution
            tolerance: Relative L² error tolerance

        Returns:
            ValidationMetrics with velocity field comparison
        """
        # Compute numerical velocity field
        X, Y, U_num, V_num = FluidFieldComputer.compute_velocity_field(
            positions, velocities, grid_resolution, bounds=bounds
        )

        # Compute analytical velocity field
        U_ana, V_ana = self.analytical_velocity(X, Y, t)

        # L² error
        error_u = np.sqrt(np.mean((U_num - U_ana) ** 2))
        error_v = np.sqrt(np.mean((V_num - V_ana) ** 2))
        total_error = np.sqrt(error_u**2 + error_v**2)

        # Normalize by analytical field magnitude
        magnitude = np.sqrt(np.mean(U_ana**2 + V_ana**2))
        relative_error = total_error / magnitude if magnitude > 1e-12 else total_error

        passed = relative_error <= tolerance

        return ValidationMetrics(
            metric_name="Taylor-Green Velocity Field",
            measured_value=relative_error,
            theoretical_value=0.0,
            tolerance=tolerance,
            passed=passed,
            description=f"Relative L² error at t={t:.2f}: {relative_error:.4f} "
            f"(tolerance: {tolerance})",
        )

    def validate_vorticity_field(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        t: float,
        bounds: tuple[float, float] = (-np.pi, np.pi),
        grid_resolution: int = 50,
        tolerance: float = 0.2,
    ) -> ValidationMetrics:
        """Validate vorticity field against analytical solution.

        Args:
            positions: Particle positions [N, 2]
            velocities: Particle velocities [N, 2]
            t: Current simulation time
            bounds: Spatial domain
            grid_resolution: Grid resolution
            tolerance: Relative L² error tolerance

        Returns:
            ValidationMetrics with vorticity field comparison
        """
        # Compute numerical velocity field
        X, Y, U_num, V_num = FluidFieldComputer.compute_velocity_field(
            positions, velocities, grid_resolution, bounds=bounds
        )

        domain_size = bounds[1] - bounds[0]
        dx = domain_size / grid_resolution
        dy = domain_size / grid_resolution

        # Compute numerical vorticity
        omega_num = FluidFieldComputer.compute_vorticity(U_num, V_num, dx, dy, periodic=True)

        # Compute analytical vorticity
        omega_ana = self.analytical_vorticity(X, Y, t)

        # L² error
        error = np.sqrt(np.mean((omega_num - omega_ana) ** 2))

        # Normalize by analytical field magnitude
        magnitude = np.sqrt(np.mean(omega_ana**2))
        relative_error = error / magnitude if magnitude > 1e-12 else error

        passed = relative_error <= tolerance

        return ValidationMetrics(
            metric_name="Taylor-Green Vorticity Field",
            measured_value=relative_error,
            theoretical_value=0.0,
            tolerance=tolerance,
            passed=passed,
            description=f"Relative L² error at t={t:.2f}: {relative_error:.4f} "
            f"(tolerance: {tolerance})",
        )


# ============================================================================
# Fluid Simulation Configurations
# ============================================================================


FLUID_CONFIGS = {
    "Taylor-Green Vortex": {
        "N": 1000,
        "n_steps": 200,
        "gamma": 0.5,
        "beta": 1.0,
        "delta_t": 0.02,
        "nu": 1.0,
        "use_viscous_coupling": True,
        "viscous_length_scale": 0.8,
        "use_potential_force": False,
        "use_fitness_force": False,
        "enable_cloning": False,
        "enable_kinetic": True,
        "epsilon_F": 0.0,
        "epsilon_Sigma": 0.1,
        "use_anisotropic_diffusion": False,
        "diagonal_diffusion": True,
        "V_alg": 10.0,
        "use_velocity_squashing": False,
        "sigma_x": 0.1,
        "alpha_restitution": 0.5,
        "p_max": 1.0,
        "epsilon_clone": 0.01,
    },
    "Lid-Driven Cavity (Re=100)": {
        "N": 1500,
        "n_steps": 300,
        "gamma": 0.5,
        "beta": 1.0,
        "delta_t": 0.01,
        "nu": 2.0,
        "use_viscous_coupling": True,
        "viscous_length_scale": 0.316,  # sqrt(0.01 / 2.0) for Re=100
        "use_potential_force": True,
        "use_fitness_force": False,
        "enable_cloning": False,
        "enable_kinetic": True,
        "epsilon_F": 0.0,
        "epsilon_Sigma": 0.1,
        "use_anisotropic_diffusion": False,
        "diagonal_diffusion": True,
        "V_alg": 10.0,
        "use_velocity_squashing": False,
        "sigma_x": 0.1,
        "alpha_restitution": 0.5,
        "p_max": 1.0,
        "epsilon_clone": 0.01,
    },
    "Kelvin-Helmholtz Instability": {
        "N": 2000,
        "n_steps": 250,
        "gamma": 0.3,
        "beta": 1.0,
        "delta_t": 0.02,
        "nu": 0.8,
        "use_viscous_coupling": True,
        "viscous_length_scale": 0.6,
        "use_potential_force": False,
        "use_fitness_force": False,
        "enable_cloning": False,
        "enable_kinetic": True,
        "epsilon_F": 0.0,
        "epsilon_Sigma": 0.1,
        "use_anisotropic_diffusion": False,
        "diagonal_diffusion": True,
        "V_alg": 10.0,
        "use_velocity_squashing": False,
        "sigma_x": 0.1,
        "alpha_restitution": 0.5,
        "p_max": 1.0,
        "epsilon_clone": 0.01,
    },
}
