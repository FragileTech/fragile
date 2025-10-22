"""Interactive 3D Manifold Visualization for Geometric Gas.

This module provides interactive visualization of the emergent Riemannian geometry
in the Geometric Gas framework. It renders the curved manifold induced by the
adaptive diffusion tensor Σ_reg(x,S) = (H + ε_Σ I)^{-1/2} in 3D.

Visualization Modes:
--------------------

**For d ≥ 3 (3D+ runs):**
    Uses the first three spatial dimensions directly to visualize walker
    trajectories, manifold surface, and principal curvature directions.

**For d = 2 (2D runs) - "Spacetime Bending" Mode:**
    Reveals how flat 2D exploration creates a curved manifold by embedding
    the 2D positions into 3D space, where the third dimension represents
    the emergent geometric curvature:

    - x, y: Original 2D walker positions
    - z: Curvature measure (trace of metric, or max eigenvalue)

    This visualization makes the "bending of spacetime" intuitive: regions
    of high fitness curvature appear as peaks/valleys in the 3D surface,
    showing how the metric g = H + ε_Σ I encodes landscape geometry.

The visualization shows:
1. Walker trajectories through 3D space over time
2. Emergent Riemannian metric g(x,S) = H + ε_Σ I as a surface
3. Principal curvature directions (eigenvectors of diffusion tensor)
4. Optional fitness landscape contours

Mathematical Framework:
    The adaptive diffusion tensor induces a Riemannian metric on state space:

    $$
    g(x,S) = H(x,S) + \\epsilon_\\Sigma I
    $$

    where H(x,S) is the Hessian of the fitness potential. The diffusion tensor is:

    $$
    D_{\\text{reg}}(x,S) = g(x,S)^{-1} = (H + \\epsilon_\\Sigma I)^{-1}
    $$

    Eigendecomposition reveals principal directions:
    - High eigenvalues → large diffusion → exploration
    - Low eigenvalues → small diffusion → exploitation

Reference:
    docs/source/2_geometric_gas/18_emergent_geometry.md
    - § 3.1: Adaptive Diffusion Tensor (Definition def-d-adaptive-diffusion)
    - § 3.2: Uniform Ellipticity (Theorem thm-uniform-ellipticity)
    - § 3.3: Lipschitz Continuity

Examples:
    >>> from fragile.core.euclidean_gas import EuclideanGas
    >>> from fragile.experiments.manifold import create_manifold_dashboard
    >>>
    >>> # Example 1: 3D run (standard mode)
    >>> gas_3d = EuclideanGas(N=50, d=3, use_anisotropic_diffusion=True, ...)
    >>> history_3d = gas_3d.run(n_steps=200)
    >>> explorer_3d, panel_3d = create_manifold_dashboard(history_3d)
    >>> panel_3d.show()
    >>>
    >>> # Example 2: 2D run (curvature embedding mode - shows "spacetime bending")
    >>> gas_2d = EuclideanGas(N=50, d=2, use_anisotropic_diffusion=True, ...)
    >>> history_2d = gas_2d.run(n_steps=200)
    >>> explorer_2d, panel_2d = create_manifold_dashboard(history_2d)
    >>> # Adjust embedding parameters to see curvature more clearly
    >>> explorer_2d.curvature_scale = 2.0
    >>> explorer_2d.embedding_mode = "first_eigenvalue"
    >>> panel_2d.show()
"""

from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
from scipy.spatial import Delaunay
import torch
from torch import Tensor

from fragile.core.fitness import FitnessOperator
from fragile.core.history import RunHistory


# HoloViews extension for 3D visualization
hv.extension("plotly")


__all__ = [
    "ManifoldExplorer",
    "compute_diffusion_tensor",
    "compute_principal_directions",
    "compute_riemannian_metric",
    "create_interactive_manifold_explorer",
    "create_manifold_dashboard",
    "extract_3d_positions",
]


# ============================================================================
# Geometry Computation Utilities
# ============================================================================


def extract_3d_positions(
    history: RunHistory,
    stage: str = "final",
    dims: tuple[int, int, int] = (0, 1, 2),
) -> np.ndarray:
    """Extract 3D positions from RunHistory for visualization.

    Args:
        history: RunHistory instance with trajectory data
        stage: Which state to extract ("before_clone", "after_clone", "final")
        dims: Which dimensions to extract (default: first 3)

    Returns:
        Array of shape [n_recorded, N, 3] with positions in selected dimensions

    Raises:
        ValueError: If requested dimensions exceed available dimensions
        ValueError: If stage is not recognized
    """
    if max(dims) >= history.d:
        msg = f"Requested dimensions {dims} exceed available dimensions (d={history.d})"
        raise ValueError(msg)

    if stage == "before_clone":
        x_full = history.x_before_clone
    elif stage == "after_clone":
        x_full = history.x_after_clone
    elif stage == "final":
        x_full = history.x_final
    else:
        msg = f"Unknown stage: {stage}. Must be 'before_clone', 'after_clone', or 'final'"
        raise ValueError(msg)

    # Extract selected dimensions
    return x_full[:, :, list(dims)].detach().cpu().numpy()


def compute_riemannian_metric(
    hessian: Tensor,
    epsilon_Sigma: float = 0.1,
) -> Tensor:
    """Compute emergent Riemannian metric g(x,S) = H(x,S) + ε_Σ I.

    The metric defines distances on the fitness landscape. Directions with high
    curvature (large eigenvalues) represent narrow fitness peaks (exploitation),
    while directions with low curvature (small eigenvalues) represent flat
    valleys (exploration).

    Reference: 18_emergent_geometry.md § 3.1, Definition def-d-adaptive-diffusion

    Args:
        hessian: Fitness Hessian tensor
            - If diagonal: [N, d] containing diagonal elements
            - If full: [N, d, d] containing full matrices
        epsilon_Sigma: Regularization parameter ensuring positive definiteness

    Returns:
        Riemannian metric tensor (same shape as hessian)
        - If diagonal input: [N, d] with g_ii = H_ii + ε_Σ
        - If full input: [N, d, d] with g = H + ε_Σ I
    """
    if hessian.dim() == 2:
        # Diagonal case: [N, d]
        return hessian + epsilon_Sigma
    if hessian.dim() == 3:
        # Full case: [N, d, d]
        _N, d, _ = hessian.shape
        identity = torch.eye(d, device=hessian.device, dtype=hessian.dtype)
        return hessian + epsilon_Sigma * identity.unsqueeze(0)
    msg = f"Hessian must be 2D (diagonal) or 3D (full), got shape {hessian.shape}"
    raise ValueError(msg)


def compute_diffusion_tensor(metric: Tensor) -> Tensor:
    """Compute diffusion tensor D_reg = g^{-1} = (H + ε_Σ I)^{-1}.

    The diffusion tensor determines the noise covariance in Langevin dynamics.
    High metric eigenvalues (curved directions) yield low diffusion (exploitation),
    while low metric eigenvalues (flat directions) yield high diffusion (exploration).

    Reference: 18_emergent_geometry.md § 3.1, equation (321)

    Args:
        metric: Riemannian metric tensor
            - If diagonal: [N, d] containing diagonal elements
            - If full: [N, d, d] containing symmetric positive definite matrices

    Returns:
        Diffusion tensor (same shape as metric)
        - If diagonal input: [N, d] with D_ii = 1 / g_ii
        - If full input: [N, d, d] with D = g^{-1}
    """
    if metric.dim() == 2:
        # Diagonal case: element-wise inverse
        return 1.0 / metric
    if metric.dim() == 3:
        # Full case: matrix inverse
        return torch.linalg.inv(metric)
    msg = f"Metric must be 2D (diagonal) or 3D (full), got shape {metric.shape}"
    raise ValueError(msg)


def compute_principal_directions(
    diffusion_tensor: Tensor,
    positions: Tensor,
    arrow_scale: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute principal curvature directions from diffusion tensor eigendecomposition.

    Performs eigendecomposition D = Q Λ Q^T where:
    - Eigenvectors (Q columns): Principal directions in tangent space
    - Eigenvalues (Λ diagonal): Diffusion magnitudes along each direction

    High eigenvalues indicate exploration directions (flat landscape),
    low eigenvalues indicate exploitation directions (curved landscape).

    Reference: 18_emergent_geometry.md § 3.2, Remark on geometric interpretation

    Args:
        diffusion_tensor: Diffusion tensor [N, d, d] or [N, d]
        positions: Walker positions [N, d] (used as arrow base points)
        arrow_scale: Scaling factor for arrow lengths

    Returns:
        arrow_starts: Arrow base points [N*d, 3] (repeated positions)
        arrow_ends: Arrow tip points [N*d, 3]
        arrow_magnitudes: Eigenvalue magnitudes [N*d] for coloring

    Note:
        For diagonal diffusion_tensor, eigenvectors are axis-aligned.
        Output is in 3D (first 3 dimensions) even if d > 3.
    """
    N, d = positions.shape
    d_viz = min(d, 3)  # Visualize only first 3 dimensions

    if diffusion_tensor.dim() == 2:
        # Diagonal case: eigenvectors are axis-aligned, eigenvalues are diagonal elements
        eigenvalues = diffusion_tensor[:, :d_viz].detach().cpu().numpy()  # [N, d_viz]

        # Create arrow data
        n_arrows = N * d_viz
        arrow_starts = np.zeros((n_arrows, 3))
        arrow_ends = np.zeros((n_arrows, 3))
        arrow_magnitudes = np.zeros(n_arrows)

        pos_3d = positions[:, :d_viz].detach().cpu().numpy()  # [N, d_viz]

        for i in range(N):
            for j in range(d_viz):
                idx = i * d_viz + j
                arrow_starts[idx, :d_viz] = pos_3d[i]

                # Arrow direction: axis-aligned (unit vector in dimension j)
                direction = np.zeros(d_viz)
                direction[j] = 1.0

                # Arrow length proportional to eigenvalue (diffusion magnitude)
                arrow_ends[idx, :d_viz] = pos_3d[i] + arrow_scale * eigenvalues[i, j] * direction
                arrow_magnitudes[idx] = eigenvalues[i, j]

    elif diffusion_tensor.dim() == 3:
        # Full case: compute eigendecomposition
        # Only use first d_viz dimensions for visualization
        D_viz = diffusion_tensor[:, :d_viz, :d_viz]  # [N, d_viz, d_viz]

        # Eigendecomposition
        eigenvalues_tensor, eigenvectors_tensor = torch.linalg.eigh(D_viz)
        eigenvalues = eigenvalues_tensor.detach().cpu().numpy()  # [N, d_viz]
        eigenvectors = eigenvectors_tensor.detach().cpu().numpy()  # [N, d_viz, d_viz]

        # Create arrow data
        n_arrows = N * d_viz
        arrow_starts = np.zeros((n_arrows, 3))
        arrow_ends = np.zeros((n_arrows, 3))
        arrow_magnitudes = np.zeros(n_arrows)

        pos_3d = positions[:, :d_viz].detach().cpu().numpy()  # [N, d_viz]

        for i in range(N):
            for j in range(d_viz):
                idx = i * d_viz + j
                arrow_starts[idx, :d_viz] = pos_3d[i]

                # Arrow direction: jth eigenvector
                direction = eigenvectors[i, :, j]  # [d_viz]

                # Arrow length proportional to eigenvalue
                arrow_ends[idx, :d_viz] = pos_3d[i] + arrow_scale * eigenvalues[i, j] * direction
                arrow_magnitudes[idx] = eigenvalues[i, j]

    else:
        msg = f"Diffusion tensor must be 2D or 3D, got shape {diffusion_tensor.shape}"
        raise ValueError(msg)

    return arrow_starts, arrow_ends, arrow_magnitudes


# ============================================================================
# Hessian Data Extraction
# ============================================================================


class HessianExtractor:
    """Extract Hessian data from RunHistory with fallback hierarchy.

    Implements the priority order:
    1. history.fitness_hessians_full (complete geometry)
    2. history.fitness_hessians_diag (diagonal approximation)
    3. Compute on-the-fly using FitnessOperator (slowest but always works)

    Reference: User requirement for flexible Hessian source handling
    """

    def __init__(
        self,
        history: RunHistory,
        fitness_op: FitnessOperator | None = None,
        verbose: bool = True,
    ):
        """Initialize Hessian extractor.

        Args:
            history: RunHistory instance
            fitness_op: FitnessOperator for on-the-fly computation (optional)
            verbose: Print information about Hessian source
        """
        self.history = history
        self.fitness_op = fitness_op
        self.verbose = verbose

        # Determine available Hessian sources
        self.has_full = history.fitness_hessians_full is not None
        self.has_diag = history.fitness_hessians_diag is not None
        self.can_compute = fitness_op is not None

        # Determine which source to use
        if self.has_full:
            self.source = "full"
            if verbose:
                print("Using full Hessian from RunHistory.fitness_hessians_full")
        elif self.has_diag:
            self.source = "diagonal"
            if verbose:
                print("Using diagonal Hessian from RunHistory.fitness_hessians_diag")
        elif self.can_compute:
            self.source = "computed"
            if verbose:
                print("Computing Hessian on-the-fly using FitnessOperator")
        else:
            msg = (
                "No Hessian data available in RunHistory and no FitnessOperator provided. "
                "Either record Hessian during simulation or provide fitness_op for computation."
            )
            raise ValueError(msg)

    def get_hessian(self, step_idx: int) -> Tensor:
        """Extract Hessian for given recorded timestep.

        Args:
            step_idx: Index in recorded arrays (0 to n_recorded-1)
                     Note: Info data starts at step 1, so use step_idx-1 for info arrays

        Returns:
            Hessian tensor:
            - If full: [N, d, d]
            - If diagonal: [N, d]

        Raises:
            IndexError: If step_idx out of bounds
        """
        if step_idx >= self.history.n_recorded:
            msg = f"step_idx {step_idx} exceeds n_recorded {self.history.n_recorded}"
            raise IndexError(msg)

        if step_idx == 0:
            # No info data at initial step - return zeros
            N, d = self.history.N, self.history.d
            if self.source == "full":
                return torch.zeros(N, d, d)
            return torch.zeros(N, d)

        # Adjust index for info arrays (start at step 1)
        info_idx = step_idx - 1

        if self.source == "full":
            return self.history.fitness_hessians_full[info_idx]

        if self.source == "diagonal":
            return self.history.fitness_hessians_diag[info_idx]

        if self.source == "computed":
            # Compute Hessian using fitness operator
            positions = self.history.x_before_clone[step_idx]
            velocities = self.history.v_before_clone[step_idx]

            # Get rewards from history if available, otherwise use placeholder
            if info_idx < len(self.history.rewards):
                rewards = self.history.rewards[info_idx]
            else:
                rewards = torch.zeros(self.history.N)

            alive_mask = self.history.alive_mask[info_idx]
            companions = self.history.companions_distance[info_idx]

            # Compute Hessian (full by default for best accuracy)
            return self.fitness_op.compute_hessian(
                positions=positions,
                velocities=velocities,
                rewards=rewards,
                alive=alive_mask,
                companions=companions,
                diagonal_only=False,  # Full Hessian for complete geometry
            )

        msg = f"Unknown Hessian source: {self.source}"
        raise RuntimeError(msg)


# ============================================================================
# Main Visualization Class
# ============================================================================


class ManifoldExplorer(param.Parameterized):
    """Interactive 3D manifold visualization for Geometric Gas.

    Provides real-time visualization of the emergent Riemannian geometry induced
    by adaptive diffusion in the Geometric Gas framework. Renders:
    - Walker trajectories in 3D (first 3 dimensions)
    - Emergent manifold surface (metric-induced curvature)
    - Principal curvature directions (diffusion eigenvectors)

    Similar to SwarmExplorer in interactive_euclidean_gas.py but specialized for
    3D geometry visualization using Plotly backend.
    """

    # Visualization toggles
    show_trajectories = param.Boolean(default=True, doc="Display walker trajectories over time")
    show_manifold = param.Boolean(default=True, doc="Display emergent Riemannian manifold surface")
    show_principal_directions = param.Boolean(
        default=True, doc="Display principal curvature direction arrows"
    )

    # Geometry parameters
    epsilon_Sigma = param.Number(  # noqa: N815 (matches mathematical notation ε_Σ)
        default=0.1, bounds=(0.01, 1.0), doc="Hessian regularization parameter ε_Σ"
    )
    arrow_scale = param.Number(
        default=0.5, bounds=(0.1, 2.0), doc="Eigenvector arrow length scale"
    )

    # Rendering options
    manifold_opacity = param.Number(default=0.3, bounds=(0.0, 1.0), doc="Manifold surface opacity")
    trajectory_color_metric = param.ObjectSelector(
        default="fitness",
        objects=["fitness", "reward", "constant"],
        doc="Color encoding for trajectories",
    )
    point_size = param.Number(default=5, bounds=(1, 20), doc="Walker point size")

    # Embedding mode for 2D runs
    embedding_mode = param.ObjectSelector(
        default="curvature_height",
        objects=["curvature_height", "metric_trace", "first_eigenvalue"],
        doc="For d=2: how to embed flat 2D into curved 3D surface",
    )
    curvature_scale = param.Number(
        default=1.0, bounds=(0.1, 5.0), doc="Vertical scaling for curvature embedding"
    )

    # Axis range controls
    fix_axes = param.Boolean(default=True, doc="Fix axis ranges to domain bounds")
    x_range = param.Range(default=(-6.0, 6.0), doc="X-axis range (if fix_axes=True)")
    y_range = param.Range(default=(-6.0, 6.0), doc="Y-axis range (if fix_axes=True)")
    z_range = param.Range(default=None, doc="Z-axis range (if fix_axes=True, None=auto)")

    def __init__(
        self,
        history: RunHistory,
        fitness_op: FitnessOperator | None = None,
        **params,
    ):
        """Initialize manifold explorer.

        Args:
            history: RunHistory instance with trajectory data
            fitness_op: FitnessOperator for on-the-fly Hessian computation (optional)
            **params: Additional parameters for param.Parameterized

        Raises:
            ValueError: If history has d < 2 (need at least 2 dimensions)
            ValueError: If no Hessian data available and no fitness_op provided

        Note:
            For d=2 (2D runs): Creates embedded 3D visualization where the third
            dimension represents metric curvature, revealing "spacetime bending"
            For d>=3: Uses first 3 spatial dimensions directly
        """
        super().__init__(**params)

        if history.d < 2:
            msg = f"Need at least 2 dimensions for manifold visualization, got d={history.d}"
            raise ValueError(msg)

        self.history = history
        self.hessian_extractor = HessianExtractor(history, fitness_op, verbose=True)

        # Determine visualization mode
        self.is_2d_embedded = history.d == 2

        if self.is_2d_embedded:
            # For 2D runs: we'll embed into 3D using curvature as height
            # Extract 2D positions, compute 3D embedding dynamically per frame
            self.x_2d = extract_3d_positions(history, stage="final", dims=(0, 1, 0))[:, :, :2]
            self.x_3d = None  # Will compute per frame with curvature embedding
            if params.get("verbose", True):
                print(
                    "2D Embedding Mode: Will visualize curvature as z-height (spacetime bending)"
                )
        else:
            # For d>=3: use first 3 dimensions directly
            self.x_3d = extract_3d_positions(history, stage="final", dims=(0, 1, 2))
            self.x_2d = None

        # Create Panel widgets
        self.time_player = pn.widgets.Player(
            name="Time",
            start=0,
            end=history.n_recorded - 1,
            value=0,
            step=1,
            interval=200,  # 200ms between frames
            loop_policy="loop",
        )
        self.time_player.sizing_mode = "stretch_width"
        self.time_player.param.watch(self._sync_stream, "value")

        # Create HoloViews stream
        self.frame_stream = hv.streams.Stream.define("Frame", frame=0)()
        self.dmap_3d = hv.DynamicMap(self._render_3d_plot, streams=[self.frame_stream])

        # Status pane
        self.status_pane = pn.pane.Markdown(self._build_status_text(), sizing_mode="stretch_width")

        # Watch parameter changes to refresh plot
        watch_params = [
            "show_trajectories",
            "show_manifold",
            "show_principal_directions",
            "epsilon_Sigma",
            "arrow_scale",
            "manifold_opacity",
            "trajectory_color_metric",
            "point_size",
            "fix_axes",
            "x_range",
            "y_range",
            "z_range",
        ]
        if self.is_2d_embedded:
            watch_params.extend(["embedding_mode", "curvature_scale"])

        self.param.watch(self._refresh_frame, watch_params)

    def _sync_stream(self, event):
        """Synchronize time player with HoloViews stream."""
        frame = int(np.clip(event.new, 0, self.history.n_recorded - 1))
        self.frame_stream.event(frame=frame)

    def _refresh_frame(self, *_):
        """Refresh current frame when parameters change."""
        self.frame_stream.event(frame=self.time_player.value)

    def _build_status_text(self) -> str:
        """Build status text with simulation info."""
        mode_text = "2D embedded in curved 3D" if self.is_2d_embedded else "3D spatial (dims 0-2)"
        return (
            f"**Manifold Visualization**: {self.history.n_steps} steps, "
            f"{self.history.N} walkers, {self.history.d}D\n\n"
            f"**Visualization mode**: {mode_text}\n\n"
            f"**Recorded**: {self.history.n_recorded} timesteps "
            f"(every {self.history.record_every} steps)\n\n"
            f"**Hessian source**: {self.hessian_extractor.source}"
        )

    def _compute_curvature_embedding(self, frame: int) -> np.ndarray:
        """Compute 3D embedding for 2D positions using metric curvature as height.

        This reveals how the emergent Riemannian geometry "bends" the flat 2D
        exploration space into a curved manifold in 3D.

        Args:
            frame: Current frame index

        Returns:
            Embedded positions [N, 3] where z = curvature measure

        Reference:
            The metric g = H + ε_Σ I encodes local geometry. We use the trace
            or dominant eigenvalue as a scalar curvature measure to lift 2D
            positions into 3D space, visualizing the "bending of spacetime".
        """
        if frame == 0:
            # No Hessian at initial frame - return flat embedding
            pos_2d = self.x_2d[frame]
            pos_3d = np.zeros((len(pos_2d), 3))
            pos_3d[:, :2] = pos_2d
            return pos_3d

        # Get Hessian and compute metric
        hessian = self.hessian_extractor.get_hessian(frame)
        metric = compute_riemannian_metric(hessian, self.epsilon_Sigma)

        # Get 2D positions
        pos_2d = self.x_2d[frame]

        # Compute curvature measure based on embedding mode
        if self.embedding_mode == "curvature_height":
            # Use trace of metric (sum of eigenvalues = total curvature)
            if metric.dim() == 2:
                # Diagonal: sum of diagonal elements
                curvature = metric.sum(dim=1).detach().cpu().numpy()
            else:
                # Full: trace of matrix
                curvature = (
                    torch.diagonal(metric, dim1=1, dim2=2).sum(dim=1).detach().cpu().numpy()
                )

        elif self.embedding_mode == "metric_trace":
            # Same as curvature_height (kept for clarity)
            if metric.dim() == 2:
                curvature = metric.sum(dim=1).detach().cpu().numpy()
            else:
                curvature = (
                    torch.diagonal(metric, dim1=1, dim2=2).sum(dim=1).detach().cpu().numpy()
                )

        elif self.embedding_mode == "first_eigenvalue":
            # Use largest eigenvalue (most curved direction)
            if metric.dim() == 2:
                # Diagonal: max of diagonal elements
                curvature = metric.max(dim=1)[0].detach().cpu().numpy()
            else:
                # Full: largest eigenvalue
                eigenvalues = torch.linalg.eigvalsh(metric)
                curvature = eigenvalues[:, -1].detach().cpu().numpy()

        # Normalize curvature for visualization
        curvature_normalized = (curvature - curvature.min()) / (
            curvature.max() - curvature.min() + 1e-8
        )

        # Build 3D embedding
        pos_3d = np.zeros((len(pos_2d), 3))
        pos_3d[:, :2] = pos_2d
        pos_3d[:, 2] = self.curvature_scale * curvature_normalized

        return pos_3d

    def _render_3d_plot(self, frame: int):
        """Render complete 3D visualization at given frame.

        Args:
            frame: Frame index (0 to n_recorded-1)

        Returns:
            HoloViews overlay with requested visualization layers
        """
        frame = int(np.clip(frame, 0, self.history.n_recorded - 1))

        # Collect requested visualization layers (can be empty if toggles disabled)
        layers = []

        # Layer 1: Walker trajectories
        if self.show_trajectories:
            traj_plot = self._render_trajectories(frame)
            if traj_plot is not None:
                layers.append(traj_plot)

        # Layer 3: Principal directions (before manifold for better visibility)
        if self.show_principal_directions:
            arrows_plot = self._render_principal_directions(frame)
            if arrows_plot is not None:
                layers.append(arrows_plot)

        # Layer 2: Manifold surface (rendered last for transparency)
        if self.show_manifold:
            manifold_plot = self._render_manifold(frame)
            if manifold_plot is not None:
                layers.append(manifold_plot)

        if not layers:
            # Provide placeholder element to keep Plotly backend happy when all layers are disabled
            layers.append(self._empty_placeholder())

        plot = hv.Overlay(layers)

        # Configure 3D plot options
        plot_opts = {
            "width": 800,
            "height": 700,
            "title": f"Emergent Geometry (frame {frame}/{self.history.n_recorded - 1}, "
            f"t={frame * self.history.record_every})",
        }

        # Add fixed axis ranges if enabled
        if self.fix_axes:
            plot_opts["xlim"] = self.x_range
            plot_opts["ylim"] = self.y_range
            if self.z_range is not None:
                plot_opts["zlim"] = self.z_range

        return plot.opts(**plot_opts)

    def _render_trajectories(self, frame: int):
        """Render walker trajectories up to current frame.

        Shows:
        - Current walker positions as 3D scatter points
        - Trajectory trails from frame 0 to current frame (optional)

        For 2D runs: Uses curvature embedding to show spacetime bending
        For 3D+ runs: Uses first 3 spatial dimensions directly

        Args:
            frame: Current frame index

        Returns:
            HoloViews Scatter3D element with walker positions
        """
        # Get positions at current frame (handle 2D embedding)
        if self.is_2d_embedded:
            positions = self._compute_curvature_embedding(frame)  # [N, 3] with z=curvature
        else:
            positions = self.x_3d[frame]  # [N, 3]

        alive_mask = (
            self.history.alive_mask[frame - 1]
            if frame > 0
            else np.ones(self.history.N, dtype=bool)
        )

        # Filter to alive walkers
        alive_positions = positions[alive_mask]

        if len(alive_positions) == 0:
            # No alive walkers - return empty scatter
            return hv.Scatter3D([], kdims=["x", "y", "z"], vdims=[])

        # Build DataFrame for scatter plot

        scatter_data = {
            "x": alive_positions[:, 0],
            "y": alive_positions[:, 1],
            "z": alive_positions[:, 2],
        }

        # Add color metric if not constant
        if self.trajectory_color_metric != "constant" and frame > 0:
            info_idx = frame - 1
            if self.trajectory_color_metric == "fitness":
                metric_values = self.history.fitness[info_idx].detach().cpu().numpy()[alive_mask]
                scatter_data["fitness"] = metric_values
                vdims = ["fitness"]
            elif self.trajectory_color_metric == "reward":
                metric_values = self.history.rewards[info_idx].detach().cpu().numpy()[alive_mask]
                scatter_data["reward"] = metric_values
                vdims = ["reward"]
            else:
                vdims = []
        else:
            vdims = []

        df = pd.DataFrame(scatter_data)

        # Create scatter plot
        scatter = hv.Scatter3D(df, kdims=["x", "y", "z"], vdims=vdims)

        # Configure scatter options
        if vdims:
            scatter = scatter.opts(
                size=self.point_size,
                color=vdims[0],
                cmap="Viridis",
                colorbar=True,
                alpha=0.8,
            )
        else:
            scatter = scatter.opts(
                size=self.point_size,
                color="navy",
                alpha=0.8,
            )

        return scatter

    def _empty_placeholder(self) -> hv.Scatter3D:
        """Return an invisible placeholder element to avoid empty overlays."""
        return hv.Scatter3D([], kdims=["x", "y", "z"], vdims=[]).opts(
            size=0,
            color="white",
            alpha=0.0,
            show_legend=False,
        )

    def _render_principal_directions(self, frame: int):
        """Render principal curvature direction arrows.

        Visualizes the principal directions of the diffusion tensor as 3D arrows
        emanating from walker positions. Arrow length is proportional to eigenvalue
        (diffusion magnitude), with colors indicating exploration (high eigenvalue)
        vs exploitation (low eigenvalue).

        Args:
            frame: Current frame index

        Returns:
            HoloViews element with arrow visualization, or None if frame=0
        """
        if frame == 0:
            # No Hessian data at initial step
            return None

        # Get Hessian for current frame
        hessian = self.hessian_extractor.get_hessian(frame)

        # Compute metric and diffusion tensor
        metric = compute_riemannian_metric(hessian, self.epsilon_Sigma)
        diffusion = compute_diffusion_tensor(metric)

        # Get positions for current frame (first 3 dimensions)
        positions_full = self.history.x_before_clone[frame]
        alive_mask = self.history.alive_mask[frame - 1]

        # Filter to alive walkers
        positions_alive = positions_full[alive_mask]

        if len(positions_alive) == 0:
            return None

        # Filter diffusion tensor to alive walkers
        if diffusion.dim() == 2:
            diffusion_alive = diffusion[alive_mask]
        else:
            diffusion_alive = diffusion[alive_mask]

        # Compute principal directions
        arrow_starts, arrow_ends, arrow_magnitudes = compute_principal_directions(
            diffusion_tensor=diffusion_alive,
            positions=positions_alive,
            arrow_scale=self.arrow_scale,
        )

        if len(arrow_starts) == 0:
            return None

        # Build arrow visualization using Plotly-style vectors
        # HoloViews doesn't have native 3D arrows, so we use Scatter3D + Path3D

        # Create paths for arrows (each arrow is a line segment)
        n_arrows = len(arrow_starts)
        arrow_paths = []

        for i in range(n_arrows):
            path = np.vstack([arrow_starts[i], arrow_ends[i]])
            arrow_paths.append(path)

        # For HoloViews Plotly, we can use Scatter3D with mode='lines'
        # or create explicit Path3D elements

        # Build DataFrame with all arrow segments
        arrow_data = []
        for i, path in enumerate(arrow_paths):
            # Add start and end points for this arrow
            arrow_data.extend(
                {
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                    "arrow_id": i,
                    "magnitude": arrow_magnitudes[i],
                }
                for point in path
            )

        df_arrows = pd.DataFrame(arrow_data)

        # Create Path3D for arrows
        paths = hv.Path3D(df_arrows, kdims=["x", "y", "z"], vdims=["magnitude", "arrow_id"])

        # Style arrows
        # Note: Plotly backend for Path3D only supports: color, dash, line_width, visible
        # Cannot use cmap or alpha with Path3D in Plotly
        return paths.opts(
            color="green",  # Fixed color (exploration direction)
            line_width=2,
        )

    def _render_manifold(self, frame: int):
        """Render emergent Riemannian manifold surface.

        Visualizes the Riemannian metric structure as a 3D surface. The metric
        g(x,S) = H + ε_Σ I encodes the fitness landscape geometry. We represent
        this by creating a mesh surface through walker positions, with surface
        height/color modulated by metric eigenvalues.

        Approach:
        - Use walker positions as mesh anchor points
        - Compute metric eigenvalues (curvature) at each position
        - Create triangulated surface (Delaunay triangulation)
        - Color/height based on mean eigenvalue (total curvature)

        Args:
            frame: Current frame index

        Returns:
            HoloViews TriSurface element, or None if insufficient points
        """
        if frame == 0:
            # No Hessian data at initial step
            return None

        # Get Hessian for current frame
        hessian = self.hessian_extractor.get_hessian(frame)

        # Compute metric
        metric = compute_riemannian_metric(hessian, self.epsilon_Sigma)

        # Get positions for current frame
        if self.is_2d_embedded:
            # Use curvature embedding for 2D runs
            positions_3d = self._compute_curvature_embedding(frame)
            alive_mask = self.history.alive_mask[frame - 1]
            positions_alive = positions_3d[alive_mask]
        else:
            # Use first 3 spatial dimensions
            positions_full = self.history.x_before_clone[frame]
            alive_mask = self.history.alive_mask[frame - 1]
            positions_alive = positions_full[alive_mask][:, :3]

        if len(positions_alive) < 4:
            # Need at least 4 points for triangulation
            return None

        # Compute metric eigenvalues for coloring
        if metric.dim() == 2:
            # Diagonal case: eigenvalues are diagonal elements
            metric_alive = metric[alive_mask][:, :3]  # First 3 dimensions
            eigenvalues_mean = metric_alive.mean(dim=1).detach().cpu().numpy()
        else:
            # Full case: compute eigenvalues of 3x3 submatrix
            metric_alive = metric[alive_mask][:, :3, :3]
            eigenvalues_tensor = torch.linalg.eigvalsh(metric_alive)
            eigenvalues_mean = eigenvalues_tensor.mean(dim=1).detach().cpu().numpy()

        # Convert positions to numpy
        if isinstance(positions_alive, np.ndarray):
            pos_np = positions_alive  # Already numpy (from 2D embedding)
        else:
            pos_np = positions_alive.detach().cpu().numpy()  # Convert from tensor

        # Build DataFrame for TriSurface

        # Perform Delaunay triangulation in 3D

        try:
            # Project to 2D for triangulation (use XY plane)
            points_2d = pos_np[:, :2]
            tri = Delaunay(points_2d)

            # Build triangle list
            triangles = []
            triangle_colors = []

            for simplex in tri.simplices:
                # Each simplex is a triangle with 3 vertex indices
                v0, v1, v2 = simplex

                # Add triangle vertices
                triangle = np.vstack([pos_np[v0], pos_np[v1], pos_np[v2]])

                # Triangle color: mean eigenvalue of the 3 vertices
                color_value = eigenvalues_mean[[v0, v1, v2]].mean()

                triangles.append(triangle)
                triangle_colors.append(color_value)

            # Build TriSurface data
            # HoloViews TriSurface expects: (x, y, z, color) or list of triangles
            trisurface_data = []

            for i, (triangle, color) in enumerate(zip(triangles, triangle_colors)):
                # Add each vertex of the triangle
                trisurface_data.extend(
                    {
                        "x": vertex[0],
                        "y": vertex[1],
                        "z": vertex[2],
                        "curvature": color,
                        "triangle_id": i,
                    }
                    for vertex in triangle
                )

            df_surf = pd.DataFrame(trisurface_data)

            # Create TriSurface
            surface = hv.TriSurface(
                df_surf, kdims=["x", "y", "z"], vdims=["curvature", "triangle_id"]
            )

            # Style surface
            # Note: Plotly backend for TriSurface uses different options than Bokeh
            # Style opts: cmap, edges_color, facecolor (no color, no alpha)
            # Plot opts: colorbar
            return surface.opts(
                cmap="Plasma",  # Purple (low curvature) to Yellow (high curvature)
                colorbar=True,
                # Note: manifold_opacity parameter not used due to Plotly limitations
            )

        except Exception as e:
            # Triangulation can fail for degenerate point configurations
            print(f"Warning: Manifold triangulation failed at frame {frame}: {e}")
            return None

    def panel(self) -> pn.Column:
        """Build Panel dashboard layout.

        Returns:
            Panel Column with controls and visualization
        """
        # Build parameter list (include embedding controls for 2D mode)
        viz_params = [
            "show_trajectories",
            "show_manifold",
            "show_principal_directions",
            "epsilon_Sigma",
            "arrow_scale",
            "manifold_opacity",
            "trajectory_color_metric",
            "point_size",
        ]

        axis_params = [
            "fix_axes",
            "x_range",
            "y_range",
            "z_range",
        ]

        if self.is_2d_embedded:
            # Add 2D embedding controls
            viz_params.extend(["embedding_mode", "curvature_scale"])

        # Build parameter panels
        viz_panel = pn.Param(
            self.param,
            parameters=viz_params,
            show_name=False,
            sizing_mode="stretch_width",
        )

        axis_panel = pn.Param(
            self.param,
            parameters=axis_params,
            show_name=False,
            sizing_mode="stretch_width",
        )

        controls = pn.Column(
            pn.pane.Markdown("### Visualization Controls"),
            viz_panel,
            pn.pane.Markdown("### Axis Ranges"),
            axis_panel,
            pn.pane.Markdown("### Playback"),
            self.time_player,
            self.status_pane,
            sizing_mode="stretch_width",
            width=350,
        )

        viz_pane = pn.panel(self.dmap_3d, sizing_mode="stretch_both")

        return pn.Row(
            controls,
            viz_pane,
            sizing_mode="stretch_both",
        )


# ============================================================================
# Factory Function
# ============================================================================


def create_manifold_dashboard(
    history: RunHistory,
    fitness_op: FitnessOperator | None = None,
    **explorer_params,
) -> tuple[ManifoldExplorer, pn.Row]:
    """Create manifold visualization dashboard.

    Factory function that creates a ManifoldExplorer and its Panel layout.

    Args:
        history: RunHistory instance with trajectory data
        fitness_op: FitnessOperator for on-the-fly Hessian computation (optional)
        **explorer_params: Additional parameters for ManifoldExplorer
            Common params:
            - x_range: tuple[float, float] - X-axis range (default: (-6, 6))
            - y_range: tuple[float, float] - Y-axis range (default: (-6, 6))
            - z_range: tuple[float, float] | None - Z-axis range (default: None=auto)
            - fix_axes: bool - Fix axis ranges to bounds (default: True)
            - epsilon_Sigma: float - Hessian regularization (default: 0.1)
            - show_trajectories: bool - Show walker trajectories (default: True)
            - show_manifold: bool - Show manifold surface (default: True)
            - show_principal_directions: bool - Show curvature arrows (default: True)

    Returns:
        explorer: ManifoldExplorer instance
        panel_layout: Panel Row ready to .show() or .servable()

    Example:
        >>> history = gas.run(n_steps=200)
        >>> explorer, panel = create_manifold_dashboard(history)
        >>> panel.show()  # Launch browser

        >>> # With custom bounds
        >>> explorer, panel = create_manifold_dashboard(
        ...     history, x_range=(-10, 10), y_range=(-10, 10), fix_axes=True
        ... )
        >>> panel.show()
    """
    explorer = ManifoldExplorer(history, fitness_op, **explorer_params)
    panel_layout = explorer.panel()
    return explorer, panel_layout


# ============================================================================
# Integrated Simulation + Visualization Dashboard
# ============================================================================


def create_interactive_manifold_explorer(
    potential: object,
    dims: int = 2,
    **config_params,
) -> pn.Column:
    """Create complete interactive workflow: configure → run → visualize manifold.

    This combines GasConfig (parameter configuration + simulation runner) with
    ManifoldExplorer (3D geometry visualization) in a single dashboard. User can:
    1. Configure simulation parameters interactively
    2. Run simulation with "Run Simulation" button
    3. Automatically see manifold visualization update with results

    Especially useful for 2D runs where the "spacetime bending" visualization
    reveals how flat 2D exploration creates curved manifolds.

    Args:
        potential: Potential function object with evaluate() method
        dims: Spatial dimension (2 for spacetime bending mode, ≥3 for direct 3D)
        **config_params: Override default GasConfig parameter values

    Returns:
        Panel Column with integrated configuration + visualization dashboard

    Example - 2D Spacetime Bending:
        >>> from fragile.experiments.convergence_analysis import create_multimodal_potential
        >>> potential, _ = create_multimodal_potential(dims=2, n_gaussians=3)
        >>> dashboard = create_interactive_manifold_explorer(potential, dims=2)
        >>> dashboard.show()
        >>> # User adjusts parameters in UI, clicks "Run Simulation"
        >>> # Manifold visualization automatically updates showing curvature embedding

    Example - 3D Direct Visualization:
        >>> potential, _ = create_multimodal_potential(dims=3, n_gaussians=5)
        >>> dashboard = create_interactive_manifold_explorer(
        ...     potential, dims=3, use_anisotropic_diffusion=True, epsilon_Sigma=0.2
        ... )
        >>> dashboard.show()

    Note:
        Requires `gas_config_dashboard` module for GasConfig class.
        The visualization automatically handles 2D vs 3D+ modes.
    """
    from fragile.experiments.gas_config_dashboard import GasConfig  # noqa: PLC0415

    # Create configuration dashboard
    config = GasConfig(potential=potential, dims=dims, **config_params)

    # Create placeholder for manifold explorer (will be populated after first run)
    manifold_pane = pn.pane.Markdown(
        "**Manifold Visualization**\n\n"
        "Configure parameters above and click 'Run Simulation' to see the emergent geometry.\n\n"
        f"**Mode**: {'2D Spacetime Bending' if dims == 2 else '3D Spatial Visualization'}\n\n"
        f"**Dimensions**: {dims}D",
        sizing_mode="stretch_both",
    )

    # Callback to update visualization when simulation completes
    def on_simulation_complete(history: RunHistory):
        """Update manifold visualization with new simulation results."""
        try:
            # Create FitnessOperator for on-the-fly Hessian computation if needed
            from fragile.core.companion_selection import CompanionSelection  # noqa: PLC0415

            companion_selection = CompanionSelection(
                method=config.companion_method,
                epsilon=config.companion_epsilon,
                lambda_alg=config.lambda_alg,
            )

            from fragile.core.fitness import FitnessParams  # noqa: PLC0415

            fitness_params = FitnessParams(
                alpha=config.alpha_fit,
                beta=config.beta_fit,
                eta=config.eta,
                lambda_alg=config.lambda_alg,
                sigma_min=config.sigma_min,
                A=config.A,
            )
            fitness_op = FitnessOperator(
                params=fitness_params,
                companion_selection=companion_selection,
            )

            # Create manifold explorer
            # Extract bounds from config for fixed axes
            bounds_extent = config.bounds_extent
            explorer = ManifoldExplorer(
                history,
                fitness_op=fitness_op,
                epsilon_Sigma=config.epsilon_Sigma,
                x_range=(-bounds_extent, bounds_extent),
                y_range=(-bounds_extent, bounds_extent),
            )

            # Update the manifold pane with new visualization
            manifold_pane.object = explorer.panel()

        except Exception as e:
            manifold_pane.object = pn.pane.Markdown(f"**Visualization Error**\n\n```\n{e!s}\n```")

    # Register callback
    config.add_completion_callback(on_simulation_complete)

    # Build integrated dashboard
    config_panel = config.panel()

    # Create two-column layout: config on left, visualization on right
    if dims == 2:
        instructions = pn.pane.Markdown(
            "### 2D Spacetime Bending Visualization\n\n"
            "This mode reveals how flat 2D exploration creates a curved manifold.\n"
            "The z-axis will show metric curvature, making the 'bending of spacetime' visible.\n\n"
            "**Try enabling:**\n"
            "- `use_anisotropic_diffusion`: Creates varied curvature\n"
            "- `use_fitness_force`: Walker dynamics follow geodesics\n"
            "- Adjust `epsilon_Sigma` to control curvature regularization",
            sizing_mode="stretch_width",
        )
    else:
        instructions = pn.pane.Markdown(
            "### 3D Manifold Visualization\n\n"
            "Visualizes walker trajectories, emergent manifold surface, and "
            "principal curvature directions in the first 3 spatial dimensions.\n\n"
            "**Recommended settings:**\n"
            "- Enable `use_anisotropic_diffusion` to see geometry effects\n"
            "- Higher `N` (walkers) for better surface triangulation",
            sizing_mode="stretch_width",
        )

    return pn.Column(
        pn.pane.Markdown("# Interactive Manifold Explorer"),
        instructions,
        pn.Row(
            config_panel,
            manifold_pane,
            sizing_mode="stretch_both",
        ),
        sizing_mode="stretch_both",
    )
