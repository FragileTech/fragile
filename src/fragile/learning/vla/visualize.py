"""Poincaré disk plots and diagnostic visualizations for VLA experiments."""

from __future__ import annotations

import numpy as np
import torch

from fragile.learning.conformal import conformal_factor_np


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _pca_2d(z: np.ndarray) -> np.ndarray:
    """Project to 2D via PCA if latent_dim > 2."""
    if z.shape[1] <= 2:
        return z[:, :2]
    z_centered = z - z.mean(axis=0)
    _, _, Vt = np.linalg.svd(z_centered, full_matrices=False)
    return z_centered @ Vt[:2].T


def plot_poincare_disk(
    z_geo: torch.Tensor | np.ndarray,
    K_chart: torch.Tensor | np.ndarray,
    task_labels: torch.Tensor | np.ndarray | None = None,
    ax=None,
    title: str = "Poincaré Disk Embedding",
):
    """Scatter plot of latent embeddings on the unit disk.

    Colors by chart assignment; optionally marks tasks with markers.
    """
    import matplotlib.pyplot as plt

    z = _to_numpy(z_geo)
    K = _to_numpy(K_chart)
    z_2d = _pca_2d(z)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.figure

    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.8, alpha=0.3)

    scatter = ax.scatter(
        z_2d[:, 0], z_2d[:, 1],
        c=K, cmap="tab10", s=8, alpha=0.6,
    )
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="Chart")

    if task_labels is not None:
        tl = _to_numpy(task_labels)
        for label in np.unique(tl):
            mask = tl == label
            ax.scatter(
                z_2d[mask, 0], z_2d[mask, 1],
                marker=f"${int(label)}$", s=30, alpha=0.5, color="black",
            )

    return fig


def plot_radius_histogram(
    z_geo: torch.Tensor | np.ndarray,
    ax=None,
    title: str = "Radial Distribution",
):
    """Histogram of ||z|| norms."""
    import matplotlib.pyplot as plt

    z = _to_numpy(z_geo)
    r = np.linalg.norm(z, axis=1)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    else:
        fig = ax.figure

    ax.hist(r, bins=50, density=True, alpha=0.7, color="steelblue")
    ax.axvline(r.mean(), color="red", linestyle="--", label=f"mean={r.mean():.3f}")
    ax.set_xlabel("||z||")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    return fig


def plot_chart_task_alignment(
    K_chart: torch.Tensor | np.ndarray,
    task_labels: torch.Tensor | np.ndarray,
    ax=None,
    title: str = "Chart-Task Alignment",
):
    """Confusion heatmap and AMI score for chart vs. task alignment."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_mutual_info_score

    K = _to_numpy(K_chart).astype(int)
    T = _to_numpy(task_labels).astype(int)

    n_charts = K.max() + 1
    n_tasks = T.max() + 1
    confusion = np.zeros((n_charts, n_tasks), dtype=int)
    for k, t in zip(K, T):
        confusion[k, t] += 1

    ami = adjusted_mutual_info_score(K, T)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.imshow(confusion, aspect="auto", cmap="Blues")
    ax.set_xlabel("Task")
    ax.set_ylabel("Chart")
    ax.set_title(f"{title}  (AMI={ami:.3f})")
    fig.colorbar(im, ax=ax)

    return fig


def plot_chart_transitions(
    K_chart: torch.Tensor | np.ndarray,
    episode_ids: torch.Tensor | np.ndarray,
    ax=None,
    title: str = "Chart Transition Matrix",
):
    """Transition matrix of chart assignments across consecutive timesteps."""
    import matplotlib.pyplot as plt

    K = _to_numpy(K_chart).astype(int)
    ep = _to_numpy(episode_ids).astype(int)

    n_charts = K.max() + 1
    trans = np.zeros((n_charts, n_charts), dtype=int)

    for i in range(len(K) - 1):
        if ep[i] == ep[i + 1]:
            trans[K[i], K[i + 1]] += 1

    # Normalize rows
    row_sums = trans.sum(axis=1, keepdims=True).clip(min=1)
    trans_norm = trans / row_sums

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig = ax.figure

    im = ax.imshow(trans_norm, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("To Chart")
    ax.set_ylabel("From Chart")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    return fig


def plot_dynamics_trajectory(
    z_pred: torch.Tensor | np.ndarray,
    z_target: torch.Tensor | np.ndarray,
    ax=None,
    title: str = "Dynamics: Predicted vs Target",
):
    """Overlay predicted and target trajectories on the Poincaré disk."""
    import matplotlib.pyplot as plt

    zp = _pca_2d(_to_numpy(z_pred))
    zt = _pca_2d(_to_numpy(z_target))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.figure

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.8, alpha=0.3)

    ax.plot(zt[:, 0], zt[:, 1], "b-o", markersize=4, alpha=0.7, label="Target")
    ax.plot(zp[:, 0], zp[:, 1], "r--s", markersize=4, alpha=0.7, label="Predicted")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend()

    return fig


def full_diagnostic(
    encoder: torch.nn.Module,
    features: torch.Tensor,
    config,
    task_labels: torch.Tensor | np.ndarray | None = None,
    save_dir: str | None = None,
) -> dict[str, float]:
    """Run all diagnostics on a batch of features.

    Args:
        encoder: Trained TopoEncoderPrimitives.
        features: [N, D] feature tensor.
        config: VLAConfig.
        task_labels: Optional [N] task labels for alignment analysis.
        save_dir: If provided, save plots to this directory.

    Returns:
        Dict of diagnostic metrics.
    """
    import matplotlib.pyplot as plt

    device = next(encoder.parameters()).device
    encoder.eval()

    with torch.no_grad():
        (
            x_recon, vq_loss, enc_rw, dec_rw,
            K_chart, z_geo, z_n, c_bar, aux,
        ) = encoder(features.to(device))

    z_np = z_geo.cpu().numpy()
    K_np = K_chart.cpu().numpy()

    metrics: dict[str, float] = {}

    # Active charts
    unique_charts = len(np.unique(K_np))
    metrics["active_charts"] = unique_charts
    print(f"Active charts: {unique_charts}/{config.num_charts}")

    # Radial stats
    r = np.linalg.norm(z_np, axis=1)
    metrics["mean_radius"] = float(r.mean())
    metrics["std_radius"] = float(r.std())
    print(f"Radius: mean={r.mean():.3f}, std={r.std():.3f}")

    # Reconstruction R²
    x_np = features.cpu().numpy()
    xr_np = x_recon.cpu().numpy()
    ss_res = ((x_np - xr_np) ** 2).sum()
    ss_tot = ((x_np - x_np.mean(axis=0)) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    metrics["recon_r2"] = float(r2)
    print(f"Reconstruction R²: {r2:.4f}")

    # Conformal factors
    cf = conformal_factor_np(z_np)
    metrics["mean_conformal"] = float(cf.mean())
    print(f"Conformal factor: mean={cf.mean():.2f}")

    # Closure ratio (fraction of variance explained by chart membership)
    from sklearn.metrics import adjusted_mutual_info_score

    if task_labels is not None:
        tl = _to_numpy(task_labels).astype(int)
        ami = adjusted_mutual_info_score(K_np, tl)
        metrics["ami"] = float(ami)
        print(f"AMI (chart vs task): {ami:.4f}")

    # Generate plots
    figs = []

    fig1 = plot_poincare_disk(z_np, K_np, task_labels, title="Poincaré Disk Embedding")
    figs.append(("poincare_disk", fig1))

    fig2 = plot_radius_histogram(z_np)
    figs.append(("radius_hist", fig2))

    if task_labels is not None:
        fig3 = plot_chart_task_alignment(K_np, _to_numpy(task_labels))
        figs.append(("chart_task", fig3))

    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs:
            fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"Plots saved to {save_dir}")
    else:
        plt.show()

    # Success criteria
    print("\n--- Success Criteria ---")
    print(f"  Active charts >= 3: {'PASS' if unique_charts >= 3 else 'FAIL'}")
    print(f"  Recon R² > 0.8:    {'PASS' if r2 > 0.8 else 'FAIL'}")
    if task_labels is not None and "ami" in metrics:
        print(f"  AMI > 0.3:         {'PASS' if metrics['ami'] > 0.3 else 'FAIL'}")

    return metrics
