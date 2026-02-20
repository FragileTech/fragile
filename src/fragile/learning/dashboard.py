"""Panel-based interactive dashboard for TopoEncoder training inspection.

Serves an interactive web app to inspect reconstructions, latent spaces,
training curves, and compare models (TopoEncoder, StandardVQ, VanillaAE)
across saved checkpoints.

Usage:
    panel serve src/fragile/learning/dashboard.py --show
"""

from __future__ import annotations

from dataclasses import dataclass, field
import glob
import os
import re
import traceback

import holoviews as hv
import numpy as np
import panel as pn
import torch

from fragile.core.benchmarks import BaselineClassifier, StandardVQ, VanillaAE
from fragile.core.layers import TopoEncoderPrimitives
from fragile.core.layers.topology import InvariantChartClassifier
from fragile.datasets import get_mnist_data
from fragile.learning.checkpoints import count_parameters, load_benchmarks, load_checkpoint
from fragile.learning.config import TopoEncoderConfig
from fragile.learning.plots import (
    _to_numpy,
    build_latent_scatter,
    chart_to_label_map,
    plot_chart_usage,
    plot_classifier_accuracy,
    plot_info_metrics,
    plot_latent_2d_slices,
    plot_latent_3d,
    plot_loss_components,
    plot_loss_curves,
    plot_reconstruction_grid,
)


os.environ.setdefault("PLOTLY_RENDERER", "json")
hv.extension("bokeh")

__all__ = ["create_app", "load_run_at_epoch", "scan_runs"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunInfo:
    """Describes a discovered training run."""

    name: str
    path: str
    epochs: list[int] = field(default_factory=list)
    has_benchmarks: bool = False


@dataclass
class LoadedModels:
    """Loaded checkpoint data ready for inference."""

    config: TopoEncoderConfig
    model_atlas: TopoEncoderPrimitives
    model_std: StandardVQ | None
    model_ae: VanillaAE | None
    classifier_head: InvariantChartClassifier | None
    classifier_std: BaselineClassifier | None
    classifier_ae: BaselineClassifier | None
    X_test: torch.Tensor
    labels_test: np.ndarray
    metrics: dict
    epoch: int


# ---------------------------------------------------------------------------
# Scanning & loading
# ---------------------------------------------------------------------------

_EPOCH_RE = re.compile(r"epoch_(\d+)\.pt$")

# Keys from TopoEncoderPrimitives.__init__ that can be forwarded from config dict
_ATLAS_INIT_KEYS = {
    "input_dim",
    "hidden_dim",
    "latent_dim",
    "num_charts",
    "codes_per_chart",
    "bundle_size",
    "covariant_attn",
    "covariant_attn_tensorization",
    "covariant_attn_rank",
    "covariant_attn_tau_min",
    "covariant_attn_denom_min",
    "covariant_attn_use_transport",
    "covariant_attn_transport_eps",
    "vision_preproc",
    "vision_in_channels",
    "vision_height",
    "vision_width",
    "vision_num_rotations",
    "vision_kernel_size",
    "vision_use_reflections",
    "vision_norm_nonlinearity",
    "vision_norm_bias",
    "vision_backbone_type",
    "vision_cifar_base_channels",
    "vision_cifar_bundle_size",
    "soft_equiv_metric",
    "soft_equiv_bundle_size",
    "soft_equiv_hidden_dim",
    "soft_equiv_use_spectral_norm",
    "soft_equiv_zero_self_mixing",
    "soft_equiv_soft_assign",
    "soft_equiv_temperature",
}


def scan_runs(outputs_dir: str = "outputs") -> list[RunInfo]:
    """Scan outputs directory for topoencoder training runs."""
    runs: list[RunInfo] = []
    pattern = os.path.join(outputs_dir, "topoencoder*")
    for run_dir in sorted(glob.glob(pattern)):
        if not os.path.isdir(run_dir):
            continue
        name = os.path.basename(run_dir)
        ckpt_dir = os.path.join(run_dir, "topoencoder")
        if not os.path.isdir(ckpt_dir):
            continue
        epochs = []
        for fname in sorted(os.listdir(ckpt_dir)):
            m = _EPOCH_RE.match(fname)
            if m:
                epochs.append(int(m.group(1)))
        if not epochs:
            continue
        has_bench = os.path.isfile(os.path.join(run_dir, "benchmarks.pt"))
        runs.append(
            RunInfo(name=name, path=run_dir, epochs=sorted(epochs), has_benchmarks=has_bench)
        )
    return runs


def _build_baseline_kwargs(cfg: dict, hidden_dim: int, is_vq: bool) -> dict:
    """Build kwargs for StandardVQ or VanillaAE from config dict."""
    kw: dict = {
        "input_dim": cfg.get("input_dim", 784),
        "hidden_dim": hidden_dim,
        "latent_dim": cfg.get("latent_dim", 2),
        "use_attention": cfg.get("baseline_attn", False),
        "attn_tokens": cfg.get("baseline_attn_tokens", 4),
        "attn_dim": cfg.get("baseline_attn_dim", 32),
        "attn_heads": cfg.get("baseline_attn_heads", 4),
        "attn_dropout": cfg.get("baseline_attn_dropout", 0.0),
        "vision_preproc": cfg.get("baseline_vision_preproc", False),
        "vision_in_channels": cfg.get("vision_in_channels", 0),
        "vision_height": cfg.get("vision_height", 0),
        "vision_width": cfg.get("vision_width", 0),
    }
    if is_vq:
        kw["num_codes"] = cfg.get("num_codes_standard", 64)
    return kw


def load_run_at_epoch(run: RunInfo, epoch: int) -> LoadedModels:
    """Load checkpoint + benchmarks, instantiate all 3 models in eval mode on CPU."""
    ckpt_path = os.path.join(run.path, "topoencoder", f"epoch_{epoch:05d}.pt")
    ckpt = load_checkpoint(ckpt_path)

    cfg_dict = ckpt.get("config", {})
    # Filter config through dataclass fields for forward-compat safety
    valid_fields = {f.name for f in TopoEncoderConfig.__dataclass_fields__.values()}
    safe_cfg = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    config = TopoEncoderConfig(**safe_cfg)

    metrics = ckpt.get("metrics", {})
    state = ckpt.get("state", {})

    # Data snapshot
    data = ckpt.get("data", {})
    X_test = data.get("X_test", torch.zeros(1, config.input_dim))
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = data.get("labels_test", np.zeros(len(X_test), dtype=int))
    if isinstance(labels_test, torch.Tensor):
        labels_test = labels_test.numpy()
    labels_test = np.asarray(labels_test)

    # --- Atlas model ---
    atlas_kwargs = {k: cfg_dict[k] for k in _ATLAS_INIT_KEYS if k in cfg_dict}
    model_atlas = TopoEncoderPrimitives(**atlas_kwargs)
    if state.get("atlas") is not None:
        model_atlas.load_state_dict(state["atlas"])
    model_atlas.eval()

    # --- Benchmark models ---
    bench: dict = {}
    if run.has_benchmarks:
        bench = load_benchmarks(os.path.join(run.path, "benchmarks.pt"))
    bench_dims = bench.get("dims", {})
    bench_state = bench.get("state", {})

    # StandardVQ
    model_std = None
    std_hidden_dim = bench_dims.get("std_hidden_dim", 0)
    std_state = state.get("std") or bench_state.get("std")
    if std_state is not None and std_hidden_dim > 0 and not config.disable_vq:
        kw = _build_baseline_kwargs(cfg_dict, std_hidden_dim, is_vq=True)
        model_std = StandardVQ(**kw)
        model_std.load_state_dict(std_state)
        model_std.eval()

    # VanillaAE
    model_ae = None
    ae_hidden_dim = bench_dims.get("ae_hidden_dim", 0)
    ae_state = state.get("ae") or bench_state.get("ae")
    if ae_state is not None and ae_hidden_dim > 0 and not config.disable_ae:
        kw = _build_baseline_kwargs(cfg_dict, ae_hidden_dim, is_vq=False)
        model_ae = VanillaAE(**kw)
        model_ae.load_state_dict(ae_state)
        model_ae.eval()

    # --- Classifier heads ---
    num_classes = config.num_classes
    classifier_head = None
    cls_state = state.get("classifier")
    if cls_state is not None:
        classifier_head = InvariantChartClassifier(
            num_charts=config.num_charts,
            num_classes=num_classes,
            latent_dim=config.latent_dim,
            bundle_size=config.classifier_bundle_size or None,
        )
        classifier_head.load_state_dict(cls_state)
        classifier_head.eval()

    classifier_std = None
    cls_std_state = state.get("classifier_std")
    if cls_std_state is not None:
        classifier_std = BaselineClassifier(config.latent_dim, num_classes)
        classifier_std.load_state_dict(cls_std_state)
        classifier_std.eval()

    classifier_ae = None
    cls_ae_state = state.get("classifier_ae")
    if cls_ae_state is not None:
        classifier_ae = BaselineClassifier(config.latent_dim, num_classes)
        classifier_ae.load_state_dict(cls_ae_state)
        classifier_ae.eval()

    return LoadedModels(
        config=config,
        model_atlas=model_atlas,
        model_std=model_std,
        model_ae=model_ae,
        classifier_head=classifier_head,
        classifier_std=classifier_std,
        classifier_ae=classifier_ae,
        X_test=X_test,
        labels_test=labels_test,
        metrics=metrics,
        epoch=epoch,
    )


def load_full_dataset(config: TopoEncoderConfig) -> tuple[torch.Tensor, np.ndarray]:
    """Load the full training dataset (e.g. 60k MNIST) for inference."""
    if config.dataset == "mnist":
        X, labels, _colors = get_mnist_data(n_samples=60000)
    elif config.dataset == "cifar10":
        from fragile.datasets import get_cifar10_data

        X, labels, _colors = get_cifar10_data(n_samples=50000)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    return X, labels


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def create_app(outputs_dir: str = "outputs") -> pn.template.FastListTemplate:
    """Create the interactive TopoEncoder inspection dashboard."""
    pn.extension("plotly", "tabulator")

    # ---- Shared state ----
    # "loaded": LoadedModels, "latent_cache": dict with full-dataset inference
    app_state: dict = {"runs": [], "loaded": None, "latent_cache": None}

    # ---- Sidebar widgets ----
    scan_btn = pn.widgets.Button(name="Scan runs", button_type="primary", width=300)
    run_selector = pn.widgets.Select(name="Run", options=[], width=300)
    epoch_slider = pn.widgets.DiscreteSlider(name="Epoch", options=[0], value=0, width=300)
    load_btn = pn.widgets.Button(name="Load checkpoint", button_type="success", width=300)
    use_full_dataset = pn.widgets.Checkbox(name="Load full dataset (60k)", value=False, width=300)
    n_samples = pn.widgets.IntSlider(
        name="Recon samples", start=4, end=24, value=8, step=4, width=300
    )
    latent_samples = pn.widgets.IntSlider(
        name="Latent samples",
        start=100,
        end=60000,
        value=1000,
        step=100,
        width=300,
    )
    color_by = pn.widgets.RadioButtonGroup(
        name="Color by",
        options=["label", "chart", "correct"],
        value="label",
        button_type="default",
    )
    point_size = pn.widgets.IntSlider(name="Point size", start=1, end=10, value=3, width=300)
    seed_input = pn.widgets.IntInput(name="Random seed", value=42, step=1, width=300)
    status = pn.pane.Markdown("Click **Scan runs** to begin.", width=300)

    sidebar = pn.Column(
        pn.pane.Markdown("## TopoEncoder Dashboard"),
        scan_btn,
        run_selector,
        epoch_slider,
        load_btn,
        use_full_dataset,
        pn.layout.Divider(),
        pn.pane.Markdown("### Display options"),
        n_samples,
        latent_samples,
        color_by,
        point_size,
        seed_input,
        pn.layout.Divider(),
        status,
        width=350,
    )

    # ---- Main panes (placeholders) ----
    loss_pane = pn.pane.HoloViews(hv.Curve([], "Epoch", "Loss"), sizing_mode="stretch_width")
    components_pane = pn.pane.HoloViews(
        hv.Curve([], "Epoch", "Value"), sizing_mode="stretch_width"
    )
    info_pane = pn.pane.HoloViews(hv.Curve([], "Epoch", "Value"), sizing_mode="stretch_width")
    accuracy_pane = pn.pane.HoloViews(
        hv.Curve([], "Epoch", "Accuracy"), sizing_mode="stretch_width"
    )

    recon_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width")
    metrics_table = pn.pane.Markdown("*Load a checkpoint to see metrics.*")

    latent_3d_pane = pn.pane.Plotly(None, sizing_mode="stretch_width", height=600)
    latent_2d_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width")
    usage_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width")

    # ---- Click-to-inspect panes ----
    inspect_label = pn.pane.Markdown("*Click a point in z0 vs z1 to inspect*")
    inspect_original = pn.pane.HoloViews(hv.Div(""), width=250, height=250)
    inspect_recon_atlas = pn.pane.HoloViews(hv.Div(""), width=250, height=250)
    inspect_recon_vq = pn.pane.HoloViews(hv.Div(""), width=250, height=250)
    inspect_recon_ae = pn.pane.HoloViews(hv.Div(""), width=250, height=250)
    inspect_row = pn.Row(
        pn.Column("**Original**", inspect_original),
        pn.Column("**TopoEncoder**", inspect_recon_atlas),
        pn.Column("**VQ**", inspect_recon_vq),
        pn.Column("**AE**", inspect_recon_ae),
    )
    tap_stream = hv.streams.Tap(x=0, y=0)

    # ---- Tabs ----
    training_tab = pn.Column(
        loss_pane, components_pane, info_pane, accuracy_pane, sizing_mode="stretch_width"
    )
    recon_tab = pn.Column(recon_pane, metrics_table, sizing_mode="stretch_width")
    latent_tab = pn.Column(
        latent_3d_pane,
        latent_2d_pane,
        inspect_label,
        inspect_row,
        usage_pane,
        sizing_mode="stretch_width",
    )

    tabs = pn.Tabs(
        ("Training Curves", training_tab),
        ("Reconstructions", recon_tab),
        ("Latent Space", latent_tab),
        sizing_mode="stretch_both",
    )

    # ---- Callbacks ----
    def _on_scan(_event=None):
        runs = scan_runs(outputs_dir)
        app_state["runs"] = runs
        if not runs:
            run_selector.options = []
            status.object = "No runs found."
            return
        run_selector.options = {r.name: r.name for r in runs}
        run_selector.value = runs[0].name
        _on_run_selected(None)
        status.object = f"Found **{len(runs)}** run(s)."

    def _on_run_selected(_event):
        name = run_selector.value
        run = next((r for r in app_state["runs"] if r.name == name), None)
        if run is None:
            return
        epoch_slider.options = run.epochs
        epoch_slider.value = run.epochs[-1] if run.epochs else 0

    def _on_load(_event=None):
        name = run_selector.value
        run = next((r for r in app_state["runs"] if r.name == name), None)
        if run is None:
            status.object = "**Error:** Select a run first."
            return
        epoch = epoch_slider.value
        status.object = f"Loading **{name}** epoch {epoch}..."
        try:
            loaded = load_run_at_epoch(run, epoch)
        except Exception as exc:
            status.object = f"**Error loading:** {exc}"
            traceback.print_exc()
            return
        app_state["loaded"] = loaded

        # Optionally load the full dataset instead of just the checkpoint test split
        if use_full_dataset.value:
            status.object = f"Loading full {loaded.config.dataset} dataset..."
            try:
                X_full, labels_full = load_full_dataset(loaded.config)
                app_state["X_infer"] = X_full
                app_state["labels_infer"] = labels_full
            except Exception as exc:
                status.object = f"**Error loading dataset:** {exc}"
                traceback.print_exc()
                app_state["X_infer"] = loaded.X_test
                app_state["labels_infer"] = loaded.labels_test
        else:
            app_state["X_infer"] = loaded.X_test
            app_state["labels_infer"] = loaded.labels_test

        n_total = len(app_state["X_infer"])
        latent_samples.end = n_total
        latent_samples.value = min(latent_samples.value, n_total)

        # Run inference on the selected data and cache
        status.object = f"Running inference on {n_total:,} samples..."
        _run_inference(loaded)
        _update_training_curves(loaded)
        _update_reconstructions(loaded)
        _refresh_latent_display()
        status.object = f"Loaded **{name}** epoch {epoch} " f"({n_total:,} samples)."

    def _run_inference(loaded: LoadedModels):
        """Run atlas model on current inference data and cache results."""
        X = app_state["X_infer"].float()
        labels = app_state["labels_infer"]

        # Batched inference to avoid OOM on large datasets
        batch_size = 2048
        z_geo_parts, K_chart_parts, router_parts = [], [], []
        cls_pred_parts = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                out = loaded.model_atlas(X[i : i + batch_size])
                z_geo_parts.append(_to_numpy(out[5]))
                K_chart_parts.append(_to_numpy(out[4]))
                router_parts.append(_to_numpy(out[2]))
                # Classifier head predictions
                if loaded.classifier_head is not None:
                    logits = loaded.classifier_head(out[2], out[5])
                    cls_pred_parts.append(_to_numpy(logits.argmax(dim=1)))

        z_geo = np.concatenate(z_geo_parts)
        K_chart = np.concatenate(K_chart_parts)
        router_weights = np.concatenate(router_parts)

        # Correctness via classifier head (preferred) or majority-vote fallback
        if cls_pred_parts:
            predicted = np.concatenate(cls_pred_parts)
        else:
            c2l = chart_to_label_map(K_chart, labels)
            predicted = c2l[K_chart]
        correct = (predicted == labels).astype(int)

        app_state["latent_cache"] = {
            "z_geo": z_geo,
            "K_chart": K_chart,
            "labels": labels,
            "correct": correct,
            "router_weights": router_weights,
        }

    def _refresh_latent_display(_event=None):
        """Re-render latent plots from cache using current display settings."""
        cache = app_state.get("latent_cache")
        if cache is None:
            return

        z_geo_full = cache["z_geo"]
        K_chart_full = cache["K_chart"]
        labels_full = cache["labels"]
        correct_full = cache["correct"]
        router_full = cache["router_weights"]

        n_total = len(z_geo_full)
        ns = min(latent_samples.value, n_total)

        # Subsample
        if ns < n_total:
            rng = np.random.default_rng(seed_input.value)
            idx = rng.choice(n_total, size=ns, replace=False)
            idx.sort()
            z_geo = z_geo_full[idx]
            K_chart = K_chart_full[idx]
            labels = labels_full[idx]
            correct = correct_full[idx]
        else:
            idx = np.arange(n_total)
            z_geo = z_geo_full
            K_chart = K_chart_full
            labels = labels_full
            correct = correct_full

        # Store for click-to-inspect lookup
        app_state["display_indices"] = idx
        app_state["display_z"] = z_geo

        cb = color_by.value
        ps = point_size.value

        latent_3d_pane.object = plot_latent_3d(
            z_geo,
            labels,
            K_chart=K_chart,
            correct=correct,
            color_by=cb,
            point_size=ps,
        )

        # Build the z0-z1 scatter separately to wire the Tap stream
        labs = _to_numpy(labels).astype(int)
        charts = _to_numpy(K_chart).astype(int)
        corr = _to_numpy(correct).astype(int)
        z_np = _to_numpy(z_geo)
        scatter_z01 = build_latent_scatter(
            z_np,
            labs,
            charts,
            corr,
            cb,
            ps,
            0,
            1,
            indices=idx,
        )
        tap_stream.source = scatter_z01

        # Build remaining pairs via plot_latent_2d_slices, then replace the first panel
        layout = plot_latent_2d_slices(
            z_geo,
            labels,
            K_chart=K_chart,
            correct=correct,
            color_by=cb,
            point_size=ps,
            indices=idx,
        )
        # Replace the first panel with the Tap-wired scatter
        panels = list(layout)
        if panels:
            panels[0] = scatter_z01
        latent_2d_pane.object = hv.Layout(panels).cols(min(3, len(panels)))

        # Chart usage from full router weights
        usage = router_full.sum(axis=0)
        usage = usage / usage.sum() if usage.sum() > 0 else usage
        usage_pane.object = plot_chart_usage(usage)

    def _update_training_curves(loaded: LoadedModels):
        m = loaded.metrics
        atlas_losses = np.asarray(m.get("atlas_losses", []))
        std_losses = np.asarray(m.get("std_losses", []))
        ae_losses = np.asarray(m.get("ae_losses", []))
        n_epochs = max(len(atlas_losses), 1)
        epochs = np.arange(n_epochs)

        loss_pane.object = plot_loss_curves(
            epochs,
            atlas_losses if len(atlas_losses) else np.zeros(1),
            std_losses if len(std_losses) else None,
            ae_losses if len(ae_losses) else None,
        )

        lc = m.get("loss_components", {})
        lc_np = {k: np.asarray(v) for k, v in lc.items()}
        components_pane.object = plot_loss_components(epochs, lc_np)

        im = m.get("info_metrics", {})
        im_np = {k: np.asarray(v) for k, v in im.items()}
        info_pane.object = plot_info_metrics(epochs, im_np)

        accuracy_pane.object = plot_classifier_accuracy(epochs, lc_np)

    def _update_reconstructions(loaded: LoadedModels):
        rng = np.random.default_rng(seed_input.value)
        ns = n_samples.value
        n_total = len(loaded.X_test)
        indices = rng.choice(n_total, size=min(ns, n_total), replace=False)
        indices.sort()

        X_sub = loaded.X_test[indices].float()
        originals = _to_numpy(X_sub)

        with torch.no_grad():
            out_atlas = loaded.model_atlas(X_sub)
            recon_atlas = _to_numpy(out_atlas[0])

            recon_vq = None
            vq_mse = None
            vq_params = None
            if loaded.model_std is not None:
                out_vq = loaded.model_std(X_sub)
                recon_vq = _to_numpy(out_vq[0])
                vq_mse = float(np.mean((originals - recon_vq) ** 2))
                vq_params = count_parameters(loaded.model_std)

            recon_ae = None
            ae_mse = None
            ae_params = None
            if loaded.model_ae is not None:
                out_ae = loaded.model_ae(X_sub)
                recon_ae = _to_numpy(out_ae[0])
                ae_mse = float(np.mean((originals - recon_ae) ** 2))
                ae_params = count_parameters(loaded.model_ae)

        atlas_mse = float(np.mean((originals - recon_atlas) ** 2))
        atlas_params = count_parameters(loaded.model_atlas)

        # Determine image shape
        cfg = loaded.config
        if cfg.vision_preproc and cfg.vision_height > 0 and cfg.vision_width > 0:
            image_shape = (cfg.vision_height, cfg.vision_width)
        else:
            side = int(np.sqrt(cfg.input_dim))
            image_shape = (side, side) if side * side == cfg.input_dim else (1, cfg.input_dim)

        recon_pane.object = plot_reconstruction_grid(
            originals, recon_atlas, recon_vq, recon_ae, ns, image_shape
        )

        # Metrics table
        lines = ["| Model | MSE | Params |", "|-------|-----|--------|"]
        lines.append(f"| TopoEncoder | {atlas_mse:.6f} | {atlas_params:,} |")
        if vq_mse is not None:
            lines.append(f"| StandardVQ | {vq_mse:.6f} | {vq_params:,} |")
        if ae_mse is not None:
            lines.append(f"| VanillaAE | {ae_mse:.6f} | {ae_params:,} |")
        metrics_table.object = "\n".join(lines)

    def _on_tap(_event=None):
        """Handle click on z0-z1 scatter: show original + reconstructed images."""
        x, y = tap_stream.x, tap_stream.y
        if app_state.get("display_z") is None or app_state.get("loaded") is None:
            return

        z = app_state["display_z"]
        dists = (z[:, 0] - x) ** 2 + (z[:, 1] - y) ** 2
        local_idx = int(np.argmin(dists))
        full_idx = int(app_state["display_indices"][local_idx])

        loaded = app_state["loaded"]
        X_infer = app_state["X_infer"]

        # Determine image shape
        cfg = loaded.config
        if cfg.vision_preproc and cfg.vision_height > 0 and cfg.vision_width > 0:
            image_shape = (cfg.vision_height, cfg.vision_width)
        else:
            side = int(np.sqrt(cfg.input_dim))
            image_shape = (side, side) if side * side == cfg.input_dim else (1, cfg.input_dim)

        def _img_to_hv(arr):
            img = arr.reshape(image_shape)
            return hv.Image(
                img,
                bounds=(0, 0, image_shape[1], image_shape[0]),
            ).opts(cmap="gray", xaxis=None, yaxis=None, width=220, height=220)

        # Original image
        original = _to_numpy(X_infer[full_idx])
        inspect_original.object = _img_to_hv(original)

        # Reconstructions
        x_in = X_infer[full_idx : full_idx + 1].float()
        with torch.no_grad():
            recon_atlas = _to_numpy(loaded.model_atlas(x_in)[0][0])
            inspect_recon_atlas.object = _img_to_hv(recon_atlas)

            if loaded.model_std is not None:
                recon_vq = _to_numpy(loaded.model_std(x_in)[0][0])
                inspect_recon_vq.object = _img_to_hv(recon_vq)
            else:
                inspect_recon_vq.object = hv.Div("")

            if loaded.model_ae is not None:
                recon_ae = _to_numpy(loaded.model_ae(x_in)[0][0])
                inspect_recon_ae.object = _img_to_hv(recon_ae)
            else:
                inspect_recon_ae.object = hv.Div("")

        # Update label with sample metadata
        cache = app_state.get("latent_cache", {})
        lbl = int(cache["labels"][full_idx]) if "labels" in cache else "?"
        chart = int(cache["K_chart"][full_idx]) if "K_chart" in cache else "?"
        corr = cache["correct"][full_idx] if "correct" in cache else "?"
        corr_str = "yes" if corr == 1 else "no"
        inspect_label.object = (
            f"**Sample #{full_idx}** — Label: {lbl}, Chart: {chart}, Correct: {corr_str}"
        )

    tap_stream.param.watch(_on_tap, ["x", "y"])

    # Wire up callbacks
    scan_btn.on_click(_on_scan)
    run_selector.param.watch(_on_run_selected, "value")
    load_btn.on_click(_on_load)

    # Latent display options: only re-render plots from cache (no re-inference)
    for w in [latent_samples, color_by, point_size]:
        w.param.watch(_refresh_latent_display, "value")

    # Reconstruction options: need to re-run inference on subset
    for w in [n_samples, seed_input]:
        w.param.watch(
            lambda _evt: _update_reconstructions(app_state["loaded"])
            if app_state.get("loaded")
            else None,
            "value",
        )

    template = pn.template.FastListTemplate(
        title="TopoEncoder Dashboard",
        sidebar=[sidebar],
        main=[tabs],
        accent_base_color="#4c78a8",
        header_background="#4c78a8",
    )

    # Auto-scan on load
    _on_scan()

    return template


# ---------------------------------------------------------------------------
# Serve entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__" or __name__.startswith("bokeh"):
    create_app().servable()
