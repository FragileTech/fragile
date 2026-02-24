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
import logging
import os
import re
import traceback

import holoviews as hv
import numpy as np
import panel as pn
from sklearn.metrics import adjusted_mutual_info_score
import torch
import torch.nn.functional as F

from fragile.core.benchmarks import BaselineClassifier, StandardVQ, VanillaAE
from fragile.core.layers import TopoEncoderPrimitives as CoreTopoEncoderPrimitives
from fragile.learning.core.layers.atlas import (
    TopoEncoderPrimitives as LegacyTopoEncoderPrimitives,
)
from fragile.core.layers.topology import InvariantChartClassifier
from fragile.datasets import get_mnist_data
from fragile.learning.checkpoints import count_parameters, load_benchmarks, load_checkpoint
from fragile.learning.conformal import (
    ablation_feature_importance,
    accuracy_vs_radius,
    calibration_test_split,
    compute_tunneling_rate,
    conditional_coverage_by_radius,
    conformal_quantile,
    conformal_quantiles_per_chart,
    conformal_quantiles_per_chart_code,
    conformal_quantiles_per_class,
    conformal_quantiles_per_radius,
    conformal_scores_geo_beta,
    conformal_scores_geodesic,
    conformal_scores_standard,
    conformal_factor_np,
    hyperbolic_knn_density,
    geodesic_isolation,
    corrupt_data,
    coverage_by_class,
    coverage_under_corruption,
    evaluate_coverage,
    expected_calibration_error,
    format_ablation_table,
    format_class_coverage_table,
    format_coverage_method_comparison,
    format_coverage_summary_table,
    format_ood_auroc_table,
    forward_pass_batch,
    load_fashion_mnist,
    ood_scores,
    plot_accuracy_vs_radius,
    plot_conditional_coverage,
    plot_corruption_coverage,
    plot_ood_roc,
    plot_reliability_diagram,
    plot_set_size_vs_radius,
    prediction_sets,
    prediction_sets_mondrian,
    radial_bins,
    recalibrate_probs,
    reliability_diagram_data,
    router_entropy,
    tune_beta,
    tune_conformal_beta,
)
from fragile.learning.config import TopoEncoderConfig
from fragile.learning.plots import (
    FASHION_MNIST_CLASSES,
    MNIST_CLASSES,
    _to_numpy,
    build_latent_scatter,
    build_ood_scatter_2d,
    build_ood_trace_3d,
    chart_to_label_map,
    plot_chart_usage,
    plot_classifier_accuracy,
    plot_info_metrics,
    plot_latent_2d_slices,
    plot_latent_3d,
    plot_loss_components,
    plot_loss_curves,
    plot_prob_bar_grid,
    plot_prob_bars,
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
    model_atlas: CoreTopoEncoderPrimitives | LegacyTopoEncoderPrimitives
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
    "conv_backbone",
    "conv_channels",
    "img_channels",
    "img_size",
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


def _infer_atlas_kwargs_from_state(
    atlas_state: dict[str, torch.Tensor] | None,
    base_kwargs: dict,
) -> dict:
    """Infer missing atlas constructor kwargs from checkpoint tensor shapes.

    This is critical when loading checkpoints produced by different topology
    settings or hidden widths.
    """
    inferred: dict = {}
    if atlas_state is None:
        return dict(base_kwargs)

    # Strongest signal for dimensions comes from decoder residual matrix.
    tex_residual = atlas_state.get("decoder.tex_residual.weight")
    if isinstance(tex_residual, torch.Tensor) and tex_residual.ndim >= 2:
        inferred["input_dim"] = int(tex_residual.shape[0])
        inferred["latent_dim"] = int(tex_residual.shape[1])

    for key, value in atlas_state.items():
        if not isinstance(value, torch.Tensor):
            continue

        if value.ndim == 2 and (
            key.endswith("feature_extractor.0.weight")
            or key.endswith("shared_feature_extractor.0.weight")
        ):
            inferred.setdefault("hidden_dim", int(value.shape[0]))

        if value.ndim == 2 and key.endswith("latent_router.weight"):
            inferred.setdefault("num_charts", int(value.shape[0]))
            inferred.setdefault("latent_dim", int(value.shape[1]))

        if value.ndim in (3, 4) and "codebook" in key:
            inferred.setdefault("num_charts", int(value.shape[0]))
            inferred.setdefault("codes_per_chart", int(value.shape[1]))
            inferred.setdefault("latent_dim", int(value.shape[2]))

        if value.ndim >= 4 and "conv" in key and key.endswith(".weight"):
            inferred.setdefault("conv_backbone", True)
            inferred.setdefault("conv_channels", int(value.shape[0]))
            if value.shape[1] > 0:
                inferred.setdefault("img_channels", int(value.shape[1]))
            if value.ndim == 4 and value.shape[2] == value.shape[3] and value.shape[2] > 1:
                inferred.setdefault("img_size", int(value.shape[2]))

    merged = dict(base_kwargs)
    merged.update(inferred)
    return merged


def _infer_conv_checkpoint(
    atlas_state: dict[str, torch.Tensor],
    hidden_dim: int,
    input_dim: int,
) -> bool:
    """Infer whether this checkpoint was saved from a convolutional TopoEncoder."""
    del hidden_dim
    del input_dim
    for key in atlas_state:
        if "encoder.feature_extractor.conv" in key:
            return True
        if "encoder.shared_feature_extractor.conv" in key:
            return True
        if "decoder.renderer.deconv" in key:
            return True
        if "decoder.renderer." in key and key.endswith("deconv.weight"):
            return True
    return False


def _infer_conv_state_kwargs(
    cfg_dict: dict,
    atlas_state: dict[str, torch.Tensor],
    hidden_dim: int,
    input_dim: int,
) -> dict:
    """Infer conv-related constructor kwargs from checkpoint tensors."""
    inferred: dict = {}
    conv_key_candidates: dict[str, int] = {}

    for key, value in atlas_state.items():
        if not isinstance(value, torch.Tensor):
            continue
        if "encoder.feature_extractor.conv" in key and key.endswith(".weight") and value.ndim >= 4:
            if "conv_channels" not in inferred:
                inferred["conv_channels"] = int(value.shape[0])
            conv_idx = conv_key_candidates.get("first_conv", 0)
            if "conv1" in key and conv_idx == 0:
                conv_key_candidates["first_conv"] = int(value.shape[1])
                inferred["img_channels"] = int(value.shape[1])
        if key == "decoder.renderer.deconv.0.weight" and value.ndim >= 4:
            inferred.setdefault("conv_channels", int(value.shape[0]))

    if conv_key_candidates.get("first_conv", 0) == 0:
        conv_key_candidates["first_conv"] = int(cfg_dict.get("img_channels", 1))
        inferred.setdefault("img_channels", conv_key_candidates["first_conv"])
    else:
        inferred.setdefault("img_channels", conv_key_candidates["first_conv"])

    img_size = int(cfg_dict.get("img_size", 0) or 0)
    if img_size <= 0:
        if input_dim > 0:
            img_channels = inferred.get("img_channels", cfg_dict.get("img_channels", 1))
            if img_channels > 0 and input_dim % img_channels == 0:
                side = int(np.sqrt(input_dim // img_channels))
                if side * side == input_dim // img_channels:
                    img_size = side
    if img_size <= 0:
        img_size = 28
    inferred.setdefault("img_size", img_size)
    inferred.setdefault("conv_channels", int(cfg_dict.get("conv_channels", hidden_dim)))
    inferred.setdefault("img_channels", int(cfg_dict.get("img_channels", 1)))

    return inferred


def _state_uses_legacy_layers(atlas_state: dict[str, torch.Tensor] | None) -> bool | None:
    """Infer whether checkpoint keys match the legacy encoder layout."""
    if atlas_state is None:
        return None
    for key in atlas_state:
        if key.startswith("encoder.shared_feature_extractor"):
            return False
        if key.startswith("encoder.feature_extractor"):
            return True
    return None


def _build_atlas_model(atlas_kwargs: dict, atlas_cls: type) -> object:
    """Instantiate TopoEncoderPrimitives with compatibility fallback for old args."""
    while True:
        try:
            return atlas_cls(**atlas_kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" in msg:
                bad_key = msg.split("'")[1]
                atlas_kwargs.pop(bad_key, None)
                continue
            if "required positional argument" in msg:
                raise
            raise


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
    atlas_state = state.get("atlas")
    atlas_kwargs = _infer_atlas_kwargs_from_state(atlas_state, atlas_kwargs)
    hidden_dim = int(atlas_kwargs.get("hidden_dim", config.hidden_dim))
    input_dim = int(atlas_kwargs.get("input_dim", config.input_dim))
    candidate_models: list[tuple[type, dict]] = []
    logger = logging.getLogger(__name__)
    if atlas_state is not None:
        is_conv = _infer_conv_checkpoint(atlas_state, hidden_dim=hidden_dim, input_dim=input_dim)
        legacy_layout = _state_uses_legacy_layers(atlas_state)

        conv_state_kwargs = _infer_conv_state_kwargs(
            cfg_dict=cfg_dict,
            atlas_state=atlas_state,
            hidden_dim=hidden_dim,
            input_dim=input_dim,
        )

        legacy_kwargs = dict(atlas_kwargs)
        if is_conv:
            legacy_kwargs["conv_backbone"] = True
            legacy_kwargs.update(conv_state_kwargs)
        else:
            legacy_kwargs.pop("conv_backbone", None)
            legacy_kwargs.pop("conv_channels", None)
            legacy_kwargs.pop("img_channels", None)
            legacy_kwargs.pop("img_size", None)

        legacy_kwargs_fallback = dict(legacy_kwargs)
        if is_conv:
            legacy_kwargs_fallback.pop("conv_backbone", None)
            legacy_kwargs_fallback.pop("conv_channels", None)
            legacy_kwargs_fallback.pop("img_channels", None)
            legacy_kwargs_fallback.pop("img_size", None)
        else:
            legacy_kwargs_fallback["conv_backbone"] = True
            legacy_kwargs_fallback.update(conv_state_kwargs)

        core_kwargs = dict(atlas_kwargs)
        core_kwargs.pop("conv_backbone", None)
        core_kwargs.pop("conv_channels", None)
        core_kwargs.pop("img_channels", None)
        core_kwargs.pop("img_size", None)

        preferred_legacy = True if legacy_layout is not False else False
        if legacy_layout is None:
            preferred_legacy = is_conv

        if preferred_legacy:
            candidate_models.append((LegacyTopoEncoderPrimitives, legacy_kwargs))
            candidate_models.append((LegacyTopoEncoderPrimitives, legacy_kwargs_fallback))
            candidate_models.append((CoreTopoEncoderPrimitives, core_kwargs))
        else:
            candidate_models.append((CoreTopoEncoderPrimitives, core_kwargs))
            candidate_models.append((LegacyTopoEncoderPrimitives, legacy_kwargs))
            candidate_models.append((LegacyTopoEncoderPrimitives, legacy_kwargs_fallback))
    else:
        candidate_models = [(CoreTopoEncoderPrimitives, atlas_kwargs)]

    model_atlas = None
    result = None
    atlas_load_error = None
    seen: set[tuple[type, tuple[tuple[str, object], ...]]] = set()
    for model_cls, attempt in candidate_models:
        attempt = dict(attempt)
        attempt_key = (model_cls, tuple(sorted(attempt.items())))
        if attempt_key in seen:
            continue
        seen.add(attempt_key)

        model_candidate = _build_atlas_model(attempt, model_cls)
        if atlas_state is None:
            model_atlas = model_candidate
            break
        try:
            result = model_candidate.load_state_dict(atlas_state, strict=False)
            model_atlas = model_candidate
            atlas_kwargs = attempt
            break
        except RuntimeError as exc:
            logger.debug("Atlas state load failed for candidate config: %s", exc)
            model_atlas = None
            result = None
            atlas_load_error = exc

    if model_atlas is None:
        if atlas_state is not None:
            raise RuntimeError(
                "Unable to load atlas state with available architecture candidates"
            ) from atlas_load_error
        model_atlas = _build_atlas_model(atlas_kwargs, CoreTopoEncoderPrimitives)
    if result is not None and (result.missing_keys or result.unexpected_keys):
        logger.warning(
            "Atlas state_dict partial load: %d missing, %d unexpected keys "
            "(architecture change between checkpoint and current code)",
            len(result.missing_keys),
            len(result.unexpected_keys),
        )
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
        cls_std_hidden = cls_std_state["net.0.weight"].shape[0]
        classifier_std = BaselineClassifier(config.latent_dim, num_classes, hidden_dim=cls_std_hidden)
        classifier_std.load_state_dict(cls_std_state)
        classifier_std.eval()

    classifier_ae = None
    cls_ae_state = state.get("classifier_ae")
    if cls_ae_state is not None:
        cls_ae_hidden = cls_ae_state["net.0.weight"].shape[0]
        classifier_ae = BaselineClassifier(config.latent_dim, num_classes, hidden_dim=cls_ae_hidden)
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


def _dataset_key(config: TopoEncoderConfig) -> str:
    """Normalize dataset name, stripping any trailing notes after '|'."""
    return config.dataset.split("|")[0].strip().lower()


def load_full_dataset(config: TopoEncoderConfig) -> tuple[torch.Tensor, np.ndarray]:
    """Load the full training dataset (e.g. 60k MNIST) for inference."""
    key = _dataset_key(config)
    if key == "mnist":
        X, labels, _colors = get_mnist_data(n_samples=60000)
    elif key == "fashion_mnist":
        from fragile.datasets import get_fashion_mnist_data

        X, labels, _colors = get_fashion_mnist_data(n_samples=60000)
    elif key == "cifar10":
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
        options=["label", "chart", "correct", "confidence"],
        value="label",
        button_type="default",
    )
    point_size = pn.widgets.IntSlider(name="Point size", start=1, end=10, value=3, width=300)
    alpha_by_confidence = pn.widgets.Checkbox(
        name="Alpha by confidence", value=False, width=300,
    )
    seed_input = pn.widgets.IntInput(name="Random seed", value=42, step=1, width=300)
    show_hierarchy = pn.widgets.Checkbox(name="Show hierarchy tree", value=False, width=300)
    tree_line_color = pn.widgets.Select(
        name="Line color", options=["black", "chart", "symbol"], value="black", width=300,
    )
    tree_line_width = pn.widgets.EditableFloatSlider(
        name="Line width", start=0.1, end=5.0, value=0.5, step=0.1, width=300,
    )
    ood_toggle = pn.widgets.Checkbox(name="Show OOD samples", value=False, width=300)
    ood_n_samples = pn.widgets.EditableIntSlider(
        name="OOD samples", start=1, end=1000, value=200, step=1, width=300,
    )
    status = pn.pane.Markdown("Click **Scan runs** to begin.", width=300)

    # ---- Conformal analysis sidebar widgets ----
    conformal_alpha = pn.widgets.FloatSlider(
        name="Miscoverage alpha", start=0.01, end=0.50, value=0.10, step=0.01, width=300,
    )
    conformal_bins = pn.widgets.IntSlider(
        name="Radial bins", start=5, end=30, value=15, width=300,
    )
    conformal_cal_frac = pn.widgets.FloatSlider(
        name="Calibration fraction", start=0.20, end=0.80, value=0.50, step=0.05, width=300,
    )
    conformal_n_corrupt = pn.widgets.IntSlider(
        name="Corruption intensities", start=3, end=10, value=5, width=300,
    )
    conformal_min_samples = pn.widgets.IntSlider(
        name="Min group size (n_min)", start=5, end=100, value=30, step=5, width=300,
    )
    run_conformal_btn = pn.widgets.Button(
        name="Run Conformal Analysis", button_type="warning", width=300,
    )
    conformal_status = pn.pane.Markdown("Load checkpoint first.", width=300)

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
        alpha_by_confidence,
        seed_input,
        pn.layout.Divider(),
        pn.pane.Markdown("### Hierarchy"),
        show_hierarchy,
        tree_line_color,
        tree_line_width,
        pn.layout.Divider(),
        pn.pane.Markdown("### OOD Overlay"),
        ood_toggle,
        ood_n_samples,
        pn.layout.Divider(),
        pn.pane.Markdown("### Conformal Analysis"),
        conformal_alpha,
        conformal_bins,
        conformal_cal_frac,
        conformal_n_corrupt,
        conformal_min_samples,
        run_conformal_btn,
        conformal_status,
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
    recon_prob_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width")
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

    # Probability bar chart panes (per-class distribution on click)
    inspect_prob_atlas = pn.pane.HoloViews(hv.Div(""), width=250, height=200)
    inspect_prob_vq = pn.pane.HoloViews(hv.Div(""), width=250, height=200)
    inspect_prob_ae = pn.pane.HoloViews(hv.Div(""), width=250, height=200)
    inspect_prob_row = pn.Row(
        pn.Column(inspect_prob_atlas),
        pn.Column(inspect_prob_vq),
        pn.Column(inspect_prob_ae),
    )

    tap_stream = hv.streams.Tap(x=0, y=0)

    # ---- Conformal analysis panes ----
    acc_vs_radius_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width")
    reliability_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width")
    ablation_table = pn.pane.Markdown("")
    coverage_summary_table = pn.pane.Markdown("")
    conditional_coverage_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width")
    set_size_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width")
    class_coverage_table = pn.pane.Markdown("")
    coverage_comparison_table = pn.pane.Markdown("")
    ood_roc_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width")
    ood_auroc_table = pn.pane.Markdown("")
    corruption_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width")

    # ---- Tabs ----
    training_tab = pn.Column(
        loss_pane, components_pane, info_pane, accuracy_pane, sizing_mode="stretch_width"
    )
    recon_tab = pn.Column(recon_pane, recon_prob_pane, metrics_table, sizing_mode="stretch_width")
    latent_tab = pn.Column(
        latent_3d_pane,
        latent_2d_pane,
        inspect_label,
        inspect_row,
        inspect_prob_row,
        usage_pane,
        sizing_mode="stretch_width",
    )
    conformal_tab = pn.Column(
        pn.pane.Markdown("## Geometric Calibration"),
        pn.Row(acc_vs_radius_pane, reliability_pane),
        ablation_table,
        pn.layout.Divider(),
        pn.pane.Markdown("## Conformal Coverage"),
        coverage_summary_table,
        pn.Row(conditional_coverage_pane, set_size_pane),
        class_coverage_table,
        coverage_comparison_table,
        pn.layout.Divider(),
        pn.pane.Markdown("## Distribution Shift"),
        pn.Row(ood_roc_pane, ood_auroc_table),
        corruption_pane,
        sizing_mode="stretch_width",
    )

    tabs = pn.Tabs(
        ("Training Curves", training_tab),
        ("Reconstructions", recon_tab),
        ("Latent Space", latent_tab),
        ("Conformal Analysis", conformal_tab),
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
        app_state.pop("ood_cache", None)

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
        z_geo_parts, K_chart_parts, K_code_parts, router_parts = [], [], [], []
        cls_pred_parts, cls_prob_parts = [], []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                enc_out = loaded.model_atlas.encoder(X[i : i + batch_size])
                z_geo_parts.append(_to_numpy(enc_out[5]))
                K_chart_parts.append(_to_numpy(enc_out[0]))
                K_code_parts.append(_to_numpy(enc_out[1]))
                router_parts.append(_to_numpy(enc_out[4]))
                # Classifier head predictions + probabilities
                if loaded.classifier_head is not None:
                    logits = loaded.classifier_head(enc_out[4], enc_out[5])
                    probs = F.softmax(logits, dim=1)
                    cls_pred_parts.append(_to_numpy(logits.argmax(dim=1)))
                    cls_prob_parts.append(_to_numpy(probs))

        z_geo = np.concatenate(z_geo_parts)
        K_chart = np.concatenate(K_chart_parts)
        K_code = np.concatenate(K_code_parts)
        router_weights = np.concatenate(router_parts)

        # Correctness via classifier head (preferred) or majority-vote fallback
        if cls_pred_parts:
            predicted = np.concatenate(cls_pred_parts)
        else:
            c2l = chart_to_label_map(K_chart, labels)
            predicted = c2l[K_chart]
        correct = (predicted == labels).astype(int)

        # Confidence = max softmax probability per sample
        if cls_prob_parts:
            probs_all = np.concatenate(cls_prob_parts)
            confidence = probs_all.max(axis=1)
        else:
            confidence = np.ones(len(labels))

        app_state["latent_cache"] = {
            "z_geo": z_geo,
            "K_chart": K_chart,
            "K_code": K_code,
            "labels": labels,
            "correct": correct,
            "router_weights": router_weights,
            "confidence": confidence,
            "softmax_probs": probs_all if cls_prob_parts else None,
        }

    def _compute_ood_latent(loaded: LoadedModels):
        """Compute OOD latent embeddings, caching by n_samples.

        Returns (z_ood, ood_labels, ood_dataset_name, class_names) or
        (None, None, None, None) for unsupported datasets.
        """
        n = ood_n_samples.value
        cache = app_state.get("ood_cache")
        if cache is not None and cache.get("n") == n:
            return cache["z_ood"], cache["labels"], cache["name"], cache["class_names"]

        ds_key = _dataset_key(loaded.config)
        if ds_key == "mnist":
            from fragile.datasets import get_fashion_mnist_data
            X_ood, ood_labels, _ = get_fashion_mnist_data(n_samples=n)
            ood_name = "Fashion-MNIST"
            class_names = FASHION_MNIST_CLASSES
        elif ds_key == "fashion_mnist":
            X_ood, ood_labels, _ = get_mnist_data(n_samples=n)
            ood_name = "MNIST"
            class_names = MNIST_CLASSES
        else:
            return None, None, None, None

        if not isinstance(X_ood, torch.Tensor):
            X_ood = torch.tensor(
                X_ood if isinstance(X_ood, np.ndarray) else X_ood.numpy(),
                dtype=torch.float32,
            )
        z_ood, _, _, _, _ = forward_pass_batch(loaded, X_ood)
        ood_labels = np.asarray(ood_labels)

        app_state["ood_cache"] = {
            "n": n, "z_ood": z_ood, "labels": ood_labels,
            "name": ood_name, "class_names": class_names,
        }
        return z_ood, ood_labels, ood_name, class_names

    def _refresh_latent_display(_event=None):
        """Re-render latent plots from cache using current display settings."""
        cache = app_state.get("latent_cache")
        if cache is None:
            return

        z_geo_full = cache["z_geo"]
        K_chart_full = cache["K_chart"]
        K_code_full = cache["K_code"]
        labels_full = cache["labels"]
        correct_full = cache["correct"]
        router_full = cache["router_weights"]
        confidence_full = cache["confidence"]

        n_total = len(z_geo_full)
        ns = min(latent_samples.value, n_total)

        # Subsample
        if ns < n_total:
            rng = np.random.default_rng(seed_input.value)
            idx = rng.choice(n_total, size=ns, replace=False)
            idx.sort()
            z_geo = z_geo_full[idx]
            K_chart = K_chart_full[idx]
            K_code = K_code_full[idx]
            labels = labels_full[idx]
            correct = correct_full[idx]
            conf = confidence_full[idx]
        else:
            idx = np.arange(n_total)
            z_geo = z_geo_full
            K_chart = K_chart_full
            K_code = K_code_full
            labels = labels_full
            correct = correct_full
            conf = confidence_full

        # Filter points outside [-1, 1] in the first 3 latent dims
        ndim = min(z_geo.shape[1], 3)
        in_range = np.all((z_geo[:, :ndim] >= -1) & (z_geo[:, :ndim] <= 1), axis=1)
        if not np.all(in_range):
            z_geo = z_geo[in_range]
            K_chart = K_chart[in_range]
            K_code = K_code[in_range]
            labels = labels[in_range]
            correct = correct[in_range]
            conf = conf[in_range]
            idx = idx[in_range]

        # Store for click-to-inspect lookup
        app_state["display_indices"] = idx
        app_state["display_z"] = z_geo

        cb = color_by.value
        ps = point_size.value
        abc = alpha_by_confidence.value

        latent_3d_pane.object = plot_latent_3d(
            z_geo,
            labels,
            K_chart=K_chart,
            correct=correct,
            color_by=cb,
            point_size=ps,
            K_code=K_code,
            show_hierarchy=show_hierarchy.value,
            tree_line_color=tree_line_color.value,
            tree_line_width=tree_line_width.value,
            confidence=conf,
            alpha_by_confidence=abc,
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
            confidence=conf,
            alpha_by_confidence=abc,
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
            confidence=conf,
            alpha_by_confidence=abc,
        )
        # Replace the first panel with the Tap-wired scatter
        panels = list(layout)
        if panels:
            panels[0] = scatter_z01

        # OOD overlay
        if ood_toggle.value:
            loaded = app_state.get("loaded")
            if loaded is not None:
                z_ood, ood_labels, ood_name, class_names = _compute_ood_latent(loaded)
                if z_ood is not None:
                    # Clip OOD to [-1, 1] to match ID filtering
                    ndim_ood = min(z_ood.shape[1], 3)
                    ood_in_range = np.all(
                        (z_ood[:, :ndim_ood] >= -1) & (z_ood[:, :ndim_ood] <= 1), axis=1,
                    )
                    z_ood_f = z_ood[ood_in_range]
                    ood_labels_f = ood_labels[ood_in_range]

                    # Determine dimension pairs (same logic as plot_latent_2d_slices)
                    dim = z_geo.shape[1]
                    pairs = []
                    if dim >= 2:
                        pairs.append((0, 1))
                    if dim >= 3:
                        pairs.append((0, 2))
                        pairs.append((1, 2))

                    # 2D: overlay on each panel
                    panels = [
                        p * build_ood_scatter_2d(z_ood_f, ood_labels_f, class_names, di, dj, ps)
                        for p, (di, dj) in zip(panels, pairs)
                    ]

                    # 3D: add OOD trace
                    fig = latent_3d_pane.object
                    if fig is not None:
                        ood_trace = build_ood_trace_3d(z_ood_f, ood_labels_f, class_names, ps)
                        fig.add_trace(ood_trace)
                        latent_3d_pane.object = fig

        latent_2d_pane.object = hv.Layout(panels).opts(shared_axes=False).cols(min(3, len(panels)))

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
        if getattr(cfg, "vision_preproc", False) and getattr(cfg, "vision_height", 0) > 0 and getattr(cfg, "vision_width", 0) > 0:
            image_shape = (cfg.vision_height, cfg.vision_width)
        else:
            side = int(np.sqrt(cfg.input_dim))
            image_shape = (side, side) if side * side == cfg.input_dim else (1, cfg.input_dim)

        recon_pane.object = plot_reconstruction_grid(
            originals, recon_atlas, recon_vq, recon_ae, ns, image_shape
        )

        # --- Per-sample class probability bars ---
        num_classes = loaded.config.num_classes
        labels_sub = _to_numpy(app_state["labels_infer"][indices]).astype(int)

        probs_topo = None
        probs_vq = None
        probs_ae = None
        with torch.no_grad():
            if loaded.classifier_head is not None:
                enc_out = loaded.model_atlas.encoder(X_sub)
                logits = loaded.classifier_head(enc_out[4], enc_out[5])
                probs_topo = F.softmax(logits, dim=1).cpu().numpy()

            if loaded.model_std is not None and loaded.classifier_std is not None:
                z_std = loaded.model_std.encoder(X_sub)
                logits = loaded.classifier_std(z_std)
                probs_vq = F.softmax(logits, dim=1).cpu().numpy()

            if loaded.model_ae is not None and loaded.classifier_ae is not None:
                z_ae = loaded.model_ae(X_sub)[1]
                logits = loaded.classifier_ae(z_ae)
                probs_ae = F.softmax(logits, dim=1).cpu().numpy()

        recon_prob_pane.object = plot_prob_bar_grid(
            probs_topo, probs_vq, probs_ae, labels_sub, ns, num_classes,
        )

        # --- AMI & accuracy on full inference set ---
        X_infer = app_state["X_infer"].float()
        labels_infer = app_state["labels_infer"]
        cache = app_state.get("latent_cache", {})
        batch_size = 2048

        # Atlas: AMI(chart, label) + classifier accuracy
        atlas_ami = None
        atlas_acc = None
        if "K_chart" in cache:
            atlas_ami = adjusted_mutual_info_score(labels_infer, cache["K_chart"])
        if loaded.classifier_head is not None and "router_weights" in cache and "z_geo" in cache:
            rw = torch.tensor(cache["router_weights"], dtype=torch.float32)
            zg = torch.tensor(cache["z_geo"], dtype=torch.float32)
            preds = []
            with torch.no_grad():
                for i in range(0, len(rw), batch_size):
                    logits = loaded.classifier_head(rw[i:i + batch_size], zg[i:i + batch_size])
                    preds.append(logits.argmax(dim=1).numpy())
            atlas_acc = float((np.concatenate(preds) == labels_infer).mean())

        # VQ: AMI(code, label) + classifier accuracy
        vq_ami = None
        vq_acc = None
        if loaded.model_std is not None:
            vq_codes_parts = []
            vq_z_parts = []
            with torch.no_grad():
                for i in range(0, len(X_infer), batch_size):
                    out = loaded.model_std(X_infer[i:i + batch_size])
                    vq_codes_parts.append(_to_numpy(out[2]))  # indices
                    vq_z_parts.append(_to_numpy(loaded.model_std.encoder(X_infer[i:i + batch_size])))
            vq_codes = np.concatenate(vq_codes_parts)
            vq_ami = adjusted_mutual_info_score(labels_infer, vq_codes)
            if loaded.classifier_std is not None:
                vq_z = torch.tensor(np.concatenate(vq_z_parts), dtype=torch.float32)
                preds = []
                with torch.no_grad():
                    for i in range(0, len(vq_z), batch_size):
                        logits = loaded.classifier_std(vq_z[i:i + batch_size])
                        preds.append(logits.argmax(dim=1).numpy())
                vq_acc = float((np.concatenate(preds) == labels_infer).mean())

        # AE: no discrete codes → no AMI; classifier accuracy
        ae_ami = None
        ae_acc = None
        if loaded.model_ae is not None and loaded.classifier_ae is not None:
            ae_z_parts = []
            with torch.no_grad():
                for i in range(0, len(X_infer), batch_size):
                    out = loaded.model_ae(X_infer[i:i + batch_size])
                    ae_z_parts.append(_to_numpy(out[1]))  # z
            ae_z = torch.tensor(np.concatenate(ae_z_parts), dtype=torch.float32)
            preds = []
            with torch.no_grad():
                for i in range(0, len(ae_z), batch_size):
                    logits = loaded.classifier_ae(ae_z[i:i + batch_size])
                    preds.append(logits.argmax(dim=1).numpy())
            ae_acc = float((np.concatenate(preds) == labels_infer).mean())

        # Metrics table
        def _fmt(v, fmt=".6f"):
            return f"{v:{fmt}}" if v is not None else "—"

        def _fmt_pct(v):
            return f"{v:.1%}" if v is not None else "—"

        lines = [
            "| Model | MSE | AMI | Accuracy | Params |",
            "|-------|-----|-----|----------|--------|",
        ]
        lines.append(
            f"| TopoEncoder | {atlas_mse:.6f} | {_fmt(atlas_ami, '.4f')} "
            f"| {_fmt_pct(atlas_acc)} | {atlas_params:,} |"
        )
        if vq_mse is not None:
            lines.append(
                f"| StandardVQ | {vq_mse:.6f} | {_fmt(vq_ami, '.4f')} "
                f"| {_fmt_pct(vq_acc)} | {vq_params:,} |"
            )
        if ae_mse is not None:
            lines.append(
                f"| VanillaAE | {ae_mse:.6f} | {_fmt(ae_ami, '.4f')} "
                f"| {_fmt_pct(ae_acc)} | {ae_params:,} |"
            )
        metrics_table.object = "\n".join(lines)

    def _on_tap(*_events):
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
        if getattr(cfg, "vision_preproc", False) and getattr(cfg, "vision_height", 0) > 0 and getattr(cfg, "vision_width", 0) > 0:
            image_shape = (cfg.vision_height, cfg.vision_width)
        else:
            side = int(np.sqrt(cfg.input_dim))
            image_shape = (side, side) if side * side == cfg.input_dim else (1, cfg.input_dim)

        def _img_to_hv(arr):
            img = arr.reshape(image_shape)
            return hv.Image(
                img,
                kdims=["img_x", "img_y"],
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
        conf_val = float(cache["confidence"][full_idx]) if "confidence" in cache else 0.0
        inspect_label.object = (
            f"**Sample #{full_idx}** — Label: {lbl}, Chart: {chart}, "
            f"Correct: {corr_str}, Confidence: {conf_val:.3f}"
        )

        # Probability bar charts for each classifier
        num_classes = loaded.config.num_classes
        with torch.no_grad():
            if loaded.classifier_head is not None:
                enc_out = loaded.model_atlas.encoder(x_in)
                logits_atlas = loaded.classifier_head(enc_out[4], enc_out[5])
                probs_atlas = F.softmax(logits_atlas, dim=1)[0].cpu().numpy()
                inspect_prob_atlas.object = plot_prob_bars(
                    probs_atlas, lbl, "TopoEncoder", num_classes,
                )
            else:
                inspect_prob_atlas.object = hv.Div("")

            if loaded.classifier_std is not None:
                z_std = loaded.model_std.encoder(x_in)
                logits_std = loaded.classifier_std(z_std)
                probs_std = F.softmax(logits_std, dim=1)[0].cpu().numpy()
                inspect_prob_vq.object = plot_prob_bars(
                    probs_std, lbl, "VQ", num_classes,
                )
            else:
                inspect_prob_vq.object = hv.Div("")

            if loaded.classifier_ae is not None:
                z_ae = loaded.model_ae(x_in)[1]
                logits_ae = loaded.classifier_ae(z_ae)
                probs_ae = F.softmax(logits_ae, dim=1)[0].cpu().numpy()
                inspect_prob_ae.object = plot_prob_bars(
                    probs_ae, lbl, "AE", num_classes,
                )
            else:
                inspect_prob_ae.object = hv.Div("")

    def _on_run_conformal(_event=None):
        """Run all conformal analyses sequentially."""
        loaded = app_state.get("loaded")
        cache = app_state.get("latent_cache")
        if loaded is None or cache is None:
            conformal_status.object = "**Error:** Load a checkpoint first."
            return
        probs_all = cache.get("softmax_probs")
        if probs_all is None:
            conformal_status.object = "**Error:** No classifier head — conformal analysis requires softmax probabilities."
            return

        alpha = conformal_alpha.value
        n_bins = conformal_bins.value
        cal_frac = conformal_cal_frac.value

        z_geo = cache["z_geo"]
        labels = cache["labels"]
        correct = cache["correct"]
        router_w = cache["router_weights"]
        charts = cache["K_chart"]
        codes = cache["K_code"]
        num_classes = loaded.config.num_classes

        try:
            # --- Analysis 1: Accuracy vs Radius ---
            conformal_status.object = "Running analysis 1/9: Accuracy vs Radius..."
            avr = accuracy_vs_radius(z_geo, correct, n_bins)
            acc_vs_radius_pane.object = plot_accuracy_vs_radius(avr)

            # --- Calibration / test split ---
            cal_idx, test_idx = calibration_test_split(len(labels), cal_frac)

            # --- Analysis 7: Reliability Diagram ---
            conformal_status.object = "Running analysis 7/9: Calibration curve..."
            raw_rel = reliability_diagram_data(probs_all[test_idx], labels[test_idx], n_bins)
            raw_ece = expected_calibration_error(probs_all[test_idx], labels[test_idx], n_bins)

            best_beta = tune_beta(probs_all[cal_idx], labels[cal_idx], z_geo[cal_idx], n_bins)
            recal_probs = recalibrate_probs(probs_all[test_idx], z_geo[test_idx], best_beta)
            recal_rel = reliability_diagram_data(recal_probs, labels[test_idx], n_bins)
            recal_ece = expected_calibration_error(recal_probs, labels[test_idx], n_bins)

            reliability_pane.object = plot_reliability_diagram(raw_rel, recal_rel)

            # --- Analyses 2-4: Conformal Coverage ---
            conformal_status.object = "Running analyses 2-4/9: Conformal methods..."
            min_samples = conformal_min_samples.value
            cal_scores_std = conformal_scores_standard(probs_all[cal_idx], labels[cal_idx])
            cal_scores_geo = conformal_scores_geodesic(
                probs_all[cal_idx], labels[cal_idx], z_geo[cal_idx],
            )
            q_std = conformal_quantile(cal_scores_std, alpha)
            q_geo = conformal_quantile(cal_scores_geo, alpha)

            # Per-chart quantiles (using geodesic scores)
            chart_qs = conformal_quantiles_per_chart(
                cal_scores_geo, charts[cal_idx], alpha, min_samples=min_samples,
            )

            # Per-(chart, code) quantiles with hierarchical fallback
            chart_code_qs, cc_stats = conformal_quantiles_per_chart_code(
                cal_scores_geo, charts[cal_idx], codes[cal_idx], alpha,
                min_samples=min_samples,
            )

            # Radial shell quantiles (using geodesic scores)
            rad_edges, rad_qs, rad_stats = conformal_quantiles_per_radius(
                cal_scores_geo, z_geo[cal_idx], alpha,
                n_shells=n_bins, min_samples=min_samples,
            )

            # Geo-β: tune beta and calibrate
            conformal_status.object = "Running analyses 2-4/9: Tuning Geo-β..."
            gb_beta = tune_conformal_beta(
                probs_all[cal_idx], labels[cal_idx], z_geo[cal_idx], alpha,
            )
            cal_scores_gb = conformal_scores_geo_beta(
                probs_all[cal_idx], labels[cal_idx], z_geo[cal_idx], gb_beta,
            )
            q_gb = conformal_quantile(cal_scores_gb, alpha)

            # Build prediction sets on test
            test_probs = probs_all[test_idx]
            test_z = z_geo[test_idx]
            test_labels = labels[test_idx]
            test_charts = charts[test_idx]
            test_codes = codes[test_idx]

            incl_std, sizes_std = prediction_sets(test_probs, q_std, "standard")
            incl_geo, sizes_geo = prediction_sets(
                test_probs, q_geo, "geodesic", z_geo=test_z,
            )
            incl_chart, sizes_chart = prediction_sets(
                test_probs, q_geo, "chart",
                z_geo=test_z, charts=test_charts, chart_quantiles=chart_qs,
            )
            incl_cc, sizes_cc = prediction_sets(
                test_probs, q_geo, "chart_code",
                z_geo=test_z, charts=test_charts, codes=test_codes,
                chart_code_quantiles=chart_code_qs,
            )
            incl_rad, sizes_rad = prediction_sets(
                test_probs, q_geo, "radial",
                z_geo=test_z, radial_edges=rad_edges, radial_quantiles=rad_qs,
            )
            incl_gb, sizes_gb = prediction_sets(
                test_probs, q_gb, "geo_beta",
                z_geo=test_z, geo_beta=gb_beta,
            )

            # Mondrian via class-conditional conformal scores
            class_qs = conformal_quantiles_per_class(
                cal_scores_std, labels[cal_idx], alpha
            )
            incl_mond, sizes_mond = prediction_sets_mondrian(test_probs, class_qs)

            cov_std, mss_std = evaluate_coverage(incl_std, test_labels)
            cov_geo, mss_geo = evaluate_coverage(incl_geo, test_labels)
            cov_chart, mss_chart = evaluate_coverage(incl_chart, test_labels)
            cov_cc, mss_cc = evaluate_coverage(incl_cc, test_labels)
            cov_rad, mss_rad = evaluate_coverage(incl_rad, test_labels)
            cov_gb, mss_gb = evaluate_coverage(incl_gb, test_labels)
            cov_mond, mss_mond = evaluate_coverage(incl_mond, test_labels)

            coverage_results = {
                "Standard": (cov_std, mss_std),
                "Geodesic": (cov_geo, mss_geo),
                "Chart": (cov_chart, mss_chart),
                "Chart×Code": (cov_cc, mss_cc),
                "Radial": (cov_rad, mss_rad),
                "Geo-β": (cov_gb, mss_gb),
                "Mondrian": (cov_mond, mss_mond),
            }
            coverage_summary_table.object = format_coverage_summary_table(coverage_results, alpha)

            # Conditional coverage by radius
            pred_sets_dict = {
                "Standard": (incl_std, sizes_std),
                "Geodesic": (incl_geo, sizes_geo),
                "Chart": (incl_chart, sizes_chart),
                "Chart×Code": (incl_cc, sizes_cc),
                "Radial": (incl_rad, sizes_rad),
                "Geo-β": (incl_gb, sizes_gb),
                "Mondrian": (incl_mond, sizes_mond),
            }
            cond_data = conditional_coverage_by_radius(
                pred_sets_dict, test_labels, test_z, n_bins,
            )
            conditional_coverage_pane.object = plot_conditional_coverage(cond_data, alpha)
            set_size_pane.object = plot_set_size_vs_radius(cond_data)

            # --- Analysis 5: Mondrian Coverage by Class ---
            conformal_status.object = "Running analysis 5/9: Mondrian coverage..."
            pred_sets_cls = {
                "Standard Cov": (incl_std, sizes_std),
                "Geodesic Cov": (incl_geo, sizes_geo),
                "Mondrian Cov": (incl_mond, sizes_mond),
                "Radial Cov": (incl_rad, sizes_rad),
                "Symbol Cov": (incl_cc, sizes_cc),
            }
            cls_data = coverage_by_class(pred_sets_cls, test_labels, num_classes)
            class_coverage_table.object = format_class_coverage_table(cls_data)
            comparison_specs = {
                "Standard": {
                    "conditions": "nothing",
                    "groups": 1,
                    "needs_labels": False,
                    "class_coverage_key": "Standard Cov",
                    "radius_coverage_key": "Standard",
                },
                "Geodesic": {
                    "conditions": "geodesic distance",
                    "groups": 1,
                    "needs_labels": False,
                    "class_coverage_key": "Geodesic Cov",
                    "radius_coverage_key": "Geodesic",
                },
                "Mondrian": {
                    "conditions": "predicted class",
                    "groups": num_classes,
                    "needs_labels": True,
                    "class_coverage_key": "Mondrian Cov",
                    "radius_coverage_key": "Mondrian",
                },
                "Radial": {
                    "conditions": "radius",
                    "groups": rad_stats["n_shells"],
                    "needs_labels": False,
                    "class_coverage_key": "Radial Cov",
                    "radius_coverage_key": "Radial",
                },
                "Symbol": {
                    "conditions": "(chart, code)",
                    "groups": cc_stats["n_groups"],
                    "needs_labels": False,
                    "class_coverage_key": "Symbol Cov",
                    "radius_coverage_key": "Chart×Code",
                },
            }
            coverage_comparison_table.object = format_coverage_method_comparison(
                comparison_specs,
                cls_data,
                cond_data,
            )

            # --- Analysis 8: Ablation ---
            conformal_status.object = "Running analysis 8/9: Feature importance..."
            radius = np.linalg.norm(z_geo, axis=1)
            v_h = router_entropy(router_w)
            abl_data = ablation_feature_importance(
                correct, cache["confidence"], radius, v_h,
            )
            ablation_md = format_ablation_table(abl_data)
            ablation_md += (
                f"\n\n**ECE (raw):** {raw_ece:.4f} | "
                f"**ECE (recal, beta={best_beta:.2f}):** {recal_ece:.4f}"
            )
            ablation_table.object = ablation_md

            # --- Analysis 6: OOD Detection ---
            ds_key = _dataset_key(loaded.config)
            if ds_key == "fashion_mnist":
                ood_name = "MNIST"
                ood_loader = lambda n: get_mnist_data(n_samples=n)[:2]  # noqa: E731
            else:
                ood_name = "Fashion-MNIST"
                ood_loader = lambda n: load_fashion_mnist(n_samples=n)  # noqa: E731
            conformal_status.object = f"Running analysis 6/9: OOD detection ({ood_name})..."
            try:
                ood_X, _ood_labels = ood_loader(len(labels))
                ood_tensor = torch.tensor(
                    ood_X if isinstance(ood_X, np.ndarray) else ood_X.numpy(),
                    dtype=torch.float32,
                )
                z_ood, rw_ood, probs_ood, _charts_ood, _codes_ood = forward_pass_batch(
                    loaded, ood_tensor
                )

                lam_id = conformal_factor_np(z_geo)
                lam_ood = conformal_factor_np(z_ood)
                ent_id = router_entropy(router_w)
                ent_ood = router_entropy(rw_ood)

                chart_to_class = chart_to_label_map(charts, labels)
                tunnel_id = compute_tunneling_rate(
                    router_w,
                    chart_to_class,
                    num_classes,
                )
                tunnel_ood = compute_tunneling_rate(
                    rw_ood,
                    chart_to_class,
                    num_classes,
                )

                codebook = np.array([])
                try:
                    atlas_encoder = getattr(loaded.model_atlas, "encoder", loaded.model_atlas)
                    cb = getattr(atlas_encoder, "codebook", None)
                    if cb is not None:
                        cb_np = _to_numpy(cb)
                        codebook = cb_np.reshape(-1, cb_np.shape[-1])
                except Exception:
                    codebook = np.array([])

                if codebook.size == 0:
                    iso_id = np.zeros(len(labels), dtype=float)
                    iso_ood = np.zeros(len(z_ood), dtype=float)
                else:
                    iso_id = geodesic_isolation(z_geo, codebook)
                    iso_ood = geodesic_isolation(z_ood, codebook)

                # Subsample in-distribution latents for efficient k-NN scoring.
                ref_size = 5000
                if len(z_geo) <= ref_size:
                    z_ref = z_geo
                else:
                    rng = np.random.default_rng(42)
                    ref_idx = rng.choice(len(z_geo), size=ref_size, replace=False)
                    z_ref = z_geo[ref_idx]
                knn_id = hyperbolic_knn_density(z_geo, z_ref, k=10)
                knn_ood = hyperbolic_knn_density(z_ood, z_ref, k=10)

                max_prob_id = 1.0 - cache["confidence"]
                max_prob_ood = (
                    1.0 - probs_ood.max(axis=1)
                    if probs_ood is not None
                    else np.ones(len(z_ood))
                )

                id_signals = {
                    "1 - max_prob": max_prob_id,
                    "1/lambda": 1.0 / lam_id,
                    "router_entropy": ent_id,
                    "tunneling_rate": tunnel_id,
                    "geodesic_isolation": iso_id,
                    "hyperbolic_knn_density": knn_id,
                }
                ood_signals_dict = {
                    "1 - max_prob": max_prob_ood,
                    "1/lambda": 1.0 / lam_ood,
                    "router_entropy": ent_ood,
                    "tunneling_rate": tunnel_ood,
                    "geodesic_isolation": iso_ood,
                    "hyperbolic_knn_density": knn_ood,
                }

                feature_order = [
                    "1 - max_prob",
                    "1/lambda",
                    "router_entropy",
                    "tunneling_rate",
                    "geodesic_isolation",
                    "hyperbolic_knn_density",
                ]
                X_id = np.column_stack([id_signals[k] for k in feature_order])
                X_ood = np.column_stack([ood_signals_dict[k] for k in feature_order])
                X = np.vstack([X_id, X_ood])
                y = np.concatenate([
                    np.zeros(len(labels), dtype=np.int64),
                    np.ones(len(z_ood), dtype=np.int64),
                ])
                try:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.preprocessing import StandardScaler

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    lr = LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                    )
                    lr.fit(X_scaled, y)
                    probs_combined = lr.predict_proba(X_scaled)[:, 1]
                    id_signals["combined"] = probs_combined[: len(labels)]
                    ood_signals_dict["combined"] = probs_combined[len(labels):]
                except Exception:
                    z_id = np.column_stack([
                        (vals - vals.mean()) / max(vals.std(), 1e-6)
                        for vals in (
                            max_prob_id,
                            1.0 / lam_id,
                            ent_id,
                            tunnel_id,
                            iso_id,
                            knn_id,
                        )
                    ])
                    z_ood = np.column_stack([
                        (vals - vals.mean()) / max(vals.std(), 1e-6)
                        for vals in (
                            max_prob_ood,
                            1.0 / lam_ood,
                            ent_ood,
                            tunnel_ood,
                            iso_ood,
                            knn_ood,
                        )
                    ])
                    score_id = z_id.mean(axis=1)
                    score_ood = z_ood.mean(axis=1)
                    id_signals["combined"] = score_id
                    ood_signals_dict["combined"] = score_ood

                aurocs = ood_scores(id_signals, ood_signals_dict)
                ood_roc_pane.object = plot_ood_roc(id_signals, ood_signals_dict)
                ood_auroc_table.object = format_ood_auroc_table(aurocs)
            except Exception as exc:
                ood_roc_pane.object = hv.Div("")
                ood_auroc_table.object = f"**OOD skipped:** {exc}"

            # --- Analysis 9: Corruption Robustness ---
            conformal_status.object = "Running analysis 9/9: Corruption robustness..."
            try:
                X_np = _to_numpy(app_state["X_infer"])
                corr_data = coverage_under_corruption(
                    loaded, X_np, labels, cal_idx, test_idx, alpha,
                    n_intensities=conformal_n_corrupt.value,
                )
                if corr_data:
                    corruption_pane.object = plot_corruption_coverage(corr_data, alpha)
                else:
                    corruption_pane.object = hv.Div("No classifier head for corruption analysis.")
            except Exception as exc:
                corruption_pane.object = hv.Div(f"Corruption analysis failed: {exc}")

            diag_parts = [
                "**Conformal analysis complete.**",
                f"n_min={min_samples}",
                f"Chart×Code: {cc_stats['n_groups']}g "
                f"({cc_stats['n_fine']}fine/{cc_stats['n_chart_fallback']}chart/"
                f"{cc_stats['n_global_fallback']}global, "
                f"min={cc_stats['min_group_size']})",
                f"Radial: {rad_stats['n_shells']}shells "
                f"({rad_stats['n_fine']}fine/{rad_stats['n_fallback']}fallback, "
                f"min={rad_stats['min_group_size']})",
                f"Geo-β={gb_beta:.2f}",
            ]
            conformal_status.object = " | ".join(diag_parts)

        except Exception as exc:
            conformal_status.object = f"**Error:** {exc}"
            traceback.print_exc()

    tap_stream.param.watch(_on_tap, ["x", "y"])

    # Wire up callbacks
    scan_btn.on_click(_on_scan)
    run_selector.param.watch(_on_run_selected, "value")
    load_btn.on_click(_on_load)
    run_conformal_btn.on_click(_on_run_conformal)

    # Latent display options: only re-render plots from cache (no re-inference)
    for w in [latent_samples, color_by, point_size, show_hierarchy, tree_line_color, tree_line_width, alpha_by_confidence, ood_toggle, ood_n_samples]:
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
