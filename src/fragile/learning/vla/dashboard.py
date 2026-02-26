"""Panel-based interactive dashboard for VLA World Model checkpoint inspection.

Reuses latent-space visualisation patterns from the TopoEncoder dashboard
(``plots.py``) but replaces class labels with task labels, removes all
accuracy/classification widgets, and shows original LeRobot camera images.

Usage:
    uv run fragile vla-dashboard --port 5009
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import logging
import os
import re
import traceback

import holoviews as hv
import numpy as np
import panel as pn
import torch
import torch.nn.functional as F

from fragile.learning.core.layers import FactorizedJumpOperator, TopoEncoderPrimitives
from fragile.learning.plots import (
    _to_numpy,
    build_latent_scatter,
    chart_to_label_map,
    plot_chart_usage,
    plot_latent_2d_slices,
    plot_latent_3d,
)

logger = logging.getLogger(__name__)

os.environ.setdefault("PLOTLY_RENDERER", "json")
hv.extension("bokeh")

__all__ = ["create_app"]

# ---------------------------------------------------------------------------
# Checkpoint filename patterns
# ---------------------------------------------------------------------------

# p{phase}_epoch_{epoch}.pt  OR  epoch_{epoch}.pt  OR  checkpoint_final.pt
_CKPT_RE = re.compile(
    r"(?:p(\d+)_)?(?:epoch_(\d+)|checkpoint_final)\.pt$"
)

# Keys forwarded to TopoEncoderPrimitives.__init__ from checkpoint args dict
_ENCODER_INIT_KEYS = {
    "input_dim", "hidden_dim", "latent_dim", "num_charts", "codes_per_chart",
    "covariant_attn", "covariant_attn_tensorization", "covariant_attn_rank",
    "covariant_attn_tau_min", "covariant_attn_denom_min",
    "covariant_attn_use_transport", "covariant_attn_transport_eps",
    "conv_backbone",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VLACheckpointInfo:
    """Metadata about a discovered VLA checkpoint."""

    path: str
    phase: int
    epoch: int
    label: str  # human-friendly label for selector


@dataclass
class VLALoaded:
    """Loaded VLA checkpoint data ready for inference."""

    encoder: TopoEncoderPrimitives
    jump_op: FactorizedJumpOperator
    world_model: object | None  # GeometricWorldModel or None
    args: dict
    epoch: int
    phase: int


# ---------------------------------------------------------------------------
# Scanning & loading
# ---------------------------------------------------------------------------


def scan_vla_runs(outputs_dir: str) -> list[VLACheckpointInfo]:
    """Scan *outputs_dir* for VLA checkpoint files."""
    results: list[VLACheckpointInfo] = []
    if not os.path.isdir(outputs_dir):
        return results
    for root, _dirs, files in os.walk(outputs_dir):
        for fname in sorted(files):
            m = _CKPT_RE.search(fname)
            if not m:
                continue
            phase = int(m.group(1)) if m.group(1) else 0
            epoch = int(m.group(2)) if m.group(2) else -1
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, outputs_dir)
            label = f"{rel}  (P{phase} E{epoch})" if epoch >= 0 else f"{rel}  (final)"
            results.append(VLACheckpointInfo(path=path, phase=phase, epoch=epoch, label=label))
    return results


def load_vla_checkpoint(ckpt_path: str) -> VLALoaded:
    """Load a VLA checkpoint and reconstruct models."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    args = ckpt.get("args") or ckpt.get("config") or {}

    # Encoder state: key may be "model" (unsup / joint) or "encoder" (3-phase train.py)
    enc_state = ckpt.get("model") or ckpt.get("encoder")
    if enc_state is None:
        raise RuntimeError("Checkpoint has no 'model' or 'encoder' state_dict")

    # Build encoder kwargs from args
    enc_kwargs: dict = {}
    for k in _ENCODER_INIT_KEYS:
        # Handle argparse-style keys (e.g. hidden_dim vs hidden-dim)
        for candidate in (k, k.replace("_", "-")):
            if candidate in args:
                enc_kwargs[k] = args[candidate]
                break
    # Defaults
    enc_kwargs.setdefault("input_dim", args.get("input_dim", args.get("feature_dim", 720)))
    enc_kwargs.setdefault("hidden_dim", 256)
    enc_kwargs.setdefault("latent_dim", args.get("latent_dim", 16))
    enc_kwargs.setdefault("num_charts", args.get("num_charts", 8))
    enc_kwargs.setdefault("codes_per_chart", args.get("codes_per_chart", 32))
    enc_kwargs.setdefault("covariant_attn", True)
    enc_kwargs.setdefault("conv_backbone", False)

    encoder = TopoEncoderPrimitives(film_conditioning=True, **enc_kwargs)
    result = encoder.load_state_dict(enc_state, strict=False)
    if result.missing_keys:
        logger.warning("Encoder: %d missing keys", len(result.missing_keys))
    encoder.eval()

    # Jump operator
    jump_state = ckpt.get("jump_op")
    jump_op = FactorizedJumpOperator(
        num_charts=enc_kwargs["num_charts"],
        latent_dim=enc_kwargs["latent_dim"],
    )
    if jump_state is not None:
        jump_op.load_state_dict(jump_state, strict=False)
    jump_op.eval()

    # World model (optional)
    world_model = None
    wm_state = ckpt.get("world_model")
    if wm_state is not None:
        try:
            from fragile.learning.vla.covariant_world_model import GeometricWorldModel

            wm_kwargs: dict = {
                "latent_dim": enc_kwargs["latent_dim"],
                "action_dim": args.get("action_dim", 6),
                "num_charts": enc_kwargs["num_charts"],
                "d_model": args.get("wm_d_model", 128),
                "hidden_dim": args.get("wm_hidden_dim", 256),
                "dt": args.get("wm_dt", 0.01),
                "gamma_friction": args.get("wm_gamma_friction", 1.0),
                "T_c": args.get("wm_T_c", 0.1),
                "alpha_potential": args.get("wm_alpha_potential", 0.5),
                "beta_curl": args.get("wm_beta_curl", 0.1),
                "gamma_risk": args.get("wm_gamma_risk", 0.01),
                "use_boris": args.get("wm_use_boris", True),
                "use_jump": args.get("wm_use_jump", True),
                "jump_rate_hidden": args.get("wm_jump_rate_hidden", 64),
                "min_length": max(args.get("wm_min_length", 0.03), 0.0),
                "risk_metric_alpha": args.get("wm_risk_metric_alpha", 0.0),
            }
            world_model = GeometricWorldModel(**wm_kwargs)
            world_model.load_state_dict(wm_state, strict=False)
            world_model.eval()
        except Exception:
            logger.warning("Could not load world model: %s", traceback.format_exc())
            world_model = None

    return VLALoaded(
        encoder=encoder,
        jump_op=jump_op,
        world_model=world_model,
        args=args,
        epoch=ckpt.get("epoch", -1),
        phase=ckpt.get("phase", 0),
    )


# ---------------------------------------------------------------------------
# Feature + task label loading
# ---------------------------------------------------------------------------


def load_feature_cache(feature_dir: str) -> dict:
    """Load all features + task labels from the feature cache directory.

    Returns dict with keys:
        features: [N, D] tensor
        task_labels: [N] int array
        episode_ids: [N] int array
        timesteps: [N] int array
    """
    from pathlib import Path

    cache = Path(feature_dir)
    meta_path = cache / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        episode_ids_list: list[int] = meta.get("episode_ids", [])
    else:
        episode_ids_list = sorted(
            int(p.name.split("_")[1])
            for p in cache.iterdir()
            if p.is_dir() and p.name.startswith("episode_")
        )

    all_features: list[torch.Tensor] = []
    all_task_labels: list[torch.Tensor] = []
    all_ep_ids: list[int] = []
    all_timesteps: list[int] = []

    for ep_id in episode_ids_list:
        ep_dir = cache / f"episode_{ep_id}"
        feat_path = ep_dir / "features.pt"
        if not feat_path.exists():
            continue
        feat = torch.load(feat_path, weights_only=True)  # [T, D]
        T = feat.shape[0]
        all_features.append(feat)

        task_path = ep_dir / "task_indices.pt"
        if task_path.exists():
            tl = torch.load(task_path, weights_only=True)
            if tl.shape[0] < T:
                tl = F.pad(tl, (0, T - tl.shape[0]), value=0)
            all_task_labels.append(tl[:T])
        else:
            all_task_labels.append(torch.zeros(T, dtype=torch.long))

        all_ep_ids.extend([ep_id] * T)
        all_timesteps.extend(range(T))

    if not all_features:
        return {
            "features": torch.zeros(0, 720),
            "task_labels": np.zeros(0, dtype=int),
            "episode_ids": np.zeros(0, dtype=int),
            "timesteps": np.zeros(0, dtype=int),
        }

    return {
        "features": torch.cat(all_features, dim=0),
        "task_labels": _to_numpy(torch.cat(all_task_labels, dim=0)).astype(int),
        "episode_ids": np.array(all_ep_ids, dtype=int),
        "timesteps": np.array(all_timesteps, dtype=int),
    }


# ---------------------------------------------------------------------------
# Lazy image provider
# ---------------------------------------------------------------------------


class VLAImageProvider:
    """Lazy loader for LeRobot camera images via direct video decoding."""

    def __init__(self, dataset_name: str, feature_cache_dir: str):
        self._dataset_name = dataset_name
        self._cache_dir = feature_cache_dir
        self._video_root: str | None = None
        self._ep_offsets: dict[int, int] | None = None  # ep_id -> global frame offset
        self._cameras: list[str] | None = None
        self._ready = False

    def _ensure_loaded(self) -> bool:
        if self._ready:
            return True
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset(self._dataset_name)
            self._video_root = str(ds.root / "videos")
            self._cameras = [
                k.replace("observation.images.", "")
                for k in (ds.meta.camera_keys or [])
            ]

            # Build episode-start-offset table from the parquet metadata
            hf = ds.hf_dataset
            ep_col = np.asarray(hf["episode_index"])
            offset = 0
            self._ep_offsets = {}
            for ep_id in sorted(np.unique(ep_col)):
                self._ep_offsets[int(ep_id)] = offset
                offset += int((ep_col == ep_id).sum())

            self._ready = True
            logger.info(
                "Image provider ready: %d episodes, cameras=%s",
                len(self._ep_offsets), self._cameras,
            )
            return True
        except Exception:
            logger.warning("Could not init image provider: %s", traceback.format_exc())
            return False

    def get_image(
        self, ep_id: int, timestep: int, camera: str = "top"
    ) -> np.ndarray | None:
        """Return [H, W, 3] uint8 image or None."""
        if not self._ensure_loaded():
            return None
        try:
            import imageio.v3 as iio

            offset = self._ep_offsets.get(ep_id)
            if offset is None:
                return None
            global_idx = offset + timestep

            # Resolve camera to video path
            cam_key = f"observation.images.{camera}"
            video_dir = os.path.join(self._video_root, cam_key, "chunk-000")
            if not os.path.isdir(video_dir):
                # Fallback: try first available camera
                for c in (self._cameras or []):
                    alt_key = f"observation.images.{c}"
                    alt_dir = os.path.join(self._video_root, alt_key, "chunk-000")
                    if os.path.isdir(alt_dir):
                        video_dir = alt_dir
                        break
                else:
                    return None

            # Find the video file (typically file-000.mp4)
            video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
            if not video_files:
                return None
            video_path = os.path.join(video_dir, sorted(video_files)[0])

            frame = iio.imread(video_path, index=global_idx, plugin="pyav")
            return frame  # [H, W, 3] uint8
        except Exception:
            logger.debug("Image read failed ep=%d t=%d: %s", ep_id, timestep,
                         traceback.format_exc())
            return None


# ---------------------------------------------------------------------------
# Helpers for image/latent display
# ---------------------------------------------------------------------------


def _tensor_to_png_pane(img_tensor_or_array, width: int = 150):
    """Convert image tensor/array to Panel PNG pane."""
    from PIL import Image

    if isinstance(img_tensor_or_array, torch.Tensor):
        arr = (
            img_tensor_or_array.permute(1, 2, 0).numpy() * 255
        ).clip(0, 255).astype(np.uint8)
    else:
        arr = img_tensor_or_array
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return pn.pane.PNG(buf.getvalue(), width=width)


def _numpy_to_png_pane(arr: np.ndarray, width: int = 150):
    """Convert [H, W, 3] uint8 array to Panel PNG pane."""
    from PIL import Image

    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return pn.pane.PNG(buf.getvalue(), width=width)


def _latent_bar_chart(z_vec: np.ndarray, width: int = 300, height: int = 100):
    """Horizontal bar chart of latent vector components with signed coloring."""
    z = _to_numpy(z_vec).ravel()
    sign = np.where(z >= 0, "pos", "neg")
    data = {
        "dim": [f"z{i}" for i in range(len(z))],
        "value": z.astype(float),
        "sign": sign,
    }
    bars = hv.Bars(data, "dim", ["value", "sign"]).opts(
        color="sign", cmap={"pos": "#4c78a8", "neg": "#e45756"},
        width=width, height=height, xrotation=90,
        xaxis="bare",
        title="Latent vector",
    )
    return bars


def _feature_recon_bar(orig: np.ndarray, recon: np.ndarray, top_k: int = 20,
                       width: int = 300, height: int = 100):
    """Bar chart of top-K largest reconstruction error dimensions."""
    diff = np.abs(orig - recon)
    top_idx = np.argsort(diff)[-top_k:][::-1]
    data = [(f"d{i}", float(diff[i])) for i in top_idx]
    bars = hv.Bars(data, "dim", "|error|").opts(
        color="#f58518", width=width, height=height, xrotation=90,
        xaxis="bare",
        title=f"Top-{top_k} recon error dims",
    )
    return bars


# ---------------------------------------------------------------------------
# Dashboard app
# ---------------------------------------------------------------------------


def create_app(outputs_dir: str = "outputs/vla") -> pn.template.FastListTemplate:
    """Create the interactive VLA World Model dashboard."""
    pn.extension("plotly", "tabulator")

    # ---- Shared state ----
    app_state: dict = {
        "checkpoints": [],
        "loaded": None,
        "cache": None,  # feature cache dict
        "image_provider": None,
    }

    # ---- Sidebar widgets ----
    scan_btn = pn.widgets.Button(name="Scan runs", button_type="primary", width=300)
    ckpt_selector = pn.widgets.Select(name="Checkpoint", options=[], width=300)
    load_btn = pn.widgets.Button(name="Load checkpoint", button_type="success", width=300)
    n_samples = pn.widgets.IntSlider(
        name="Recon samples", start=4, end=24, value=8, step=4, width=300,
    )
    latent_samples = pn.widgets.IntSlider(
        name="Latent samples", start=100, end=50000, value=2000, step=100, width=300,
    )
    seed_input = pn.widgets.IntInput(name="Random seed", value=42, step=1, width=300)
    color_by = pn.widgets.RadioButtonGroup(
        name="Color by",
        options=["timestep", "chart", "episode", "radius"],
        value="timestep",
        button_type="default",
    )
    point_size = pn.widgets.IntSlider(name="Point size", start=1, end=10, value=3, width=300)
    show_latents = pn.widgets.Checkbox(name="Show latent points", value=True, width=300)
    show_chart_centers = pn.widgets.Checkbox(name="Show chart centers", value=False, width=300)
    show_code_centers = pn.widgets.Checkbox(name="Show code centers", value=False, width=300)
    show_tree_lines = pn.widgets.Checkbox(name="Show tree lines", value=False, width=300)
    tree_line_color = pn.widgets.Select(
        name="Line color", options=["black", "chart", "symbol"], value="black", width=300,
    )
    tree_line_width = pn.widgets.EditableFloatSlider(
        name="Line width", start=0.1, end=5.0, value=0.5, step=0.1, width=300,
    )
    camera_selector = pn.widgets.Select(
        name="Camera", options=["top", "wrist"], value="top", width=300,
    )
    status = pn.pane.Markdown("Click **Scan runs** to begin.", width=300)

    sidebar = pn.Column(
        pn.pane.Markdown("## VLA Dashboard"),
        scan_btn,
        ckpt_selector,
        load_btn,
        pn.layout.Divider(),
        pn.pane.Markdown("### Display"),
        n_samples,
        latent_samples,
        seed_input,
        color_by,
        point_size,
        show_latents,
        pn.layout.Divider(),
        pn.pane.Markdown("### Hierarchy"),
        show_chart_centers,
        show_code_centers,
        show_tree_lines,
        tree_line_color,
        tree_line_width,
        pn.layout.Divider(),
        pn.pane.Markdown("### Images"),
        camera_selector,
        pn.layout.Divider(),
        status,
        width=350,
    )

    # ---- Main panes (placeholders) ----
    # Tab 1: Latent Space
    latent_3d_pane = pn.pane.Plotly(None, sizing_mode="stretch_width", height=600)
    latent_2d_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width")
    usage_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width")
    inspect_label = pn.pane.Markdown("*Click a point in z0 vs z1 to inspect*")
    inspect_image = pn.Column(
        pn.pane.Markdown("*(click a point)*"), width=200, height=200,
    )
    inspect_latent_bar = pn.pane.HoloViews(hv.Div(""), width=350, height=150)
    inspect_meta = pn.pane.Markdown("")
    inspect_row = pn.Row(
        pn.Column("**Camera image**", inspect_image),
        pn.Column("**Latent vector**", inspect_latent_bar),
        pn.Column("**Metadata**", inspect_meta),
    )
    tap_stream = hv.streams.Tap(x=0, y=0)

    # Tab 2: Reconstructions
    recon_pane = pn.Column(pn.pane.Markdown("*Load a checkpoint to see reconstructions.*"))
    recon_summary = pn.pane.Markdown("")

    # Tab 3: Dynamics
    dynamics_pane = pn.Column(
        pn.pane.Markdown("*Load a checkpoint with a world model to see dynamics.*")
    )

    # ---- Tabs ----
    latent_tab = pn.Column(
        latent_3d_pane,
        latent_2d_pane,
        inspect_label,
        inspect_row,
        usage_pane,
        sizing_mode="stretch_width",
    )
    recon_tab = pn.Column(recon_pane, recon_summary, sizing_mode="stretch_width")
    dynamics_tab = pn.Column(dynamics_pane, sizing_mode="stretch_width")

    tabs = pn.Tabs(
        ("Latent Space", latent_tab),
        ("Reconstructions", recon_tab),
        ("Dynamics", dynamics_tab),
        sizing_mode="stretch_both",
    )

    # ---- Callbacks ----
    def _on_scan(_event=None):
        ckpts = scan_vla_runs(outputs_dir)
        app_state["checkpoints"] = ckpts
        if not ckpts:
            ckpt_selector.options = []
            status.object = "No checkpoints found."
            return
        ckpt_selector.options = {c.label: c.label for c in ckpts}
        ckpt_selector.value = ckpts[-1].label
        status.object = f"Found **{len(ckpts)}** checkpoint(s)."

    def _on_load(_event=None):
        label = ckpt_selector.value
        info = next((c for c in app_state["checkpoints"] if c.label == label), None)
        if info is None:
            status.object = "**Error:** Select a checkpoint first."
            return
        status.object = f"Loading {info.label}..."
        try:
            loaded = load_vla_checkpoint(info.path)
        except Exception as exc:
            status.object = f"**Error loading:** {exc}"
            traceback.print_exc()
            return
        app_state["loaded"] = loaded

        # Find feature cache dir
        feature_dir = _find_feature_dir(info.path, outputs_dir, loaded.args)
        if feature_dir is not None:
            try:
                cache = load_feature_cache(feature_dir)
                app_state["cache"] = cache
            except Exception as exc:
                status.object = f"Loaded model but feature cache error: {exc}"
                app_state["cache"] = None
        else:
            app_state["cache"] = None
            status.object = "Loaded model but no feature cache found."

        # Image provider — try checkpoint args first, then feature cache meta.json
        dataset_name = loaded.args.get("dataset_name", loaded.args.get("dataset", ""))
        if not dataset_name and feature_dir:
            meta_path = os.path.join(feature_dir, "meta.json")
            if os.path.isfile(meta_path):
                try:
                    meta = json.loads(open(meta_path).read())
                    dataset_name = meta.get("dataset", "")
                except Exception:
                    pass
        if dataset_name and feature_dir:
            app_state["image_provider"] = VLAImageProvider(dataset_name, feature_dir)
        else:
            app_state["image_provider"] = None

        _refresh_all()
        wm_str = "with world model" if loaded.world_model is not None else "encoder only"
        n_feat = app_state["cache"]["features"].shape[0] if app_state["cache"] else 0
        status.object = (
            f"Loaded P{loaded.phase} E{loaded.epoch} ({wm_str}). "
            f"{n_feat} feature frames."
        )

    def _find_feature_dir(
        ckpt_path: str, outputs_dir: str, args: dict
    ) -> str | None:
        """Try to locate the feature cache directory."""
        # 1. From args
        for key in ("feature_cache_dir", "feature-cache-dir"):
            if key in args:
                candidate = args[key]
                if os.path.isdir(candidate):
                    return candidate

        # 2. Sibling features/ directory
        ckpt_dir = os.path.dirname(ckpt_path)
        for d in (ckpt_dir, os.path.dirname(ckpt_dir), outputs_dir):
            candidate = os.path.join(d, "features")
            if os.path.isdir(candidate):
                return candidate
        return None

    def _refresh_all():
        _refresh_latent()
        _refresh_recon()
        _refresh_dynamics()

    def _refresh_latent():
        loaded = app_state.get("loaded")
        cache = app_state.get("cache")
        if loaded is None or cache is None:
            return

        features = cache["features"]
        task_labels = cache["task_labels"]
        episode_ids = cache["episode_ids"]

        N = features.shape[0]
        n_lat = min(latent_samples.value, N)
        rng = np.random.RandomState(seed_input.value)
        idx = rng.choice(N, size=n_lat, replace=False) if n_lat < N else np.arange(N)

        x_sub = features[idx]
        task_sub = task_labels[idx]
        ep_sub = episode_ids[idx]

        # Forward pass
        with torch.no_grad():
            enc_out = loaded.encoder.encoder(x_sub)
            K_code = enc_out[1]  # index 1 is K_code from PrimitiveAttentiveAtlasEncoder
            (
                x_recon, vq_loss, enc_rw, dec_rw,
                K_chart, z_geo, z_n, c_bar, aux,
            ) = loaded.encoder(x_sub)

        K_code_np = _to_numpy(K_code).astype(int)

        z_np = _to_numpy(z_geo)
        K_np = _to_numpy(K_chart).astype(int)
        radii = np.linalg.norm(z_np, axis=1)

        # Map color_by -> labels for the scatter functions
        timesteps = cache["timesteps"]
        ts_sub = timesteps[idx]
        cb = color_by.value
        if cb == "timestep":
            labels_for_color = ts_sub
        elif cb == "chart":
            labels_for_color = K_np
        elif cb == "episode":
            labels_for_color = ep_sub
        elif cb == "radius":
            # Bin radii into 10 bins for discrete coloring
            labels_for_color = np.digitize(radii, np.linspace(0, radii.max() + 1e-8, 11)) - 1
        else:
            labels_for_color = ts_sub

        dummy_correct = np.ones(len(labels_for_color), dtype=int)

        # Use "label" color mode since we've already set labels_for_color
        scatter_color = "label"

        # 3D scatter
        try:
            fig3d = plot_latent_3d(
                z_np, labels_for_color, K_chart=K_np, correct=dummy_correct,
                color_by=scatter_color, point_size=point_size.value,
                show_points=show_latents.value,
                show_chart_centers=show_chart_centers.value,
                show_code_centers=show_code_centers.value,
                show_tree_lines=show_tree_lines.value,
                tree_line_color=tree_line_color.value,
                tree_line_width=tree_line_width.value,
                K_code=K_code_np,
                show_leaf_lines=False,
            )
            latent_3d_pane.object = fig3d
        except Exception:
            logger.warning("3D scatter error: %s", traceback.format_exc())

        # 2D slices — build manually so we can wire the Tap stream
        try:
            dim = z_np.shape[1]
            pairs = []
            if dim >= 2:
                pairs.append((0, 1))
            if dim >= 3:
                pairs.extend([(0, 2), (1, 2)])

            scatter_panels = []
            for di, dj in pairs:
                scatter = build_latent_scatter(
                    z_np, labels_for_color, K_np, dummy_correct,
                    scatter_color, point_size.value, di, dj, indices=idx,
                    K_code=K_code_np,
                    show_code_centers=show_code_centers.value,
                    show_points=show_latents.value,
                )
                scatter_panels.append(scatter)

            if scatter_panels:
                # Wire tap stream to the first scatter (z0 vs z1)
                tap_stream.source = scatter_panels[0]
                layout_2d = hv.Layout(scatter_panels).opts(
                    shared_axes=False,
                ).cols(min(3, len(scatter_panels)))
                latent_2d_pane.object = layout_2d
        except Exception:
            logger.warning("2D scatter error: %s", traceback.format_exc())

        # Chart usage
        try:
            n_charts = int(K_np.max()) + 1
            usage = np.zeros(n_charts)
            for c in K_np:
                usage[c] += 1
            usage /= usage.sum() + 1e-8
            usage_pane.object = plot_chart_usage(usage)
        except Exception:
            logger.warning("Chart usage error: %s", traceback.format_exc())

        # Store sub-arrays for click inspect
        app_state["latent_sub"] = {
            "z_np": z_np, "K_np": K_np, "K_code_np": K_code_np,
            "task_sub": task_sub,
            "ep_sub": ep_sub, "idx": idx, "x_sub": x_sub,
            "x_recon": _to_numpy(x_recon), "radii": radii,
        }

    def _on_tap(x, y):
        """Handle click on 2D scatter to inspect nearest point."""
        sub = app_state.get("latent_sub")
        if sub is None:
            return
        z_np = sub["z_np"]
        if z_np.shape[1] < 2:
            return

        # Find nearest point in z0-z1 display space
        dists = (z_np[:, 0] - x) ** 2 + (z_np[:, 1] - y) ** 2
        nearest = int(np.argmin(dists))

        z_vec = z_np[nearest]
        chart = int(sub["K_np"][nearest])
        task = int(sub["task_sub"][nearest])
        ep = int(sub["ep_sub"][nearest])
        global_idx = int(sub["idx"][nearest])
        cache = app_state["cache"]
        ts = int(cache["timesteps"][global_idx]) if cache is not None else 0
        radius = float(sub["radii"][nearest])

        # Camera image
        provider = app_state.get("image_provider")
        inspect_image.clear()
        if provider is not None:
            img = provider.get_image(ep, ts, camera=camera_selector.value)
            if img is not None:
                inspect_image.append(_numpy_to_png_pane(img, width=180))
            else:
                inspect_image.append(pn.pane.Markdown("*Image unavailable*"))
        else:
            inspect_image.append(pn.pane.Markdown("*No image provider*"))

        # Latent bar chart
        try:
            inspect_latent_bar.object = _latent_bar_chart(z_vec, width=300, height=130)
        except Exception:
            pass

        # Metadata
        inspect_meta.object = (
            f"**Task:** {task}  \n"
            f"**Episode:** {ep}  \n"
            f"**Timestep:** {ts}  \n"
            f"**Chart:** {chart}  \n"
            f"**||z||:** {radius:.4f}  \n"
            f"**Index:** {global_idx}"
        )

    tap_stream.param.watch(lambda event: _on_tap(event.new, tap_stream.y), "x")

    def _refresh_recon():
        loaded = app_state.get("loaded")
        cache = app_state.get("cache")
        if loaded is None or cache is None:
            recon_pane.clear()
            recon_pane.append(pn.pane.Markdown("*No data loaded.*"))
            return

        features = cache["features"]
        task_labels = cache["task_labels"]
        episode_ids = cache["episode_ids"]
        timesteps = cache["timesteps"]

        N = features.shape[0]
        n_rec = min(n_samples.value, N)
        rng = np.random.RandomState(seed_input.value + 1)
        idx = rng.choice(N, size=n_rec, replace=False) if n_rec < N else np.arange(N)

        x_sub = features[idx]
        with torch.no_grad():
            enc_out = loaded.encoder.encoder(x_sub)
            K_code = enc_out[1]  # index 1 is K_code from PrimitiveAttentiveAtlasEncoder
            (
                x_recon, vq_loss, enc_rw, dec_rw,
                K_chart, z_geo, z_n, c_bar, aux,
            ) = loaded.encoder(x_sub)

        K_code_np = _to_numpy(K_code).astype(int)
        x_np = _to_numpy(x_sub)
        xr_np = _to_numpy(x_recon)
        z_np = _to_numpy(z_geo)
        K_np = _to_numpy(K_chart).astype(int)
        radii = np.linalg.norm(z_np, axis=1)

        provider = app_state.get("image_provider")

        rows = []
        mse_list = []
        for i in range(n_rec):
            gi = int(idx[i])
            ep = int(episode_ids[gi])
            ts = int(timesteps[gi])
            task = int(task_labels[gi])
            chart = int(K_np[i])
            r = float(radii[i])
            mse = float(((x_np[i] - xr_np[i]) ** 2).mean())
            mse_list.append(mse)

            # Image column
            img_pane = pn.pane.Markdown("*N/A*")
            if provider is not None:
                img = provider.get_image(ep, ts, camera=camera_selector.value)
                if img is not None:
                    img_pane = _numpy_to_png_pane(img, width=120)

            # Latent bar
            latent_bar = pn.pane.HoloViews(
                _latent_bar_chart(z_np[i], width=250, height=80),
                width=270, height=100,
            )

            # Recon error bar
            recon_bar = pn.pane.HoloViews(
                _feature_recon_bar(x_np[i], xr_np[i], top_k=15, width=250, height=80),
                width=270, height=100,
            )

            # Metadata
            meta = pn.pane.Markdown(
                f"Task: {task} | Ep: {ep} | t: {ts}  \n"
                f"Chart: {chart} | ||z||: {r:.3f} | MSE: {mse:.4f}"
            )

            row = pn.Row(img_pane, latent_bar, recon_bar, meta)
            rows.append(row)

        # Summary
        mean_mse = float(np.mean(mse_list)) if mse_list else 0.0
        ss_res = ((x_np - xr_np) ** 2).sum()
        ss_tot = ((x_np - x_np.mean(axis=0)) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        n_active = len(np.unique(K_np))
        mean_r = float(radii.mean())

        recon_pane.clear()
        recon_pane.extend(rows)
        recon_summary.object = (
            f"**Summary:** Mean MSE = {mean_mse:.5f} | R² = {r2:.4f} | "
            f"Active charts = {n_active} | Mean ||z|| = {mean_r:.4f}"
        )

    def _refresh_dynamics():
        loaded = app_state.get("loaded")
        cache = app_state.get("cache")
        if loaded is None or loaded.world_model is None:
            dynamics_pane.clear()
            dynamics_pane.append(
                pn.pane.Markdown("*No world model in this checkpoint.*")
            )
            return

        dynamics_pane.clear()

        # Run encoder on all cached features to get z_geo + K_chart
        features = cache["features"]
        episode_ids = cache["episode_ids"]
        task_labels = cache["task_labels"]

        N = features.shape[0]
        n_sub = min(5000, N)
        rng = np.random.RandomState(seed_input.value + 2)
        idx = rng.choice(N, size=n_sub, replace=False) if n_sub < N else np.arange(N)

        with torch.no_grad():
            (
                x_recon, vq_loss, enc_rw, dec_rw,
                K_chart, z_geo, z_n, c_bar, aux,
            ) = loaded.encoder(features[idx])

        z_np = _to_numpy(z_geo)
        K_np = _to_numpy(K_chart).astype(int)
        ep_sub = episode_ids[idx]
        tl_sub = task_labels[idx]

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from fragile.learning.vla.visualize import (
            plot_chart_task_alignment,
            plot_chart_transitions,
            plot_dynamics_trajectory,
        )

        # Chart transitions
        try:
            fig_trans = plot_chart_transitions(K_np, ep_sub)
            buf = io.BytesIO()
            fig_trans.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig_trans)
            buf.seek(0)
            dynamics_pane.append(pn.pane.Markdown("### Chart Transition Matrix"))
            dynamics_pane.append(pn.pane.PNG(buf.getvalue(), width=500))
        except Exception:
            logger.warning("Chart transitions error: %s", traceback.format_exc())

        # Chart-task alignment
        try:
            fig_align = plot_chart_task_alignment(K_np, tl_sub)
            buf = io.BytesIO()
            fig_align.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig_align)
            buf.seek(0)
            dynamics_pane.append(pn.pane.Markdown("### Chart-Task Alignment"))
            dynamics_pane.append(pn.pane.PNG(buf.getvalue(), width=500))
        except Exception:
            logger.warning("Chart-task alignment error: %s", traceback.format_exc())

        # Dynamics trajectory (pick first episode with enough steps)
        try:
            unique_eps = np.unique(ep_sub)
            for ep_id in unique_eps[:3]:
                mask = ep_sub == ep_id
                if mask.sum() < 4:
                    continue
                z_ep = z_np[mask]
                # Predict one-step forward via world model
                z_t = torch.from_numpy(z_ep[:-1]).float()
                # Use zero actions for visualization
                dummy_actions = torch.zeros(z_t.shape[0], loaded.args.get("action_dim", 6))
                K_t = torch.from_numpy(K_np[mask][:-1]).long()
                with torch.no_grad():
                    wm_out = loaded.world_model(z_t, dummy_actions, K_t)
                z_pred = _to_numpy(wm_out["z_next"])
                z_target = z_ep[1:]

                fig_traj = plot_dynamics_trajectory(z_pred, z_target,
                                                    title=f"Episode {ep_id}")
                buf = io.BytesIO()
                fig_traj.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                plt.close(fig_traj)
                buf.seek(0)
                dynamics_pane.append(pn.pane.Markdown(f"### Trajectory: Episode {ep_id}"))
                dynamics_pane.append(pn.pane.PNG(buf.getvalue(), width=500))
                break  # Just show one trajectory
        except Exception:
            logger.warning("Dynamics trajectory error: %s", traceback.format_exc())

    # ---- Wire callbacks ----
    scan_btn.on_click(_on_scan)
    load_btn.on_click(_on_load)

    # Refresh on widget changes
    for w in (color_by, point_size, latent_samples, seed_input,
              show_latents, show_chart_centers, show_code_centers,
              show_tree_lines, tree_line_color, tree_line_width):
        w.param.watch(lambda _: _refresh_latent(), "value")

    n_samples.param.watch(lambda _: _refresh_recon(), "value")

    # ---- Template ----
    template = pn.template.FastListTemplate(
        title="VLA World Model Dashboard",
        sidebar=[sidebar],
        main=[tabs],
    )
    return template
