"""Reusable GEVP diagnostics widgets and update logic for dashboard tabs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import torch


COMPANION_SUFFIX = "_companion"
SPATIAL_PREFIX = "spatial_"
SPATIAL_CANONICAL_LABEL_MAP: dict[str, str] = {
    "spatial_nucleon_score_abs": "nucleon",
    "spatial_pseudoscalar_score_directed": "pseudoscalar",
}


def is_companion_channel(name: str) -> bool:
    return str(name).endswith(COMPANION_SUFFIX)


def base_channel_name(name: str) -> str:
    raw = str(name)
    if raw.endswith(COMPANION_SUFFIX):
        return raw[: -len(COMPANION_SUFFIX)]
    if raw.startswith(SPATIAL_PREFIX):
        return raw[len(SPATIAL_PREFIX) :]
    return raw


def display_channel_name(raw_name: str) -> str:
    base = base_channel_name(raw_name)
    if is_companion_channel(raw_name):
        return base
    spatial_name = f"{SPATIAL_PREFIX}{base}"
    return SPATIAL_CANONICAL_LABEL_MAP.get(spatial_name, spatial_name)


def is_companion_original_result(name: str, result: Any) -> bool:
    if is_companion_channel(name):
        return True
    mass_fit = getattr(result, "mass_fit", None)
    if not isinstance(mass_fit, dict):
        return False
    if str(mass_fit.get("source", "")).strip().lower() == "original_companion":
        return True
    base_channel = str(mass_fit.get("base_channel", "")).strip()
    return is_companion_channel(base_channel)


def display_channel_name_for_original(raw_name: str, result: Any) -> str:
    raw = str(raw_name)
    if raw.startswith(SPATIAL_PREFIX) or raw.endswith(COMPANION_SUFFIX):
        return display_channel_name(raw)
    base = base_channel_name(raw)
    if is_companion_original_result(raw, result):
        return base
    spatial_name = f"{SPATIAL_PREFIX}{base}"
    return SPATIAL_CANONICAL_LABEL_MAP.get(spatial_name, spatial_name)


def channel_family_key(name: str) -> str:
    """Map operator names to a channel family used in redundancy analysis."""
    base = base_channel_name(str(name))
    for prefix in (
        "nucleon",
        "pseudoscalar",
        "scalar",
        "vector",
        "axial_vector",
        "glueball",
        "tensor",
    ):
        if base == prefix or base.startswith(f"{prefix}_"):
            return prefix
    if "_" in base:
        return base.split("_", maxsplit=1)[0]
    return base


def _extract_n_valid_windows(result: Any) -> int:
    mass_fit = getattr(result, "mass_fit", None)
    if isinstance(mass_fit, dict):
        raw = mass_fit.get("n_valid_windows", None)
        if raw is not None:
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                pass

    window_masses = getattr(result, "window_masses", None)
    if isinstance(window_masses, torch.Tensor):
        if int(window_masses.numel()) <= 0:
            return 0
        return int(torch.isfinite(window_masses).sum().item())
    if isinstance(window_masses, list | tuple):
        count = 0
        for value in window_masses:
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                count += 1
        return count
    return 0


def _entry_filter_reason(
    entry: dict[str, Any],
    *,
    min_r2: float,
    min_windows: int,
    max_error_pct: float,
    remove_artifacts: bool,
) -> str | None:
    reasons: list[str] = []
    mass = float(entry.get("mass", float("nan")))
    r2 = float(entry.get("r2", float("nan")))
    n_windows = max(0, int(entry.get("n_valid_windows", 0)))
    mass_error = float(entry.get("mass_error", float("nan")))

    if math.isfinite(min_r2):
        if not math.isfinite(r2) or r2 < min_r2:
            r2_text = "nan" if not math.isfinite(r2) else f"{r2:.3g}"
            reasons.append(f"r2={r2_text}<{min_r2:.3g}")
    if n_windows < min_windows:
        reasons.append(f"n_windows={n_windows}<{min_windows}")
    if math.isfinite(max_error_pct) and max_error_pct >= 0:
        if math.isfinite(mass) and mass > 0 and math.isfinite(mass_error) and mass_error >= 0:
            err_pct = abs(mass_error / mass) * 100.0
        else:
            err_pct = float("inf")
        if err_pct > max_error_pct:
            err_text = f"{err_pct:.3g}" if math.isfinite(err_pct) else "inf"
            reasons.append(f"err_pct={err_text}>{max_error_pct:.3g}")
    if remove_artifacts:
        if not math.isfinite(mass_error):
            reasons.append("mass_error=nan_or_inf")
        elif mass_error == 0.0:
            reasons.append("mass_error==0")
        if math.isfinite(mass) and mass == 0.0:
            reasons.append("mass==0")

    if reasons:
        return ", ".join(reasons)
    return None


def apply_gevp_entry_filters(
    entries: list[dict[str, Any]],
    *,
    min_r2: float,
    min_windows: int,
    max_error_pct: float,
    remove_artifacts: bool,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Filter entries using the same quality gates as the GEVP basis builder."""
    kept: list[dict[str, Any]] = []
    excluded: list[tuple[str, str]] = []
    for entry in entries:
        reason = _entry_filter_reason(
            entry,
            min_r2=min_r2,
            min_windows=min_windows,
            max_error_pct=max_error_pct,
            remove_artifacts=remove_artifacts,
        )
        if reason is None:
            kept.append(entry)
        else:
            excluded.append((str(entry.get("operator_label", "unknown")), reason))
    return kept, excluded


def collect_family_operator_entries(
    per_scale_results: Mapping[str, Sequence[Any]],
    family: str,
    *,
    original_results: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Collect multiscale + original operator entries for one family."""
    entries: list[dict[str, Any]] = []

    for raw_name, results_per_scale in per_scale_results.items():
        raw_name_s = str(raw_name)
        if channel_family_key(raw_name_s) != family:
            continue
        display_name = display_channel_name(raw_name_s)
        for scale_idx, result in enumerate(results_per_scale):
            if result is None or int(getattr(result, "n_samples", 0)) <= 0:
                continue
            series = getattr(result, "series", None)
            if series is None or int(series.numel()) <= 0:
                continue
            mass_fit = getattr(result, "mass_fit", {})
            entries.append({
                "operator_label": f"{display_name}@s{int(scale_idx)}",
                "source_channel": raw_name_s,
                "scale_label": f"s{int(scale_idx)}",
                "series": series.detach().float(),
                "mass": float(mass_fit.get("mass", float("nan"))),
                "mass_error": float(mass_fit.get("mass_error", float("nan"))),
                "r2": float(mass_fit.get("r_squared", float("nan"))),
                "n_valid_windows": _extract_n_valid_windows(result),
                "n_samples": int(getattr(result, "n_samples", 0)),
                "source_kind": "multiscale",
            })

    if isinstance(original_results, Mapping):
        for raw_name, result in original_results.items():
            raw_name_s = str(raw_name)
            if raw_name_s.endswith("_multiscale_best"):
                continue
            if channel_family_key(raw_name_s) != family:
                continue
            if result is None or int(getattr(result, "n_samples", 0)) <= 0:
                continue
            series = getattr(result, "series", None)
            if series is None or int(series.numel()) <= 0:
                continue
            display_name = display_channel_name_for_original(raw_name_s, result)
            mass_fit = getattr(result, "mass_fit", {})
            entries.append({
                "operator_label": f"{display_name}@full",
                "source_channel": raw_name_s,
                "scale_label": "full_original_no_threshold",
                "series": series.detach().float(),
                "mass": float(mass_fit.get("mass", float("nan"))),
                "mass_error": float(mass_fit.get("mass_error", float("nan"))),
                "r2": float(mass_fit.get("r_squared", float("nan"))),
                "n_valid_windows": _extract_n_valid_windows(result),
                "n_samples": int(getattr(result, "n_samples", 0)),
                "source_kind": "original",
            })

    if not entries:
        return entries

    label_counts: dict[str, int] = {}
    for entry in entries:
        label = str(entry["operator_label"])
        count = label_counts.get(label, 0)
        if count > 0:
            entry["operator_label"] = f"{label}#{count + 1}"
        label_counts[label] = count + 1
    return entries


def analyze_operator_quality_vectorized(
    entries: list[dict[str, Any]],
    *,
    t0: int = 2,
    eig_rel_cutoff: float = 1e-2,
    importance_cutoff: float = 0.05,
    use_connected: bool = True,
) -> dict[str, Any] | None:
    """Analyze operator redundancy and quality with vectorized torch ops."""
    if len(entries) < 2:
        return None

    lengths = [int(entry["series"].numel()) for entry in entries]
    t_len = min(lengths)
    k_count = len(entries)
    if t_len <= 4 or k_count < 2:
        return None

    t0_eff = max(1, int(t0))
    if t0_eff >= t_len - 1:
        t0_eff = max(1, (t_len - 1) // 2)
    if t_len - t0_eff <= 1:
        return None

    series = torch.stack([entry["series"][:t_len] for entry in entries], dim=0).float()  # [K,T]
    if use_connected:
        series = series - series.mean(dim=1, keepdim=True)

    source = series[:, : t_len - t0_eff]
    sink = series[:, t0_eff:]
    c0 = (sink @ source.transpose(0, 1)) / float(t_len - t0_eff)
    c0 = 0.5 * (c0 + c0.transpose(0, 1))

    evals, evecs = torch.linalg.eigh(c0)
    sort_idx = torch.argsort(evals, descending=True)
    evals = evals[sort_idx].real.float()
    evecs = evecs[:, sort_idx].real.float()

    eps = torch.tensor(1e-12, dtype=torch.float32, device=series.device)
    positive = evals > 0
    if torch.any(positive):
        max_eval = torch.clamp_min(evals[positive][0], eps)
        sig_mask = evals > (float(eig_rel_cutoff) * max_eval)
        rel_eval = evals / max_eval
    else:
        max_abs = torch.clamp_min(evals.abs().max(), eps)
        sig_mask = evals.abs() > (float(eig_rel_cutoff) * max_abs)
        rel_eval = evals / max_abs

    n_significant = int(sig_mask.sum().item())
    if n_significant <= 0:
        sig_mask[0] = True
        n_significant = 1

    pos_sig = evals[sig_mask & (evals > 0)]
    if pos_sig.numel() >= 1:
        cond_c0 = float(pos_sig.max().item() / max(float(pos_sig.min().item()), 1e-12))
    else:
        cond_c0 = float("nan")

    weights = torch.where(sig_mask, torch.clamp_min(evals, 0.0), torch.zeros_like(evals))
    if float(weights.sum().item()) <= 0:
        weights = torch.where(sig_mask, evals.abs(), torch.zeros_like(evals))
    importance = (evecs.square() * weights.unsqueeze(0)).sum(dim=1)
    importance = importance / torch.clamp_min(importance.max(), eps)

    active_idx = torch.nonzero(sig_mask, as_tuple=False).flatten()
    active_vecs = torch.abs(evecs[:, active_idx])
    dom_local = torch.argmax(active_vecs, dim=1)
    dom_mode = active_idx[dom_local]
    dom_loading = active_vecs[torch.arange(k_count, device=series.device), dom_local]

    mass = torch.tensor(
        [float(entry["mass"]) for entry in entries], dtype=torch.float32, device=series.device
    )
    mass_error = torch.tensor(
        [float(entry["mass_error"]) for entry in entries],
        dtype=torch.float32,
        device=series.device,
    )
    r2 = torch.tensor(
        [float(entry["r2"]) for entry in entries], dtype=torch.float32, device=series.device
    )
    n_samples = torch.tensor(
        [int(entry["n_samples"]) for entry in entries],
        dtype=torch.float32,
        device=series.device,
    )

    mass_valid = torch.isfinite(mass) & (mass > 0)
    err_valid = torch.isfinite(mass_error) & (mass_error >= 0)
    err_pct = torch.full_like(mass, float("inf"))
    err_pct = torch.where(mass_valid & err_valid, (mass_error / mass).abs() * 100.0, err_pct)
    r2_score = torch.where(
        torch.isfinite(r2), torch.clamp((r2 + 1.0) / 2.0, 0.0, 1.0), torch.zeros_like(r2)
    )
    err_score = torch.exp(-torch.clamp(err_pct, min=0.0, max=500.0) / 100.0)
    sample_score = torch.log1p(torch.clamp_min(n_samples, 0.0))
    sample_score = sample_score / torch.clamp_min(sample_score.max(), eps)

    quality_score = 0.55 * importance + 0.20 * r2_score + 0.15 * err_score + 0.10 * sample_score

    redundant_candidate = importance < float(importance_cutoff)
    low_signal = (~mass_valid) | (err_pct > 100.0) | (~torch.isfinite(r2)) | (r2 < 0.0)
    suggestion_code = torch.where(
        low_signal,
        torch.full_like(importance, 2, dtype=torch.int64),
        torch.where(
            redundant_candidate,
            torch.full_like(importance, 1, dtype=torch.int64),
            torch.zeros_like(importance, dtype=torch.int64),
        ),
    )

    suggestions = []
    for code in suggestion_code.tolist():
        if int(code) == 0:
            suggestions.append("keep_candidate")
        elif int(code) == 1:
            suggestions.append("redundancy_candidate")
        else:
            suggestions.append("low_signal_candidate")

    rows: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        rows.append({
            "operator": entry["operator_label"],
            "source_channel": entry["source_channel"],
            "scale_label": entry["scale_label"],
            "source_kind": entry["source_kind"],
            "mass": float(mass[idx].item()) if torch.isfinite(mass[idx]) else float("nan"),
            "mass_error": (
                float(mass_error[idx].item()) if torch.isfinite(mass_error[idx]) else float("nan")
            ),
            "mass_error_pct": (
                float(err_pct[idx].item()) if torch.isfinite(err_pct[idx]) else float("nan")
            ),
            "r2": float(r2[idx].item()) if torch.isfinite(r2[idx]) else float("nan"),
            "n_valid_windows": int(entry.get("n_valid_windows", 0)),
            "n_samples": int(entry["n_samples"]),
            "importance": float(importance[idx].item()),
            "quality_score": float(quality_score[idx].item()),
            "dominant_mode": int(dom_mode[idx].item()),
            "dominant_loading": float(dom_loading[idx].item()),
            "suggestion": suggestions[idx],
        })

    order = torch.argsort(importance, descending=True)
    ranked_ops = [entries[int(i)]["operator_label"] for i in order.tolist()]

    return {
        "rows": rows,
        "eigenvalues": evals.detach().cpu().numpy(),
        "relative_eigenvalues": rel_eval.detach().cpu().numpy(),
        "significant_mask": sig_mask.detach().cpu().numpy().astype(bool, copy=False),
        "correlation_matrix": c0.detach().cpu().numpy(),
        "operator_labels": [str(entry["operator_label"]) for entry in entries],
        "n_significant": int(n_significant),
        "cond_c0": float(cond_c0),
        "n_operators": int(k_count),
        "t_len": int(t_len),
        "t0": int(t0_eff),
        "eig_rel_cutoff": float(eig_rel_cutoff),
        "importance_cutoff": float(importance_cutoff),
        "ranked_operators": ranked_ops,
    }


def build_eigenspectrum_plot(
    analysis: dict[str, Any],
    *,
    family_label: str,
) -> hv.Overlay | None:
    """Create the eigenvalue spectrum visualization."""
    evals = np.asarray(analysis.get("eigenvalues", []), dtype=float)
    rel = np.asarray(analysis.get("relative_eigenvalues", []), dtype=float)
    sig = np.asarray(analysis.get("significant_mask", []), dtype=bool)
    if evals.size == 0 or rel.size == 0:
        return None

    mode_idx = np.arange(evals.size, dtype=int)
    rel_abs = np.abs(rel)
    cutoff = float(analysis.get("eig_rel_cutoff", 1e-2))

    df = pd.DataFrame({
        "mode": mode_idx,
        "rel_abs": rel_abs,
        "eigenvalue": evals,
        "significant": np.where(sig, "significant", "candidate_redundant"),
    })
    bars = hv.Bars(df, kdims=["mode"], vdims=["rel_abs", "eigenvalue", "significant"]).opts(
        width=900,
        height=300,
        xlabel="mode index (sorted by eigenvalue)",
        ylabel="|eigenvalue| / reference",
        title=f"Operator Spectrum: {family_label}",
        color="significant",
        cmap=["#e45756", "#4c78a8"],
        tools=["hover"],
        show_grid=True,
    )
    line = hv.Curve(
        (
            np.asarray([0, max(1, evals.size - 1)], dtype=float),
            np.asarray([cutoff, cutoff], dtype=float),
        ),
        kdims=["mode"],
        vdims=["threshold"],
    ).opts(color="#f58518", line_width=2, line_dash="dashed")
    return (bars * line).opts(legend_position="top_right")


def build_kept_eigenvalue_table(analysis: dict[str, Any]) -> pd.DataFrame:
    """Create a table for significant/kept GEVP eigen-modes and ratios."""
    evals = np.asarray(analysis.get("eigenvalues", []), dtype=float)
    rel = np.asarray(analysis.get("relative_eigenvalues", []), dtype=float)
    sig = np.asarray(analysis.get("significant_mask", []), dtype=bool)
    if evals.size == 0 or rel.size == 0 or sig.size == 0:
        return pd.DataFrame()

    n = min(evals.size, rel.size, sig.size)
    evals = evals[:n]
    rel = rel[:n]
    sig = sig[:n]
    kept_idx = np.flatnonzero(sig)
    if kept_idx.size == 0:
        return pd.DataFrame()

    top_mode = int(kept_idx[0])
    top_eval = float(evals[top_mode])
    rows: list[dict[str, float | int]] = []
    prev_eval = float("nan")
    for kept_rank, mode in enumerate(kept_idx, start=1):
        eig = float(evals[int(mode)])
        ratio_to_top = float("nan")
        if math.isfinite(top_eval) and top_eval != 0.0 and math.isfinite(eig):
            ratio_to_top = eig / top_eval
        ratio_to_prev = float("nan")
        if kept_rank > 1 and math.isfinite(prev_eval) and prev_eval != 0.0 and math.isfinite(eig):
            ratio_to_prev = eig / prev_eval
        rows.append({
            "kept_rank": int(kept_rank),
            "mode": int(mode),
            "eigenvalue": eig,
            "abs_rel_to_top": float(abs(rel[int(mode)])),
            "ratio_to_top": ratio_to_top,
            "ratio_to_prev_kept": ratio_to_prev,
        })
        prev_eval = eig
    return pd.DataFrame(rows)


def build_correlation_matrix_heatmap(
    analysis: dict[str, Any],
    *,
    family_label: str,
) -> hv.HeatMap | None:
    """Create a heatmap for the operator correlation matrix C(t0)."""
    matrix = np.asarray(analysis.get("correlation_matrix", []), dtype=float)
    labels = list(analysis.get("operator_labels", []))
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        return None

    n = int(matrix.shape[0])
    rows: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(n):
            rows.append({
                "mode_i": int(i),
                "mode_j": int(j),
                "operator_i": labels[i] if i < len(labels) else f"op{i}",
                "operator_j": labels[j] if j < len(labels) else f"op{j}",
                "corr": float(matrix[i, j])
                if math.isfinite(float(matrix[i, j]))
                else float("nan"),
            })
    df = pd.DataFrame(rows)

    finite = matrix[np.isfinite(matrix)]
    abs_max = float(np.nanmax(np.abs(finite))) if finite.size > 0 else 1.0
    abs_max = max(abs_max, 1e-6)
    return hv.HeatMap(
        df,
        kdims=["mode_j", "mode_i"],
        vdims=[("corr", "C(t0)"), "operator_i", "operator_j"],
    ).opts(
        width=900,
        height=340,
        xlabel="Mode j",
        ylabel="Mode i",
        title=f"GEVP Correlation Matrix C(t0): {family_label}",
        cmap="RdBu_r",
        clim=(-abs_max, abs_max),
        colorbar=True,
        tools=["hover"],
        show_grid=True,
    )


@dataclass
class GEVPDashboardWidgets:
    """Panel widgets for the reusable GEVP diagnostics section."""

    summary: pn.pane.Markdown
    eigenspectrum_plot: pn.pane.HoloViews
    eigenvalue_table: pn.widgets.Tabulator
    correlation_matrix_heatmap: pn.pane.HoloViews
    operator_quality_table: pn.widgets.Tabulator


def create_gevp_dashboard_widgets() -> GEVPDashboardWidgets:
    """Create GEVP diagnostics widgets."""
    return GEVPDashboardWidgets(
        summary=pn.pane.Markdown(
            "**Operator quality report:** select a channel to inspect eigen-spectrum and quality metrics.",
            sizing_mode="stretch_width",
        ),
        eigenspectrum_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        ),
        eigenvalue_table=pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        ),
        correlation_matrix_heatmap=pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        ),
        operator_quality_table=pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        ),
    )


def build_gevp_dashboard_sections(w: GEVPDashboardWidgets) -> list[Any]:
    """Return reusable GEVP diagnostics section blocks for tab composition."""
    return [
        pn.layout.Divider(),
        pn.pane.Markdown("### GEVP Eigenvalue Spectrum"),
        pn.pane.Markdown(
            "_Computed from the channel-family operator correlation matrix C(t₀). "
            "This is diagnostics/report only; no operators are removed automatically._"
        ),
        w.summary,
        w.eigenspectrum_plot,
        pn.pane.Markdown("### Kept GEVP Modes"),
        pn.pane.Markdown(
            "_Significant eigen-modes retained by the relative-eigenvalue cutoff. "
            "Ratios help gauge spectral separation._"
        ),
        w.eigenvalue_table,
        pn.pane.Markdown("### Operator Correlation Matrix"),
        w.correlation_matrix_heatmap,
        pn.pane.Markdown("### Operator Quality Report"),
        w.operator_quality_table,
    ]


def clear_gevp_dashboard(w: GEVPDashboardWidgets) -> None:
    """Reset GEVP diagnostics widgets."""
    w.summary.object = "**Operator quality report:** select a channel to inspect eigen-spectrum and quality metrics."
    w.eigenspectrum_plot.object = None
    w.eigenvalue_table.value = pd.DataFrame()
    w.correlation_matrix_heatmap.object = None
    w.operator_quality_table.value = pd.DataFrame()


def extract_final_gevp_mass(
    family_key: str,
    companion_gevp_results: Mapping[str, Any] | None,
) -> tuple[float | None, float]:
    """Extract final GEVP mass/error for a family from companion results, if present."""
    if not isinstance(companion_gevp_results, Mapping):
        return None, float("nan")
    gevp_channel_name = f"{family_key}_gevp"
    gevp_candidate = companion_gevp_results.get(gevp_channel_name)
    if gevp_candidate is None:
        return None, float("nan")
    gevp_fit = getattr(gevp_candidate, "mass_fit", None)
    if not isinstance(gevp_fit, dict):
        return None, float("nan")
    gm = float(gevp_fit.get("mass", float("nan")))
    if not (math.isfinite(gm) and gm > 0):
        return None, float("nan")
    return gm, float(gevp_fit.get("mass_error", float("nan")))


def update_gevp_dashboard(
    w: GEVPDashboardWidgets,
    *,
    selected_channel_name: str,
    raw_channel_name: str,
    per_scale_results: Mapping[str, Sequence[Any]],
    original_results: Mapping[str, Any] | None,
    companion_gevp_results: Mapping[str, Any] | None,
    min_r2: float,
    min_windows: int,
    max_error_pct: float,
    remove_artifacts: bool,
    selected_families: Sequence[str] | None = None,
    selected_scale_labels: Sequence[str] | None = None,
    t0: int = 2,
    eig_rel_cutoff: float = 1e-2,
    importance_cutoff: float = 0.05,
    use_connected: bool = True,
) -> dict[str, Any]:
    """Update reusable GEVP dashboard section for the selected channel."""
    default_family = channel_family_key(raw_channel_name)
    available_families = sorted({channel_family_key(str(name)) for name in per_scale_results})
    families = [str(fam) for fam in (selected_families or []) if str(fam) in available_families]
    if not families:
        families = [default_family]
    family_scope = ", ".join(families)

    selected_scale_set = {str(scale) for scale in (selected_scale_labels or []) if str(scale)}
    if selected_scale_set:
        scale_scope = ", ".join(sorted(selected_scale_set))
    else:
        scale_scope = "all"

    quality_entries_raw: list[dict[str, Any]] = []
    for family_key in families:
        quality_entries_raw.extend(
            collect_family_operator_entries(
                per_scale_results,
                family_key,
                original_results=original_results
                if isinstance(original_results, Mapping)
                else None,
            )
        )
    if selected_scale_set:
        quality_entries_raw = [
            entry
            for entry in quality_entries_raw
            if str(entry.get("scale_label", "")) in selected_scale_set
        ]

    if len(families) == 1:
        gevp_mass, gevp_error = extract_final_gevp_mass(families[0], companion_gevp_results)
    else:
        gevp_mass, gevp_error = None, float("nan")
    quality_entries, filtered_out = apply_gevp_entry_filters(
        quality_entries_raw,
        min_r2=min_r2,
        min_windows=min_windows,
        max_error_pct=max_error_pct,
        remove_artifacts=remove_artifacts,
    )
    min_r2_str = f"{min_r2:.3g}" if math.isfinite(min_r2) else "off"
    max_error_pct_str = f"{max_error_pct:.3g}%" if math.isfinite(max_error_pct) else "off"
    filter_header = (
        f"`R² >= {min_r2_str}`, `n_windows >= {min_windows}`, "
        f"`error % <= {max_error_pct_str}`, "
        f"`remove artifacts={'on' if remove_artifacts else 'off'}`"
    )

    quality_analysis = analyze_operator_quality_vectorized(
        quality_entries,
        t0=t0,
        eig_rel_cutoff=eig_rel_cutoff,
        importance_cutoff=importance_cutoff,
        use_connected=use_connected,
    )
    if quality_analysis is None:
        if not quality_entries and filtered_out:
            preview = ", ".join(f"{label}({reason})" for label, reason in filtered_out[:4])
            suffix = " ..." if len(filtered_out) > 4 else ""
            w.summary.object = (
                f"**{selected_channel_name} (families: {family_scope}):** "
                "no operators satisfy active GEVP filters.  \n"
                f"_Scope:_ `scales={scale_scope}`.  \n"
                f"_Filters:_ {filter_header}.  \n"
                f"_Excluded:_ `{len(filtered_out)}` operators (`{preview}{suffix}`)."
            )
        else:
            w.summary.object = (
                f"**{selected_channel_name} (families: {family_scope}):** "
                "insufficient operator-series data for "
                "eigen-spectrum analysis after applying active GEVP filters.  \n"
                f"_Scope:_ `scales={scale_scope}`.  \n"
                f"_Filters:_ {filter_header}.  \n"
                f"_Shown:_ `{len(quality_entries)}` operators; _Excluded:_ `{len(filtered_out)}`."
            )
        if gevp_mass is not None and math.isfinite(gevp_mass) and gevp_mass > 0:
            gevp_line = f"- Final GEVP mass: `{gevp_mass:.6g}`"
            if math.isfinite(gevp_error) and gevp_error > 0:
                gevp_line += f" ± `{gevp_error:.2g}`"
            w.summary.object += f"  \n{gevp_line}"
        w.eigenspectrum_plot.object = None
        w.eigenvalue_table.value = pd.DataFrame()
        w.correlation_matrix_heatmap.object = None
        w.operator_quality_table.value = (
            pd.DataFrame([
                {
                    "operator": entry["operator_label"],
                    "source_channel": entry["source_channel"],
                    "scale_label": entry["scale_label"],
                    "source_kind": entry["source_kind"],
                    "mass": entry["mass"],
                    "mass_error": entry["mass_error"],
                    "r2": entry["r2"],
                    "n_valid_windows": int(entry.get("n_valid_windows", 0)),
                    "n_samples": entry["n_samples"],
                    "suggestion": "insufficient_data",
                }
                for entry in quality_entries
            ])
            if quality_entries
            else pd.DataFrame()
        )
    else:
        n_ops = int(quality_analysis.get("n_operators", 0))
        n_sig = int(quality_analysis.get("n_significant", 0))
        cond_c0 = float(quality_analysis.get("cond_c0", float("nan")))
        cond_str = f"{cond_c0:.3g}" if math.isfinite(cond_c0) else "n/a"
        top_ops = ", ".join(list(quality_analysis.get("ranked_operators", []))[:3])
        if not top_ops:
            top_ops = "n/a"
        w.summary.object = (
            f"**{selected_channel_name} (families: {family_scope}) operator report:** "
            f"`{n_ops}` operators analyzed, `{n_sig}` significant eigen-modes, "
            f"`{max(0, n_ops - n_sig)}` redundancy candidates, "
            f"`cond(C(t0))={cond_str}`.  \n"
            f"_Scope:_ `scales={scale_scope}`.  \n"
            f"_Filters:_ {filter_header}.  \n"
            f"_Shown:_ `{len(quality_entries)}` operators; _Excluded:_ `{len(filtered_out)}`.  \n"
            f"_Top operators by importance:_ `{top_ops}`.  \n"
            "_Table includes only operators passing active GEVP-quality filters._"
        )
        if gevp_mass is not None and math.isfinite(gevp_mass) and gevp_mass > 0:
            gevp_line = f"- Final GEVP mass: `{gevp_mass:.6g}`"
            if math.isfinite(gevp_error) and gevp_error > 0:
                gevp_line += f" ± `{gevp_error:.2g}`"
            w.summary.object += f"  \n{gevp_line}"
        qdf = pd.DataFrame(quality_analysis.get("rows", []))
        if not qdf.empty:
            qdf = qdf.sort_values(
                ["quality_score", "importance", "operator"],
                ascending=[False, False, True],
                kind="stable",
            )
        w.operator_quality_table.value = qdf
        w.eigenspectrum_plot.object = build_eigenspectrum_plot(
            quality_analysis,
            family_label=family_scope,
        )
        w.eigenvalue_table.value = build_kept_eigenvalue_table(quality_analysis)
        w.correlation_matrix_heatmap.object = build_correlation_matrix_heatmap(
            quality_analysis,
            family_label=family_scope,
        )

    return {
        "family_key": default_family,
        "selected_families": families,
        "selected_scales": sorted(selected_scale_set),
        "quality_analysis": quality_analysis,
        "quality_entries": quality_entries,
        "filtered_out": filtered_out,
        "filter_header": filter_header,
        "gevp_mass": gevp_mass,
        "gevp_error": gevp_error,
    }
