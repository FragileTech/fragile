"""Multiscale tab: widgets, layout, and update logic extracted from the dashboard closure."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import torch

from fragile.fractalai.qft.multiscale_analysis import (
    analyze_channel_across_scales,
    build_estimator_table_rows,
    build_pairwise_table_rows,
    format_consensus_summary,
)
from fragile.fractalai.qft.multiscale_strong_force import MultiscaleStrongForceOutput
from fragile.fractalai.qft.operator_analysis import (
    build_consensus_plot,
    build_mass_vs_scale_plot,
    build_multiscale_correlator_plot,
    build_multiscale_effective_mass_plot,
    build_per_scale_channel_plots,
)
from fragile.fractalai.qft.smeared_operators import (
    compute_pairwise_distance_matrices_from_history,
)


COMPANION_SUFFIX = "_companion"
SPATIAL_PREFIX = "spatial_"
SPATIAL_CANONICAL_LABEL_MAP: dict[str, str] = {
    "spatial_nucleon_score_abs": "nucleon",
    "spatial_pseudoscalar_score_directed": "pseudoscalar",
}


def _is_companion_channel(name: str) -> bool:
    return str(name).endswith(COMPANION_SUFFIX)


def _base_channel_name(name: str) -> str:
    raw = str(name)
    if raw.endswith(COMPANION_SUFFIX):
        return raw[: -len(COMPANION_SUFFIX)]
    if raw.startswith(SPATIAL_PREFIX):
        return raw[len(SPATIAL_PREFIX) :]
    return raw


def _display_channel_name(raw_name: str) -> str:
    base = _base_channel_name(raw_name)
    if _is_companion_channel(raw_name):
        return base
    spatial_name = f"{SPATIAL_PREFIX}{base}"
    return SPATIAL_CANONICAL_LABEL_MAP.get(spatial_name, spatial_name)


def _display_channel_name_for_original(raw_name: str, result: Any) -> str:
    raw = str(raw_name)
    if raw.startswith(SPATIAL_PREFIX) or raw.endswith(COMPANION_SUFFIX):
        return _display_channel_name(raw)
    base = _base_channel_name(raw)
    if _is_companion_original_result(raw, result):
        return base
    spatial_name = f"{SPATIAL_PREFIX}{base}"
    return SPATIAL_CANONICAL_LABEL_MAP.get(spatial_name, spatial_name)


def _is_companion_original_result(name: str, result: Any) -> bool:
    if _is_companion_channel(name):
        return True
    mass_fit = getattr(result, "mass_fit", None)
    if not isinstance(mass_fit, dict):
        return False
    if str(mass_fit.get("source", "")).strip().lower() == "original_companion":
        return True
    base_channel = str(mass_fit.get("base_channel", "")).strip()
    return _is_companion_channel(base_channel)


def _channel_family_key(name: str) -> str:
    """Map operator names to a channel family used in redundancy analysis."""
    base = _base_channel_name(str(name))
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
    """Extract valid fit-window count from a channel result."""
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
    """Return exclusion reason if an entry fails GEVP-quality filters."""
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


def _apply_gevp_entry_filters(
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


def _collect_family_operator_entries(
    output: MultiscaleStrongForceOutput,
    family: str,
    *,
    original_results: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Collect multiscale + original operator entries for one family."""
    entries: list[dict[str, Any]] = []

    for raw_name, results_per_scale in output.per_scale_results.items():
        raw_name_s = str(raw_name)
        if _channel_family_key(raw_name_s) != family:
            continue
        display_name = _display_channel_name(raw_name_s)
        for scale_idx, result in enumerate(results_per_scale):
            if result is None or int(getattr(result, "n_samples", 0)) <= 0:
                continue
            series = getattr(result, "series", None)
            if series is None or int(series.numel()) <= 0:
                continue
            mass_fit = getattr(result, "mass_fit", {})
            entries.append(
                {
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
                }
            )

    if isinstance(original_results, dict):
        for raw_name, result in original_results.items():
            raw_name_s = str(raw_name)
            if raw_name_s.endswith("_multiscale_best"):
                continue
            if _channel_family_key(raw_name_s) != family:
                continue
            if result is None or int(getattr(result, "n_samples", 0)) <= 0:
                continue
            series = getattr(result, "series", None)
            if series is None or int(series.numel()) <= 0:
                continue
            display_name = _display_channel_name_for_original(raw_name_s, result)
            mass_fit = getattr(result, "mass_fit", {})
            entries.append(
                {
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
                }
            )

    if not entries:
        return entries

    # Deduplicate labels while preserving all operators by adding suffixes.
    label_counts: dict[str, int] = {}
    for entry in entries:
        label = str(entry["operator_label"])
        count = label_counts.get(label, 0)
        if count > 0:
            entry["operator_label"] = f"{label}#{count + 1}"
        label_counts[label] = count + 1
    return entries


def _analyze_operator_quality_vectorized(
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

    mass = torch.tensor([float(entry["mass"]) for entry in entries], dtype=torch.float32, device=series.device)
    mass_error = torch.tensor(
        [float(entry["mass_error"]) for entry in entries],
        dtype=torch.float32,
        device=series.device,
    )
    r2 = torch.tensor([float(entry["r2"]) for entry in entries], dtype=torch.float32, device=series.device)
    n_samples = torch.tensor(
        [int(entry["n_samples"]) for entry in entries],
        dtype=torch.float32,
        device=series.device,
    )

    mass_valid = torch.isfinite(mass) & (mass > 0)
    err_valid = torch.isfinite(mass_error) & (mass_error >= 0)
    err_pct = torch.full_like(mass, float("inf"))
    err_pct = torch.where(mass_valid & err_valid, (mass_error / mass).abs() * 100.0, err_pct)
    r2_score = torch.where(torch.isfinite(r2), torch.clamp((r2 + 1.0) / 2.0, 0.0, 1.0), torch.zeros_like(r2))
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
        rows.append(
            {
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
            }
        )

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


def _build_eigenspectrum_plot(
    analysis: dict[str, Any],
    *,
    family_label: str,
) -> hv.Overlay | None:
    """Create the eigenvalue spectrum visualization for the multiscale tab."""
    evals = np.asarray(analysis.get("eigenvalues", []), dtype=float)
    rel = np.asarray(analysis.get("relative_eigenvalues", []), dtype=float)
    sig = np.asarray(analysis.get("significant_mask", []), dtype=bool)
    if evals.size == 0 or rel.size == 0:
        return None

    mode_idx = np.arange(evals.size, dtype=int)
    rel_abs = np.abs(rel)
    cutoff = float(analysis.get("eig_rel_cutoff", 1e-2))

    df = pd.DataFrame(
        {
            "mode": mode_idx,
            "rel_abs": rel_abs,
            "eigenvalue": evals,
            "significant": np.where(sig, "significant", "candidate_redundant"),
        }
    )
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


def _build_kept_eigenvalue_table(analysis: dict[str, Any]) -> pd.DataFrame:
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
        rows.append(
            {
                "kept_rank": int(kept_rank),
                "mode": int(mode),
                "eigenvalue": eig,
                "abs_rel_to_top": float(abs(rel[int(mode)])),
                "ratio_to_top": ratio_to_top,
                "ratio_to_prev_kept": ratio_to_prev,
            }
        )
        prev_eval = eig
    return pd.DataFrame(rows)


def _build_correlation_matrix_heatmap(
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
            rows.append(
                {
                    "mode_i": int(i),
                    "mode_j": int(j),
                    "operator_i": labels[i] if i < len(labels) else f"op{i}",
                    "operator_j": labels[j] if j < len(labels) else f"op{j}",
                    "corr": float(matrix[i, j]) if math.isfinite(float(matrix[i, j])) else float("nan"),
                }
            )
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


# ---------------------------------------------------------------------------
# Dataclass holding multiscale-tab widget references
# ---------------------------------------------------------------------------


@dataclass
class MultiscaleTabWidgets:
    """All Panel widgets that belong to the dedicated Multiscale tab."""

    status: pn.pane.Markdown
    geodesic_heatmap: pn.pane.HoloViews
    geodesic_histogram: pn.pane.HoloViews
    channel_select: pn.widgets.Select
    corr_plot: pn.pane.HoloViews
    meff_plot: pn.pane.HoloViews
    mass_vs_scale_plot: pn.pane.HoloViews
    estimator_table: pn.widgets.Tabulator
    pairwise_table: pn.widgets.Tabulator
    consensus_summary: pn.pane.Markdown
    systematics_badge: pn.pane.Alert
    consensus_plot: pn.pane.HoloViews
    operator_quality_summary: pn.pane.Markdown
    eigenspectrum_plot: pn.pane.HoloViews
    eigenvalue_table: pn.widgets.Tabulator
    correlation_matrix_heatmap: pn.pane.HoloViews
    operator_quality_table: pn.widgets.Tabulator
    per_scale_plots: pn.Column


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_multiscale_widgets() -> MultiscaleTabWidgets:
    """Create and return all multiscale-tab widgets with default values."""
    return MultiscaleTabWidgets(
        status=pn.pane.Markdown(
            "**Multiscale:** enable multiscale kernels in Companion Strong Force settings, "
            "then run Compute Companion Strong Force Channels.",
            sizing_mode="stretch_width",
        ),
        geodesic_heatmap=pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        ),
        geodesic_histogram=pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        ),
        channel_select=pn.widgets.Select(
            name="Channel",
            options=["nucleon"],
            value="nucleon",
            sizing_mode="stretch_width",
        ),
        corr_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        ),
        meff_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        ),
        mass_vs_scale_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        ),
        estimator_table=pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        ),
        pairwise_table=pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        ),
        consensus_summary=pn.pane.Markdown(
            "**Scale-as-Estimator Consensus:** run multiscale analysis to populate.",
            sizing_mode="stretch_width",
        ),
        systematics_badge=pn.pane.Alert(
            "Systematics verdict: run multiscale analysis to evaluate.",
            alert_type="secondary",
            sizing_mode="stretch_width",
        ),
        consensus_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        ),
        operator_quality_summary=pn.pane.Markdown(
            "**Operator quality report:** select a channel to inspect eigen-spectrum and quality metrics.",
            sizing_mode="stretch_width",
        ),
        eigenspectrum_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        ),
        eigenvalue_table=pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        ),
        correlation_matrix_heatmap=pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        ),
        operator_quality_table=pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        ),
        per_scale_plots=pn.Column(sizing_mode="stretch_width"),
    )


# ---------------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------------


def build_multiscale_tab_layout(w: MultiscaleTabWidgets) -> pn.Column:
    """Return the ``pn.Column`` that forms the Multiscale tab content."""
    return pn.Column(
        w.status,
        pn.Row(w.channel_select, sizing_mode="stretch_width"),
        pn.pane.Markdown("### Full Geodesic Distance Matrix"),
        pn.pane.Markdown(
            "_Computed from recorded neighbor graph and selected edge-weight mode "
            "on the representative frame used by the current multiscale run._"
        ),
        w.geodesic_heatmap,
        pn.pane.Markdown("### Geodesic Distance Distribution"),
        w.geodesic_histogram,
        pn.layout.Divider(),
        pn.pane.Markdown("### Correlator Across Scales"),
        pn.pane.Markdown(
            "_One color per scale; scatter+errorbars from the same multiscale kernel family._"
        ),
        w.corr_plot,
        pn.pane.Markdown("### Effective Mass Across Scales"),
        w.meff_plot,
        pn.layout.Divider(),
        pn.pane.Markdown("### Mass vs Scale"),
        w.mass_vs_scale_plot,
        pn.layout.Divider(),
        pn.pane.Markdown("### Scale-as-Estimator Analysis"),
        w.estimator_table,
        pn.pane.Markdown("### Pairwise Discrepancies"),
        w.pairwise_table,
        pn.pane.Markdown("### Consensus / Systematics"),
        w.systematics_badge,
        w.consensus_summary,
        w.consensus_plot,
        pn.layout.Divider(),
        pn.pane.Markdown("### GEVP Eigenvalue Spectrum"),
        pn.pane.Markdown(
            "_Computed from the channel-family operator correlation matrix C(t₀). "
            "This is diagnostics/report only; no operators are removed automatically._"
        ),
        w.operator_quality_summary,
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
        pn.layout.Divider(),
        pn.pane.Markdown("### Per-Scale Channel Plots"),
        w.per_scale_plots,
        sizing_mode="stretch_both",
    )


# ---------------------------------------------------------------------------
# Clear helper
# ---------------------------------------------------------------------------


def clear_multiscale_tab(w: MultiscaleTabWidgets, status_text: str) -> None:
    """Reset every multiscale-tab widget to its empty/default state."""
    w.status.object = status_text
    w.geodesic_heatmap.object = None
    w.geodesic_histogram.object = None
    w.corr_plot.object = None
    w.meff_plot.object = None
    w.mass_vs_scale_plot.object = None
    w.estimator_table.value = pd.DataFrame()
    w.pairwise_table.value = pd.DataFrame()
    w.consensus_summary.object = (
        "**Scale-as-Estimator Consensus:** run multiscale analysis to populate."
    )
    w.systematics_badge.object = (
        "Systematics verdict: run multiscale analysis to evaluate."
    )
    w.systematics_badge.alert_type = "secondary"
    w.consensus_plot.object = None
    w.operator_quality_summary.object = (
        "**Operator quality report:** select a channel to inspect eigen-spectrum and quality metrics."
    )
    w.eigenspectrum_plot.object = None
    w.eigenvalue_table.value = pd.DataFrame()
    w.correlation_matrix_heatmap.object = None
    w.operator_quality_table.value = pd.DataFrame()
    w.per_scale_plots.objects = []


# ---------------------------------------------------------------------------
# Main update (the "Part B" of the old _update_anisotropic_edge_multiscale_views)
# ---------------------------------------------------------------------------


def update_multiscale_tab(
    w: MultiscaleTabWidgets,
    output: MultiscaleStrongForceOutput,
    scale_values: np.ndarray,
    state: dict,
    *,
    history: Any | None = None,
    original_results: dict[str, Any] | None = None,
    kernel_distance_method: str = "auto",
    edge_weight_mode: str = "riemannian_kernel_volume",
    kernel_assume_all_alive: bool = True,
) -> None:
    """Populate the dedicated Multiscale tab from a completed multiscale run.

    Parameters
    ----------
    w:
        Widget dataclass returned by :func:`create_multiscale_widgets`.
    output:
        The ``MultiscaleStrongForceOutput`` from the latest computation.
    scale_values:
        1-D numpy array of scale values (already detached/cpu).
    state:
        The shared dashboard ``state`` dict – used to stash output for the
        channel-selector callback.
    history:
        Optional ``RunHistory`` for geodesic distance computation.
    kernel_distance_method, edge_weight_mode, kernel_assume_all_alive:
        Settings forwarded to ``compute_pairwise_distance_matrices_from_history``.
    """
    # -- Status header -------------------------------------------------------
    status_lines = [
        "## Multiscale",
        f"- Scales available: `{len(scale_values)}`",
        f"- Frames analyzed: `{len(output.frame_indices)}`",
        f"- Bootstrap mode: `{output.bootstrap_mode_applied}`",
    ]
    base_channel_count = sum(
        1
        for channel_name in output.per_scale_results
        if not _is_companion_channel(str(channel_name))
    )
    companion_channel_count = sum(
        1
        for channel_name in output.per_scale_results
        if _is_companion_channel(str(channel_name))
    )
    status_lines.append(
        f"- Measurement groups: `non_companion={base_channel_count}`, "
        f"`companion={companion_channel_count}`"
    )

    # -- 1) Full geodesic distance matrix heatmap ----------------------------
    geodesic_error: str | None = None
    geodesic_frame_idx: int | None = None
    geodesic_max_scale = float("nan")
    try:
        if history is not None and output.frame_indices:
            geodesic_frame_idx = int(output.frame_indices[-1])
            _, distance_batch = compute_pairwise_distance_matrices_from_history(
                history,
                method=kernel_distance_method,
                frame_indices=[geodesic_frame_idx],
                batch_size=1,
                edge_weight_mode=edge_weight_mode,
                assume_all_alive=kernel_assume_all_alive,
                device=None,
                dtype=torch.float32,
            )
            if distance_batch.numel() > 0:
                distance_matrix = distance_batch[0].detach().cpu().numpy().astype(
                    np.float64, copy=False
                )
                finite_geodesics = distance_matrix[np.isfinite(distance_matrix) & (distance_matrix > 0)]
                if finite_geodesics.size > 0:
                    geodesic_max_scale = float(np.nanmax(finite_geodesics))
                n_walkers = int(distance_matrix.shape[0])
                if n_walkers == 0 or distance_matrix.shape[1] != n_walkers:
                    w.geodesic_heatmap.object = None
                    geodesic_error = "distance matrix is not square"
                else:
                    matrix = np.array(distance_matrix, dtype=np.float64, copy=True)
                    matrix[~np.isfinite(matrix)] = np.nan
                    matrix = np.nanmean(np.stack([matrix, matrix.T], axis=0), axis=0)
                    finite_positive = np.isfinite(matrix) & (matrix > 0)
                    if np.any(finite_positive):
                        fill_value = float(np.nanpercentile(matrix[finite_positive], 99.0))
                        if not np.isfinite(fill_value) or fill_value <= 0.0:
                            fill_value = float(np.nanmax(matrix[finite_positive]))
                    else:
                        fill_value = 1.0
                    if not np.isfinite(fill_value) or fill_value <= 0.0:
                        fill_value = 1.0
                    matrix = np.where(np.isfinite(matrix), matrix, fill_value)
                    matrix = np.maximum(matrix, 0.0)
                    np.fill_diagonal(matrix, 0.0)

                    clustering_note = "not clustered"
                    order = np.arange(n_walkers, dtype=np.int64)
                    try:
                        from scipy.cluster.hierarchy import leaves_list, linkage
                        from scipy.spatial.distance import squareform

                        if n_walkers >= 2:
                            condensed = squareform(matrix, checks=False)
                            linkage_mat = linkage(
                                condensed,
                                method="average",
                                optimal_ordering=True,
                            )
                            order = leaves_list(linkage_mat).astype(np.int64, copy=False)
                            clustering_note = "average-linkage"
                        else:
                            clustering_note = "single walker"
                    except Exception as cluster_exc:
                        clustering_note = f"fallback (no clustering): {cluster_exc!s}"

                    matrix_ordered = matrix[np.ix_(order, order)].astype(np.float64, copy=False)
                    log_floor = np.finfo(np.float64).tiny
                    log_matrix = np.log10(np.maximum(matrix_ordered, log_floor))
                    np.fill_diagonal(log_matrix, np.nan)
                    w.geodesic_heatmap.object = hv.QuadMesh(
                        (order, order, log_matrix.astype(np.float32, copy=False)),
                        kdims=["walker_j", "walker_i"],
                        vdims=["log10_geodesic_distance"],
                    ).opts(
                        width=900,
                        height=420,
                        cmap="Viridis",
                        colorbar=True,
                        xlabel="Walker index j (clustered order)",
                        ylabel="Walker index i (clustered order)",
                        title=(
                            f"Geodesic Distance Heatmap (log₁₀) "
                            f"(frame={geodesic_frame_idx}, {clustering_note})"
                        ),
                        tools=["hover"],
                        show_grid=False,
                    )
                    # Histogram of off-diagonal geodesic distances.
                    upper_tri = matrix_ordered[np.triu_indices(n_walkers, k=1)]
                    valid_dists = upper_tri[np.isfinite(upper_tri) & (upper_tri > 0)]
                    if valid_dists.size >= 2:
                        log_dists = np.log10(valid_dists)
                        hist_freq, hist_edges = np.histogram(log_dists, bins=60)
                        w.geodesic_histogram.object = hv.Histogram(
                            (hist_freq, hist_edges),
                            kdims=["log10_distance"],
                            vdims=["count"],
                        ).opts(
                            width=900,
                            height=260,
                            xlabel="log₁₀(geodesic distance)",
                            ylabel="Pair count",
                            title="Geodesic Distance Distribution",
                            color="#4c78a8",
                            line_color="white",
                            show_grid=True,
                        )
                    else:
                        w.geodesic_histogram.object = None
                    status_lines.append(f"- Heatmap clustering: `{clustering_note}`")
            else:
                w.geodesic_heatmap.object = None
                geodesic_error = "distance matrix is empty"
        else:
            w.geodesic_heatmap.object = None
            geodesic_error = "history or frame indices unavailable"
    except Exception as exc:
        w.geodesic_heatmap.object = None
        geodesic_error = str(exc)

    if geodesic_frame_idx is not None:
        status_lines.append(f"- Heatmap frame: `{geodesic_frame_idx}`")
    if math.isfinite(geodesic_max_scale) and geodesic_max_scale > 0:
        status_lines.append(f"- Estimated full-scale radius (max geodesic): `{geodesic_max_scale:.6g}`")
    if geodesic_error:
        status_lines.append(f"- Heatmap note: `{geodesic_error}`")

    # -- 2) Populate channel selector and store output for callback ----------
    raw_channels = sorted(
        output.per_scale_results.keys(),
        key=lambda name: (0 if _is_companion_channel(str(name)) else 1, _base_channel_name(str(name))),
    )
    display_to_raw: dict[str, str] = {}
    display_to_original: dict[str, str] = {}
    for raw_name in raw_channels:
        display_name = _display_channel_name(str(raw_name))
        if display_name not in display_to_raw:
            display_to_raw[display_name] = str(raw_name)
            display_to_original.setdefault(display_name, str(raw_name))
    if isinstance(original_results, dict):
        for raw_name, result in original_results.items():
            raw_name_s = str(raw_name)
            if raw_name_s.endswith("_multiscale_best"):
                continue
            display_name = _display_channel_name_for_original(raw_name_s, result)
            display_to_original.setdefault(display_name, raw_name_s)
    available_channels = list(display_to_raw.keys())
    for display_name in display_to_original:
        if display_name not in available_channels:
            available_channels.append(display_name)
    available_channels.sort(
        key=lambda name: (
            0 if not str(name).startswith(SPATIAL_PREFIX) else 1,
            _base_channel_name(str(name)),
        )
    )

    w.channel_select.options = available_channels
    if "nucleon" in available_channels:
        default_channel = "nucleon"
    elif "spatial_nucleon_score_abs" in available_channels:
        default_channel = "spatial_nucleon_score_abs"
    elif "spatial_nucleon" in available_channels:
        default_channel = "spatial_nucleon"
    else:
        default_channel = available_channels[0] if available_channels else ""
    w.channel_select.value = default_channel
    state["_multiscale_output"] = output
    state["_multiscale_scale_values"] = scale_values
    state["_multiscale_original_results"] = original_results
    state["_multiscale_display_to_raw"] = display_to_raw
    state["_multiscale_display_to_original"] = display_to_original
    state["_multiscale_geodesic_max_scale"] = geodesic_max_scale

    # -- 3) Inner function: update channel-specific views --------------------
    def _update_channel_views(channel_name: str) -> None:
        if not channel_name:
            return
        ms_output = state.get("_multiscale_output")
        sv = state.get("_multiscale_scale_values")
        display_to_raw_map = state.get("_multiscale_display_to_raw", {})
        display_to_original_map = state.get("_multiscale_display_to_original", {})
        raw_channel_name = str(display_to_raw_map.get(channel_name, channel_name))
        display_channel_name = str(channel_name)
        if ms_output is None or sv is None:
            return
        channel_results = ms_output.per_scale_results.get(raw_channel_name, [])

        bundle = (
            analyze_channel_across_scales(channel_results, sv, display_channel_name)
            if channel_results
            else None
        )

        # -- Look up original (no-scale-filter) mass for reference overlay --
        original_mass: float | None = None
        original_error: float = float("nan")
        original_r2: float = float("nan")
        original_result_obj: Any | None = None
        original_scale_estimate = float(state.get("_multiscale_geodesic_max_scale", float("nan")))
        if not (math.isfinite(original_scale_estimate) and original_scale_estimate > 0):
            sv_arr = np.asarray(sv, dtype=float)
            finite_scales = sv_arr[np.isfinite(sv_arr) & (sv_arr > 0)]
            if finite_scales.size > 0:
                original_scale_estimate = float(np.max(finite_scales))
        orig_results = state.get("_multiscale_original_results")
        if isinstance(orig_results, dict):
            mapped_key = display_to_original_map.get(display_channel_name)
            if mapped_key is not None:
                orig_result = orig_results.get(mapped_key)
                if orig_result is not None:
                    original_result_obj = orig_result
                    mass_fit = getattr(orig_result, "mass_fit", None)
                    if isinstance(mass_fit, dict):
                        original_mass_candidate = float(mass_fit.get("mass", float("nan")))
                        if math.isfinite(original_mass_candidate) and original_mass_candidate > 0:
                            original_mass = original_mass_candidate
                            original_error = float(mass_fit.get("mass_error", float("nan")))
                            original_r2 = float(mass_fit.get("r_squared", float("nan")))
            base_name = _base_channel_name(raw_channel_name)
            is_companion_selection = _is_companion_channel(raw_channel_name)
            if is_companion_selection:
                candidate_keys = [
                    base_name,
                    raw_channel_name,
                    display_channel_name,
                    f"{base_name}{COMPANION_SUFFIX}",
                ]
            else:
                candidate_keys = [
                    display_channel_name,
                    raw_channel_name,
                    f"{SPATIAL_PREFIX}{base_name}",
                    base_name,
                ]
            for key in candidate_keys:
                orig_result = orig_results.get(key)
                if orig_result is None:
                    continue
                if _is_companion_original_result(str(key), orig_result) != is_companion_selection:
                    continue
                original_result_obj = orig_result
                mass_fit = getattr(orig_result, "mass_fit", None)
                if not isinstance(mass_fit, dict):
                    continue
                original_mass_candidate = float(mass_fit.get("mass", float("nan")))
                if not (math.isfinite(original_mass_candidate) and original_mass_candidate > 0):
                    continue
                original_mass = original_mass_candidate
                original_error = float(mass_fit.get("mass_error", float("nan")))
                original_r2 = float(mass_fit.get("r_squared", float("nan")))
                break

        family_key = _channel_family_key(raw_channel_name)
        gevp_mass: float | None = None
        gevp_error = float("nan")
        if _is_companion_channel(raw_channel_name):
            gevp_channel_name = f"{family_key}_gevp"
            gevp_results_all = state.get("companion_strong_force_results")
            if isinstance(gevp_results_all, dict):
                gevp_candidate = gevp_results_all.get(gevp_channel_name)
                if gevp_candidate is not None:
                    gevp_fit = getattr(gevp_candidate, "mass_fit", None)
                    if isinstance(gevp_fit, dict):
                        gm = float(gevp_fit.get("mass", float("nan")))
                        if math.isfinite(gm) and gm > 0:
                            gevp_mass = gm
                            gevp_error = float(gevp_fit.get("mass_error", float("nan")))

        quality_entries_raw = _collect_family_operator_entries(
            ms_output,
            family_key,
            original_results=orig_results if isinstance(orig_results, dict) else None,
        )
        try:
            gevp_min_r2 = float(state.get("_multiscale_gevp_min_operator_r2", 0.5))
        except (TypeError, ValueError):
            gevp_min_r2 = 0.5
        try:
            gevp_min_windows = max(0, int(state.get("_multiscale_gevp_min_operator_windows", 10)))
        except (TypeError, ValueError):
            gevp_min_windows = 10
        try:
            gevp_max_error_pct = float(state.get("_multiscale_gevp_max_operator_error_pct", 30.0))
        except (TypeError, ValueError):
            gevp_max_error_pct = 30.0
        gevp_remove_artifacts = bool(
            state.get(
                "_multiscale_gevp_remove_artifacts",
                state.get("_multiscale_gevp_exclude_zero_error_operators", True),
            )
        )
        quality_entries, filtered_out = _apply_gevp_entry_filters(
            quality_entries_raw,
            min_r2=gevp_min_r2,
            min_windows=gevp_min_windows,
            max_error_pct=gevp_max_error_pct,
            remove_artifacts=gevp_remove_artifacts,
        )
        min_r2_str = f"{gevp_min_r2:.3g}" if math.isfinite(gevp_min_r2) else "off"
        max_error_pct_str = (
            f"{gevp_max_error_pct:.3g}%" if math.isfinite(gevp_max_error_pct) else "off"
        )
        filter_header = (
            f"`R² >= {min_r2_str}`, `n_windows >= {gevp_min_windows}`, "
            f"`error % <= {max_error_pct_str}`, "
            f"`remove artifacts={'on' if gevp_remove_artifacts else 'off'}`"
        )

        quality_analysis = _analyze_operator_quality_vectorized(
            quality_entries,
            t0=2,
            eig_rel_cutoff=1e-2,
            importance_cutoff=0.05,
            use_connected=True,
        )
        if quality_analysis is None:
            if not quality_entries and filtered_out:
                preview = ", ".join(
                    f"{label}({reason})" for label, reason in filtered_out[:4]
                )
                suffix = " ..." if len(filtered_out) > 4 else ""
                w.operator_quality_summary.object = (
                    f"**{family_key}:** no operators satisfy active GEVP filters.  \n"
                    f"_Filters:_ {filter_header}.  \n"
                    f"_Excluded:_ `{len(filtered_out)}` operators (`{preview}{suffix}`)."
                )
            else:
                w.operator_quality_summary.object = (
                    f"**{family_key}:** insufficient operator-series data for eigen-spectrum analysis "
                    f"after applying active GEVP filters.  \n"
                    f"_Filters:_ {filter_header}.  \n"
                    f"_Shown:_ `{len(quality_entries)}` operators; "
                    f"_Excluded:_ `{len(filtered_out)}`."
                )
            if gevp_mass is not None and math.isfinite(gevp_mass) and gevp_mass > 0:
                gevp_line = f"- Final GEVP mass: `{gevp_mass:.6g}`"
                if math.isfinite(gevp_error) and gevp_error > 0:
                    gevp_line += f" ± `{gevp_error:.2g}`"
                w.operator_quality_summary.object += f"  \n{gevp_line}"
            w.eigenspectrum_plot.object = None
            w.eigenvalue_table.value = pd.DataFrame()
            w.correlation_matrix_heatmap.object = None
            w.operator_quality_table.value = (
                pd.DataFrame(
                    [
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
                    ]
                )
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
            w.operator_quality_summary.object = (
                f"**{family_key} operator report:** "
                f"`{n_ops}` operators analyzed, `{n_sig}` significant eigen-modes, "
                f"`{max(0, n_ops - n_sig)}` redundancy candidates, "
                f"`cond(C(t0))={cond_str}`.  \n"
                f"_Filters:_ {filter_header}.  \n"
                f"_Shown:_ `{len(quality_entries)}` operators; "
                f"_Excluded:_ `{len(filtered_out)}`.  \n"
                f"_Top operators by importance:_ `{top_ops}`.  \n"
                "_Table includes only operators passing active GEVP-quality filters._"
            )
            if gevp_mass is not None and math.isfinite(gevp_mass) and gevp_mass > 0:
                gevp_line = f"- Final GEVP mass: `{gevp_mass:.6g}`"
                if math.isfinite(gevp_error) and gevp_error > 0:
                    gevp_line += f" ± `{gevp_error:.2g}`"
                w.operator_quality_summary.object += f"  \n{gevp_line}"
            qdf = pd.DataFrame(quality_analysis.get("rows", []))
            if not qdf.empty:
                qdf = qdf.sort_values(
                    ["quality_score", "importance", "operator"],
                    ascending=[False, False, True],
                    kind="stable",
                )
            w.operator_quality_table.value = qdf
            w.eigenspectrum_plot.object = _build_eigenspectrum_plot(
                quality_analysis,
                family_label=family_key,
            )
            w.eigenvalue_table.value = _build_kept_eigenvalue_table(quality_analysis)
            w.correlation_matrix_heatmap.object = _build_correlation_matrix_heatmap(
                quality_analysis,
                family_label=family_key,
            )

        if bundle is None:
            w.corr_plot.object = build_multiscale_correlator_plot(
                [],
                np.asarray([], dtype=float),
                display_channel_name,
                reference_result=original_result_obj,
                reference_label="original (no filter)",
                reference_scale=original_scale_estimate,
            )
            w.meff_plot.object = build_multiscale_effective_mass_plot(
                [],
                np.asarray([], dtype=float),
                display_channel_name,
                reference_result=original_result_obj,
                reference_label="original (no filter)",
                reference_scale=original_scale_estimate,
            )
            w.mass_vs_scale_plot.object = build_mass_vs_scale_plot(
                [],
                display_channel_name,
                reference_mass=original_mass,
                reference_scale=original_scale_estimate,
                gevp_mass=gevp_mass,
                gevp_error=gevp_error,
                gevp_scale=original_scale_estimate,
            )
            est_rows = build_estimator_table_rows(
                [],
                original_mass=original_mass,
                original_error=original_error,
                original_r2=original_r2,
                original_scale=original_scale_estimate,
            )
            if gevp_mass is not None and math.isfinite(gevp_mass) and gevp_mass > 0:
                gevp_row: dict[str, float | str] = {
                    "scale": "GEVP (final)",
                    "scale_value": float("nan"),
                    "mass": float(gevp_mass),
                    "mass_error": float(gevp_error),
                    "mass_error_pct": (
                        abs(float(gevp_error) / float(gevp_mass)) * 100.0
                        if math.isfinite(gevp_error) and gevp_error > 0
                        else float("nan")
                    ),
                    "r_squared": float("nan"),
                }
                if original_mass is not None and math.isfinite(original_mass) and original_mass > 0:
                    gevp_row["delta_vs_original_pct"] = (
                        (float(gevp_mass) - float(original_mass)) / float(original_mass) * 100.0
                    )
                est_rows.append(gevp_row)
            w.estimator_table.value = pd.DataFrame(est_rows) if est_rows else pd.DataFrame()
            if original_mass is not None:
                w.consensus_summary.object = (
                    f"**{display_channel_name}:** original-only result available; "
                    "no multiscale sweep was produced for this channel in the current run."
                )
            else:
                w.consensus_summary.object = (
                    f"**{display_channel_name}:** no per-scale or original results available."
                )
            if gevp_mass is not None and math.isfinite(gevp_mass) and gevp_mass > 0:
                gevp_line = f"- Final GEVP mass: `{gevp_mass:.6g}`"
                if math.isfinite(gevp_error) and gevp_error > 0:
                    gevp_line += f" ± `{gevp_error:.2g}`"
                w.consensus_summary.object += f"  \n{gevp_line}"
            w.pairwise_table.value = pd.DataFrame()
            w.systematics_badge.object = (
                "Systematics verdict: no multiscale sweep available for this channel."
            )
            w.systematics_badge.alert_type = "secondary"
            w.consensus_plot.object = None
            w.per_scale_plots.objects = []
            return

        # Correlator plot (scatter+errorbars per scale).
        w.corr_plot.object = build_multiscale_correlator_plot(
            channel_results,
            sv,
            display_channel_name,
            reference_result=original_result_obj,
            reference_label="original (no filter)",
            reference_scale=original_scale_estimate,
        )
        # Effective mass plot.
        w.meff_plot.object = build_multiscale_effective_mass_plot(
            channel_results,
            sv,
            display_channel_name,
            reference_result=original_result_obj,
            reference_label="original (no filter)",
            reference_scale=original_scale_estimate,
        )
        # Mass vs scale with consensus overlay.
        w.mass_vs_scale_plot.object = build_mass_vs_scale_plot(
            bundle.measurements,
            display_channel_name,
            consensus=bundle.consensus,
            reference_mass=original_mass,
            reference_scale=original_scale_estimate,
            gevp_mass=gevp_mass,
            gevp_error=gevp_error,
            gevp_scale=original_scale_estimate,
        )
        est_rows = build_estimator_table_rows(
            bundle.measurements,
            original_mass=original_mass,
            original_error=original_error,
            original_r2=original_r2,
            original_scale=original_scale_estimate,
        )
        if gevp_mass is not None and math.isfinite(gevp_mass) and gevp_mass > 0:
            gevp_row: dict[str, float | str] = {
                "scale": "GEVP (final)",
                "scale_value": float("nan"),
                "mass": float(gevp_mass),
                "mass_error": float(gevp_error),
                "mass_error_pct": (
                    abs(float(gevp_error) / float(gevp_mass)) * 100.0
                    if math.isfinite(gevp_error) and gevp_error > 0
                    else float("nan")
                ),
                "r_squared": float("nan"),
            }
            if original_mass is not None and math.isfinite(original_mass) and original_mass > 0:
                gevp_row["delta_vs_original_pct"] = (
                    (float(gevp_mass) - float(original_mass)) / float(original_mass) * 100.0
                )
            est_rows.append(gevp_row)
        w.estimator_table.value = (
            pd.DataFrame(est_rows) if est_rows else pd.DataFrame()
        )
        # Pairwise discrepancy table.
        pw_rows = build_pairwise_table_rows(bundle.discrepancies)
        w.pairwise_table.value = (
            pd.DataFrame(pw_rows).sort_values("abs_delta_pct", ascending=False)
            if pw_rows
            else pd.DataFrame()
        )
        # Consensus summary + badge.
        w.consensus_summary.object = format_consensus_summary(
            bundle.consensus, bundle.discrepancies, display_channel_name,
            reference_mass=original_mass,
        )
        if gevp_mass is not None and math.isfinite(gevp_mass) and gevp_mass > 0:
            gevp_line = f"- Final GEVP mass: `{gevp_mass:.6g}`"
            if math.isfinite(gevp_error) and gevp_error > 0:
                gevp_line += f" ± `{gevp_error:.2g}`"
            if math.isfinite(bundle.consensus.mass) and bundle.consensus.mass > 0:
                delta_gevp_pct = (
                    (float(bundle.consensus.mass) - float(gevp_mass)) / float(gevp_mass) * 100.0
                )
                gevp_line += f" (consensus delta `{delta_gevp_pct:+.2f}%`)"
            w.consensus_summary.object += f"  \n{gevp_line}"
        w.systematics_badge.object = (
            f"Systematics verdict: {bundle.verdict.label}. {bundle.verdict.details}"
        )
        w.systematics_badge.alert_type = bundle.verdict.alert_type
        # Consensus plot.
        w.consensus_plot.object = build_consensus_plot(
            bundle.measurements, bundle.consensus, display_channel_name,
            reference_mass=original_mass,
            gevp_mass=gevp_mass,
            gevp_error=gevp_error,
        )
        # Per-scale ChannelPlot layouts.
        per_scale_layouts = build_per_scale_channel_plots(channel_results, sv)
        if per_scale_layouts:
            children: list[Any] = []
            for label, layout in per_scale_layouts:
                children.append(pn.pane.Markdown(f"#### {label}"))
                children.append(layout)
            w.per_scale_plots.objects = children
        else:
            w.per_scale_plots.objects = []

    # -- Wire channel selector callback --------------------------------------
    def _on_channel_change(event: Any) -> None:
        _update_channel_views(str(event.new))

    w.channel_select.param.watch(_on_channel_change, "value")

    # -- Initial render for default channel ----------------------------------
    _update_channel_views(default_channel)

    w.status.object = "  \n".join(status_lines)
