"""Dashboard regressions for Dirac companion correlator wiring."""

from __future__ import annotations

import pandas as pd
import pytest
import torch


pn = pytest.importorskip("panel")
hv = pytest.importorskip("holoviews")

from fragile.physics.app import companion_correlators as cc
from fragile.physics.new_channels.dirac_spinors import DiracOperatorSeries
from fragile.physics.operators.pipeline import PipelineResult
from fragile.physics.qft_utils import resolve_frame_indices
from tests.physics.new_channels.conftest import MockRunHistory


def _walk_panel_objects(root):
    """Yield a Panel object and all descendants reachable via ``objects``."""
    yield root
    for child in getattr(root, "objects", []):
        yield from _walk_panel_objects(child)


def _find_named_widget(root, widget_type, name: str):
    """Find a widget by type and display name inside a Panel tree."""
    for obj in _walk_panel_objects(root):
        if isinstance(obj, widget_type) and getattr(obj, "name", None) == name:
            return obj
    raise AssertionError(f"Could not find widget {name!r} of type {widget_type.__name__}.")


def _fake_color_states_batch(history, start_idx, h_eff, mass, ell0, end_idx=None):
    """Deterministic lightweight replacement for dashboard tests."""
    del h_eff, mass, ell0
    if end_idx is None:
        end_idx = history.n_recorded
    x = torch.as_tensor(history.x_before_clone[start_idx:end_idx], dtype=torch.float32)
    imag = 0.25 * torch.roll(x, shifts=1, dims=-1)
    color = torch.complex(x[..., :3], imag[..., :3])
    valid = torch.ones(color.shape[:2], dtype=torch.bool, device=color.device)
    return color, valid


def test_companion_tab_passes_full_pair_backbone_to_dirac(monkeypatch):
    """Dirac modes in the companion dashboard should use the standard pair backbone."""
    history = MockRunHistory(N=12, d=4, n_recorded=18, seed=17)
    state: dict[str, object] = {"companion_correlator_output": None}
    captured: dict[str, torch.Tensor] = {}

    monkeypatch.setattr(cc, "compute_color_states_batch", _fake_color_states_batch)
    monkeypatch.setattr(cc, "estimate_ell0_auto", lambda history, method: 1.0)
    monkeypatch.setattr(cc, "build_summary_table", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(cc, "build_correlator_table", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        cc,
        "build_grouped_correlator_plot",
        lambda *args, **kwargs: hv.Curve(([0.0, 1.0], [1.0, 0.5])),
    )
    monkeypatch.setattr(
        cc,
        "build_grouped_meff_plot",
        lambda *args, **kwargs: hv.Curve(([0.0, 1.0], [0.5, 0.5])),
    )

    def _fake_dirac_operator_series(
        *,
        color,
        color_valid,
        sample_indices,
        neighbor_indices,
        alive,
        sample_edge_weights=None,
    ):
        del color_valid, alive, sample_edge_weights
        captured["sample_indices"] = sample_indices.detach().cpu()
        captured["neighbor_indices"] = neighbor_indices.detach().cpu()
        T = color.shape[0]
        zeros = torch.zeros(T, dtype=torch.float32, device=color.device)
        return DiracOperatorSeries(
            scalar=zeros.clone(),
            pseudoscalar=zeros.clone(),
            vector=zeros.clone(),
            axial_vector=zeros.clone(),
            tensor=zeros.clone(),
            tensor_0k=zeros.clone(),
            n_valid_pairs=zeros.to(torch.int64),
            spinor_valid_fraction=zeros.clone(),
        )

    import fragile.physics.new_channels.dirac_spinors as dirac_spinors

    monkeypatch.setattr(dirac_spinors, "compute_dirac_operator_series", _fake_dirac_operator_series)

    def run_tab_computation(state_dict, status, label, callback):
        del state_dict, status, label
        callback(history)

    section = cc.build_companion_correlator_tab(
        state=state,
        run_tab_computation=run_tab_computation,
    )

    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Scalar").value = ["dirac"]
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Pseudoscalar").value = []
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Baryon").value = []
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Vector").value = []
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Glueball").value = []
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Axial Vector").value = []
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Tensor").value = []

    section.on_run(None)

    result = state["companion_correlator_output"]
    assert isinstance(result, PipelineResult)
    assert "scalar_dirac" in result.correlators

    pair_indices = captured["neighbor_indices"]
    assert pair_indices.ndim == 3
    assert pair_indices.shape[-1] == 2

    frame_indices = resolve_frame_indices(
        history=history,
        warmup_fraction=0.3,
        end_fraction=1.0,
    )
    start_idx = frame_indices[0]
    end_idx = frame_indices[-1] + 1
    expected_distance = history.companions_distance[start_idx - 1 : end_idx - 1]
    expected_clone = history.companions_clone[start_idx - 1 : end_idx - 1]
    expected_pairs = torch.stack([expected_distance, expected_clone], dim=-1)

    assert torch.equal(pair_indices, expected_pairs[: pair_indices.shape[0]])
