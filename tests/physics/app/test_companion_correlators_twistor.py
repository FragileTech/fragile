"""Dashboard regressions for twistor companion correlator wiring."""

from __future__ import annotations

import pandas as pd
import pytest
import torch


pn = pytest.importorskip("panel")
hv = pytest.importorskip("holoviews")

from fragile.physics.app import companion_correlators as cc
from fragile.physics.operators.pipeline import PipelineResult
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
    imag = 0.5 * torch.roll(x, shifts=1, dims=-1)
    color = torch.complex(x, imag)
    valid = torch.ones(color.shape[:2], dtype=torch.bool, device=color.device)
    return color, valid


def _build_twistor_test_section(monkeypatch):
    """Build a companion correlator section with lightweight dashboard stubs."""
    state: dict[str, object] = {"companion_correlator_output": None}
    history = MockRunHistory(N=12, d=4, n_recorded=18, seed=7)

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

    def run_tab_computation(state_dict, status, label, callback):
        del status, label
        callback(history)

    section = cc.build_companion_correlator_tab(
        state=state,
        run_tab_computation=run_tab_computation,
    )
    return section, state


def test_companion_tab_computes_scalar_family_twistor_variants(monkeypatch):
    """Selecting scalar-family twistor modes should emit fit-ready channel keys."""
    section, state = _build_twistor_test_section(monkeypatch)

    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Scalar").value = ["twistor"]
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Pseudoscalar").value = ["twistor"]
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Glueball").value = ["twistor"]
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Baryon").value = []
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Vector").value = []
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Axial Vector").value = []
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Tensor").value = []

    section.on_run(None)

    result = state["companion_correlator_output"]
    assert isinstance(result, PipelineResult)
    assert set(result.correlators) == {
        "scalar_twistor",
        "pseudoscalar_twistor",
        "glueball_twistor",
    }
    assert set(result.operators) == {
        "scalar_twistor",
        "pseudoscalar_twistor",
        "glueball_twistor",
    }
    for key in result.correlators:
        assert result.correlators[key].ndim == 1
        assert torch.isfinite(result.correlators[key]).all()


def test_companion_tab_computes_spin_family_twistor_variants(monkeypatch):
    """Selecting vector/axial/tensor twistor modes should emit fit-ready channel keys."""
    section, state = _build_twistor_test_section(monkeypatch)

    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Scalar").value = []
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Pseudoscalar").value = []
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Glueball").value = []
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Baryon").value = []
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Vector").value = ["twistor"]
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Axial Vector").value = ["twistor"]
    _find_named_widget(section.tab, pn.widgets.MultiSelect, "Tensor").value = ["twistor"]

    section.on_run(None)

    result = state["companion_correlator_output"]
    assert isinstance(result, PipelineResult)
    assert set(result.correlators) == {
        "vector_twistor",
        "axial_vector_twistor",
        "tensor_twistor",
    }
    assert set(result.operators) == {
        "vector_twistor",
        "axial_vector_twistor",
        "tensor_twistor",
    }
    assert result.operators["vector_twistor"].shape[-1] == 3
    assert result.operators["axial_vector_twistor"].shape[-1] == 3
    assert result.operators["tensor_twistor"].ndim == 1
