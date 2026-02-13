"""Tests for strong-force anchored mass calibration modes."""

from __future__ import annotations

import pytest

from fragile.fractalai.qft.dashboard import (
    _build_anchor_rows,
    BARYON_REFS,
    MESON_REFS,
)


def test_build_anchor_rows_per_anchor_mode_uses_one_scale_per_row() -> None:
    masses = {"baryon": 2.0, "meson": 1.0}

    rows = _build_anchor_rows(
        masses,
        glueball_ref=None,
        sqrt_sigma_ref=None,
        anchor_mode="per_anchor_row",
    )

    assert len(rows) == len(BARYON_REFS) + len(MESON_REFS)
    proton_row = next(row for row in rows if row.get("anchor") == "baryon->proton")
    expected_scale = BARYON_REFS["proton"] / masses["baryon"]
    assert proton_row["scale_GeV_per_alg"] == pytest.approx(expected_scale)
    assert proton_row["baryon_pred_GeV"] == pytest.approx(BARYON_REFS["proton"])


def test_build_anchor_rows_family_fixed_mode_uses_one_scale_per_family() -> None:
    masses = {"baryon": 2.0, "meson": 1.0, "glueball": 3.0}

    rows = _build_anchor_rows(
        masses,
        glueball_ref=("glueball", 2.4),
        sqrt_sigma_ref=None,
        anchor_mode="family_fixed",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["anchor"] == "family-fixed"
    assert row["baryon_pred_GeV"] == pytest.approx(
        masses["baryon"] * row["baryon_scale_GeV_per_alg"]
    )
    assert row["meson_pred_GeV"] == pytest.approx(masses["meson"] * row["meson_scale_GeV_per_alg"])
    assert row["glueball_pred_GeV"] == pytest.approx(
        masses["glueball"] * row["glueball_scale_GeV_per_alg"]
    )


def test_build_anchor_rows_global_fixed_mode_uses_one_global_scale() -> None:
    masses = {"baryon": 2.0, "meson": 1.0, "glueball": 3.0}

    rows = _build_anchor_rows(
        masses,
        glueball_ref=("glueball", 2.4),
        sqrt_sigma_ref=None,
        anchor_mode="global_fixed",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["anchor"] == "global-fixed"
    scale = row["scale_GeV_per_alg"]
    assert scale > 0.0
    assert row["baryon_pred_GeV"] == pytest.approx(masses["baryon"] * scale)
    assert row["meson_pred_GeV"] == pytest.approx(masses["meson"] * scale)
    assert row["glueball_pred_GeV"] == pytest.approx(masses["glueball"] * scale)
