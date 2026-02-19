"""AIC parity tests for anisotropic_edge_channels component label/factor helpers.

Compares pure functions from the AIC copy against the original qft module
to ensure they produce identical results (regression baseline).
"""

from __future__ import annotations

import pytest
import torch

from fragile.fractalai.qft.anisotropic_edge_channels import (
    _channel_component_name,
    _component_factors,
    _component_labels,
    COMPONENT_MODES,
    SUPPORTED_CHANNELS,
)
from fragile.physics.aic.anisotropic_edge_channels import (
    _channel_component_name as new_comp_name,
    _component_factors as new_factors,
    _component_labels as new_labels,
    COMPONENT_MODES as NEW_COMPONENT_MODES,
    SUPPORTED_CHANNELS as NEW_SUPPORTED_CHANNELS,
)


# ---------------------------------------------------------------------------
# TestParityComponentLabels
# ---------------------------------------------------------------------------


class TestParityComponentLabels:
    """Verify _component_labels returns identical lists for old and new."""

    @pytest.mark.parametrize(
        ("dim", "mode"),
        [
            (3, "isotropic"),
            (3, "axes"),
            (3, "isotropic+axes"),
            (3, "quadrupole"),
            (3, "isotropic+quadrupole"),
            (4, "isotropic"),
            (4, "axes"),
        ],
    )
    def test_labels_match(self, dim: int, mode: str) -> None:
        old = _component_labels(dim, mode)
        new = new_labels(dim, mode)
        assert old == new, f"dim={dim}, mode={mode}: {old} != {new}"


# ---------------------------------------------------------------------------
# TestParityComponentFactors
# ---------------------------------------------------------------------------


class TestParityComponentFactors:
    """Verify _component_factors returns identical dicts for old and new."""

    @pytest.mark.parametrize("mode", list(COMPONENT_MODES))
    def test_factors_match(self, mode: str) -> None:
        gen = torch.Generator()
        gen.manual_seed(42)
        direction = torch.randn(10, 3, generator=gen)

        old = _component_factors(direction, mode)
        new = new_factors(direction, mode)

        assert set(old.keys()) == set(
            new.keys()
        ), f"mode={mode}: key mismatch {set(old.keys())} vs {set(new.keys())}"
        for key in old:
            assert torch.equal(old[key], new[key]), f"mode={mode}, key={key}: tensors differ"


# ---------------------------------------------------------------------------
# TestParityChannelComponentName
# ---------------------------------------------------------------------------


class TestParityChannelComponentName:
    """Verify _channel_component_name returns identical strings."""

    @pytest.mark.parametrize(
        ("channel", "component"),
        [
            ("scalar", "iso"),
            ("scalar", "axis_0"),
            ("vector", "q01"),
        ],
    )
    def test_name_match(self, channel: str, component: str) -> None:
        old = _channel_component_name(channel, component)
        new = new_comp_name(channel, component)
        assert old == new, f"({channel}, {component}): {old!r} != {new!r}"


# ---------------------------------------------------------------------------
# TestParityConstants
# ---------------------------------------------------------------------------


class TestParityConstants:
    """Verify module-level constants are identical between old and new."""

    def test_component_modes(self) -> None:
        assert COMPONENT_MODES == NEW_COMPONENT_MODES

    def test_supported_channels(self) -> None:
        assert SUPPORTED_CHANNELS == NEW_SUPPORTED_CHANNELS
