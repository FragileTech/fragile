"""Parity tests: fragile.fractalai.qft.electroweak_channels vs fragile.physics.electroweak.electroweak_channels.

Each test imports from BOTH locations and asserts exact numerical equality.
The new module removes alive/bounds plumbing; since mock histories have all
walkers alive (alive_mask=ones) and pbc=False/bounds=None, the MC-time path
produces identical results.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from fragile.fractalai.qft.electroweak_channels import (
    compute_all_electroweak_channels as old_compute_all,
    compute_electroweak_channels as old_compute_channels,
    compute_electroweak_coupling_constants as old_compute_coupling,
    compute_electroweak_snapshot_operators as old_compute_snapshot,
    compute_emergent_electroweak_scales as old_compute_scales,
    ELECTROWEAK_BASE_CHANNELS as old_ELECTROWEAK_BASE_CHANNELS,
    ELECTROWEAK_CHANNELS as old_ELECTROWEAK_CHANNELS,
    ELECTROWEAK_DIRECTIONAL_SU2_CHANNELS as old_ELECTROWEAK_DIRECTIONAL_SU2_CHANNELS,
    ELECTROWEAK_PARITY_CHANNELS as old_ELECTROWEAK_PARITY_CHANNELS,
    ELECTROWEAK_SYMMETRY_BREAKING_CHANNELS as old_ELECTROWEAK_SYMMETRY_BREAKING_CHANNELS,
    ELECTROWEAK_WALKER_TYPE_SU2_CHANNELS as old_ELECTROWEAK_WALKER_TYPE_SU2_CHANNELS,
    ElectroweakChannelConfig as old_ElectroweakChannelConfig,
)
from fragile.physics.electroweak.electroweak_channels import (
    compute_all_electroweak_channels as new_compute_all,
    compute_electroweak_channels as new_compute_channels,
    compute_electroweak_coupling_constants as new_compute_coupling,
    compute_electroweak_snapshot_operators as new_compute_snapshot,
    compute_emergent_electroweak_scales as new_compute_scales,
    ELECTROWEAK_BASE_CHANNELS as new_ELECTROWEAK_BASE_CHANNELS,
    ELECTROWEAK_CHANNELS as new_ELECTROWEAK_CHANNELS,
    ELECTROWEAK_DIRECTIONAL_SU2_CHANNELS as new_ELECTROWEAK_DIRECTIONAL_SU2_CHANNELS,
    ELECTROWEAK_PARITY_CHANNELS as new_ELECTROWEAK_PARITY_CHANNELS,
    ELECTROWEAK_SYMMETRY_BREAKING_CHANNELS as new_ELECTROWEAK_SYMMETRY_BREAKING_CHANNELS,
    ELECTROWEAK_WALKER_TYPE_SU2_CHANNELS as new_ELECTROWEAK_WALKER_TYPE_SU2_CHANNELS,
    ElectroweakChannelConfig as new_ElectroweakChannelConfig,
    ElectroweakChannelOutput,
)
from tests.physics.electroweak.conftest import (
    assert_channel_results_equal,
    assert_dict_floats_equal,
    assert_dict_tensors_equal,
)


# Fields removed from the new ElectroweakChannelConfig
_REMOVED_CONFIG_FIELDS = {
    "time_axis",
    "euclidean_time_dim",
    "euclidean_time_bins",
    "euclidean_time_range",
    "mc_time_index",
}


# ===================================================================
# Constants parity
# ===================================================================


class TestParityConstants:
    def test_electroweak_base_channels(self):
        assert old_ELECTROWEAK_BASE_CHANNELS == new_ELECTROWEAK_BASE_CHANNELS

    def test_electroweak_channels(self):
        assert old_ELECTROWEAK_CHANNELS == new_ELECTROWEAK_CHANNELS

    def test_electroweak_directional_su2_channels(self):
        assert old_ELECTROWEAK_DIRECTIONAL_SU2_CHANNELS == new_ELECTROWEAK_DIRECTIONAL_SU2_CHANNELS

    def test_electroweak_walker_type_su2_channels(self):
        assert old_ELECTROWEAK_WALKER_TYPE_SU2_CHANNELS == new_ELECTROWEAK_WALKER_TYPE_SU2_CHANNELS

    def test_electroweak_symmetry_breaking_channels(self):
        assert (
            old_ELECTROWEAK_SYMMETRY_BREAKING_CHANNELS
            == new_ELECTROWEAK_SYMMETRY_BREAKING_CHANNELS
        )

    def test_electroweak_parity_channels(self):
        assert old_ELECTROWEAK_PARITY_CHANNELS == new_ELECTROWEAK_PARITY_CHANNELS


# ===================================================================
# Config parity
# ===================================================================


class TestParityConfig:
    def test_shared_fields_identical(self):
        """All fields present in BOTH configs have the same default values."""
        old_cfg = old_ElectroweakChannelConfig()
        new_cfg = new_ElectroweakChannelConfig()
        old_fields = {f.name: getattr(old_cfg, f.name) for f in dataclasses.fields(old_cfg)}
        new_fields = {f.name: getattr(new_cfg, f.name) for f in dataclasses.fields(new_cfg)}
        # New config should be a strict subset (minus removed fields)
        for name, value in new_fields.items():
            assert name in old_fields, f"New field {name!r} not in old config"
            assert (
                old_fields[name] == value
            ), f"Field {name!r}: old={old_fields[name]!r} new={value!r}"

    def test_removed_fields(self):
        """Fields removed from new config should still exist in old config."""
        old_cfg = old_ElectroweakChannelConfig()
        new_cfg = new_ElectroweakChannelConfig()
        old_field_names = {f.name for f in dataclasses.fields(old_cfg)}
        new_field_names = {f.name for f in dataclasses.fields(new_cfg)}
        removed = old_field_names - new_field_names
        assert removed == _REMOVED_CONFIG_FIELDS

    def test_no_unexpected_new_fields(self):
        """New config should not introduce fields absent from old config."""
        old_field_names = {f.name for f in dataclasses.fields(old_ElectroweakChannelConfig())}
        new_field_names = {f.name for f in dataclasses.fields(new_ElectroweakChannelConfig())}
        added = new_field_names - old_field_names
        assert added == set(), f"Unexpected new fields: {added}"


# ===================================================================
# Output dataclass
# ===================================================================


class TestOutputStructure:
    def test_output_no_avg_alive_walkers(self):
        """ElectroweakChannelOutput should not have avg_alive_walkers field."""
        fields = {f.name for f in dataclasses.fields(ElectroweakChannelOutput)}
        assert "avg_alive_walkers" not in fields
        assert "avg_edges" in fields


# ===================================================================
# compute_electroweak_channels parity
# ===================================================================


class TestParityComputeElectroweakChannels:
    def test_default_channels(self, mock_history):
        old = old_compute_channels(mock_history)
        new = new_compute_channels(mock_history)
        assert old.frame_indices == new.frame_indices
        assert old.n_valid_frames == new.n_valid_frames
        assert old.avg_edges == new.avg_edges
        assert_channel_results_equal(old.channel_results, new.channel_results)

    def test_explicit_channels(self, mock_history):
        channels = ["u1_phase", "su2_phase"]
        old = old_compute_channels(mock_history, channels=channels)
        new = new_compute_channels(mock_history, channels=channels)
        assert old.frame_indices == new.frame_indices
        assert old.n_valid_frames == new.n_valid_frames
        assert_channel_results_equal(old.channel_results, new.channel_results)

    def test_score_directed(self, mock_history):
        cfg = old_ElectroweakChannelConfig(su2_operator_mode="score_directed")
        old = old_compute_channels(mock_history, config=cfg)
        cfg2 = new_ElectroweakChannelConfig(su2_operator_mode="score_directed")
        new = new_compute_channels(mock_history, config=cfg2)
        assert_channel_results_equal(old.channel_results, new.channel_results)

    def test_walker_type_split(self, mock_history_with_will_clone):
        cfg = old_ElectroweakChannelConfig(enable_walker_type_split=True)
        old = old_compute_channels(mock_history_with_will_clone, config=cfg)
        cfg2 = new_ElectroweakChannelConfig(enable_walker_type_split=True)
        new = new_compute_channels(mock_history_with_will_clone, config=cfg2)
        assert_channel_results_equal(old.channel_results, new.channel_results)


# ===================================================================
# compute_all_electroweak_channels parity
# ===================================================================


class TestParityComputeAll:
    def test_default(self, mock_history):
        old = old_compute_all(mock_history)
        new = new_compute_all(mock_history)
        assert_channel_results_equal(old, new)

    def test_explicit_channels(self, mock_history):
        channels = ["u1_phase", "su2_doublet"]
        old = old_compute_all(mock_history, channels=channels)
        new = new_compute_all(mock_history, channels=channels)
        assert_channel_results_equal(old, new)


# ===================================================================
# compute_electroweak_snapshot_operators parity
# ===================================================================


class TestParitySnapshotOperators:
    def test_default_frame(self, mock_history):
        old = old_compute_snapshot(mock_history)
        new = new_compute_snapshot(mock_history)
        assert_dict_tensors_equal(old, new)

    def test_explicit_frame(self, mock_history):
        old = old_compute_snapshot(mock_history, frame_idx=5)
        new = new_compute_snapshot(mock_history, frame_idx=5)
        assert_dict_tensors_equal(old, new)

    def test_explicit_channels(self, mock_history):
        channels = ["u1_phase"]
        old = old_compute_snapshot(mock_history, channels=channels)
        new = new_compute_snapshot(mock_history, channels=channels)
        assert_dict_tensors_equal(old, new)


# ===================================================================
# compute_electroweak_coupling_constants parity
# ===================================================================


class TestParityCouplingConstants:
    def test_default(self, mock_history):
        old = old_compute_coupling(mock_history, h_eff=1.0)
        new = new_compute_coupling(mock_history, h_eff=1.0)
        assert_dict_floats_equal(old, new)


# ===================================================================
# compute_emergent_electroweak_scales parity
# ===================================================================


class TestParityEmergentScales:
    def test_default(self, mock_history):
        old = old_compute_scales(mock_history)
        new = new_compute_scales(mock_history)
        assert_dict_floats_equal(old, new)
