"""Comprehensive tests for physics/operators/config.py dataclasses."""

from __future__ import annotations

import dataclasses
import math

import pytest

from fragile.physics.operators.config import (
    BaryonOperatorConfig,
    ChannelConfigBase,
    CorrelatorConfig,
    GlueballOperatorConfig,
    MesonOperatorConfig,
    MultiscaleConfig,
    TensorOperatorConfig,
    VectorOperatorConfig,
)


# ---------------------------------------------------------------------------
# TestChannelConfigBase
# ---------------------------------------------------------------------------


class TestChannelConfigBase:
    """Tests for the base channel configuration dataclass."""

    def test_default_values(self):
        cfg = ChannelConfigBase()
        assert cfg.warmup_fraction == 0.1
        assert cfg.end_fraction == 1.0
        assert cfg.h_eff == 1.0
        assert cfg.mass == 1.0
        assert cfg.ell0 is None
        assert cfg.color_dims is None
        assert cfg.eps == 1e-12
        assert cfg.pair_selection == "both"

    def test_override_all_fields(self):
        cfg = ChannelConfigBase(
            warmup_fraction=0.2,
            end_fraction=0.8,
            h_eff=2.0,
            mass=0.5,
            ell0=math.pi,
            color_dims=(1, 2, 3),
            eps=1e-8,
            pair_selection="distance",
        )
        assert cfg.warmup_fraction == 0.2
        assert cfg.end_fraction == 0.8
        assert cfg.h_eff == 2.0
        assert cfg.mass == 0.5
        assert cfg.ell0 == math.pi
        assert cfg.color_dims == (1, 2, 3)
        assert cfg.eps == 1e-8
        assert cfg.pair_selection == "distance"

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(ChannelConfigBase)

    def test_fields_count(self):
        fields = dataclasses.fields(ChannelConfigBase)
        assert len(fields) == 8

    def test_field_names(self):
        names = {f.name for f in dataclasses.fields(ChannelConfigBase)}
        expected = {
            "warmup_fraction",
            "end_fraction",
            "h_eff",
            "mass",
            "ell0",
            "color_dims",
            "eps",
            "pair_selection",
        }
        assert names == expected

    def test_partial_override(self):
        cfg = ChannelConfigBase(warmup_fraction=0.5, mass=2.0)
        assert cfg.warmup_fraction == 0.5
        assert cfg.mass == 2.0
        # Others keep defaults
        assert cfg.end_fraction == 1.0
        assert cfg.h_eff == 1.0
        assert cfg.eps == 1e-12

    def test_equality(self):
        cfg1 = ChannelConfigBase()
        cfg2 = ChannelConfigBase()
        assert cfg1 == cfg2

    def test_inequality(self):
        cfg1 = ChannelConfigBase(mass=1.0)
        cfg2 = ChannelConfigBase(mass=2.0)
        assert cfg1 != cfg2


# ---------------------------------------------------------------------------
# TestMesonOperatorConfig
# ---------------------------------------------------------------------------


class TestMesonOperatorConfig:
    """Tests for MesonOperatorConfig."""

    def test_inherits_from_base(self):
        assert issubclass(MesonOperatorConfig, ChannelConfigBase)

    def test_default_values(self):
        cfg = MesonOperatorConfig()
        assert cfg.operator_mode == "standard"

    def test_inherits_base_defaults(self):
        cfg = MesonOperatorConfig()
        assert cfg.warmup_fraction == 0.1
        assert cfg.end_fraction == 1.0
        assert cfg.h_eff == 1.0
        assert cfg.mass == 1.0
        assert cfg.ell0 is None
        assert cfg.color_dims is None
        assert cfg.eps == 1e-12
        assert cfg.pair_selection == "both"

    def test_override_own_field(self):
        cfg = MesonOperatorConfig(operator_mode="score_directed")
        assert cfg.operator_mode == "score_directed"

    def test_override_base_and_own_fields(self):
        cfg = MesonOperatorConfig(
            warmup_fraction=0.3,
            mass=3.0,
            operator_mode="connected",
        )
        assert cfg.warmup_fraction == 0.3
        assert cfg.mass == 3.0
        assert cfg.operator_mode == "connected"

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(MesonOperatorConfig)

    def test_own_fields(self):
        # Should have all base fields plus operator_mode
        all_fields = {f.name for f in dataclasses.fields(MesonOperatorConfig)}
        base_fields = {f.name for f in dataclasses.fields(ChannelConfigBase)}
        own_fields = all_fields - base_fields
        assert own_fields == {"operator_mode"}

    def test_isinstance_of_base(self):
        cfg = MesonOperatorConfig()
        assert isinstance(cfg, ChannelConfigBase)


# ---------------------------------------------------------------------------
# TestVectorOperatorConfig
# ---------------------------------------------------------------------------


class TestVectorOperatorConfig:
    """Tests for VectorOperatorConfig."""

    def test_inherits_from_base(self):
        assert issubclass(VectorOperatorConfig, ChannelConfigBase)

    def test_default_values(self):
        cfg = VectorOperatorConfig()
        assert cfg.position_dims is None
        assert cfg.use_unit_displacement is False
        assert cfg.operator_mode == "standard"
        assert cfg.projection_mode == "full"

    def test_inherits_base_defaults(self):
        cfg = VectorOperatorConfig()
        assert cfg.warmup_fraction == 0.1
        assert cfg.eps == 1e-12
        assert cfg.pair_selection == "both"

    def test_override_position_dims(self):
        cfg = VectorOperatorConfig(position_dims=(0, 1, 2))
        assert cfg.position_dims == (0, 1, 2)

    def test_override_use_unit_displacement(self):
        cfg = VectorOperatorConfig(use_unit_displacement=True)
        assert cfg.use_unit_displacement is True

    def test_override_projection_mode(self):
        cfg = VectorOperatorConfig(projection_mode="transverse")
        assert cfg.projection_mode == "transverse"

    def test_all_overrides(self):
        cfg = VectorOperatorConfig(
            warmup_fraction=0.05,
            position_dims=(3, 4, 5),
            use_unit_displacement=True,
            operator_mode="score_directed",
            projection_mode="longitudinal",
        )
        assert cfg.warmup_fraction == 0.05
        assert cfg.position_dims == (3, 4, 5)
        assert cfg.use_unit_displacement is True
        assert cfg.operator_mode == "score_directed"
        assert cfg.projection_mode == "longitudinal"

    def test_own_fields(self):
        all_fields = {f.name for f in dataclasses.fields(VectorOperatorConfig)}
        base_fields = {f.name for f in dataclasses.fields(ChannelConfigBase)}
        own_fields = all_fields - base_fields
        assert own_fields == {
            "position_dims",
            "use_unit_displacement",
            "operator_mode",
            "projection_mode",
        }

    def test_isinstance_of_base(self):
        cfg = VectorOperatorConfig()
        assert isinstance(cfg, ChannelConfigBase)


# ---------------------------------------------------------------------------
# TestBaryonOperatorConfig
# ---------------------------------------------------------------------------


class TestBaryonOperatorConfig:
    """Tests for BaryonOperatorConfig."""

    def test_inherits_from_base(self):
        assert issubclass(BaryonOperatorConfig, ChannelConfigBase)

    def test_default_values(self):
        cfg = BaryonOperatorConfig()
        assert cfg.operator_mode == "det_abs"
        assert cfg.flux_exp_alpha == 1.0

    def test_inherits_base_defaults(self):
        cfg = BaryonOperatorConfig()
        assert cfg.warmup_fraction == 0.1
        assert cfg.mass == 1.0
        assert cfg.color_dims is None

    def test_override_operator_mode(self):
        for mode in ("det_abs", "det_real", "eps_abs", "eps_real"):
            cfg = BaryonOperatorConfig(operator_mode=mode)
            assert cfg.operator_mode == mode

    def test_override_flux_exp_alpha(self):
        cfg = BaryonOperatorConfig(flux_exp_alpha=2.5)
        assert cfg.flux_exp_alpha == 2.5

    def test_override_base_and_own(self):
        cfg = BaryonOperatorConfig(
            h_eff=0.5,
            operator_mode="eps_abs",
            flux_exp_alpha=0.1,
        )
        assert cfg.h_eff == 0.5
        assert cfg.operator_mode == "eps_abs"
        assert cfg.flux_exp_alpha == 0.1

    def test_own_fields(self):
        all_fields = {f.name for f in dataclasses.fields(BaryonOperatorConfig)}
        base_fields = {f.name for f in dataclasses.fields(ChannelConfigBase)}
        own_fields = all_fields - base_fields
        assert own_fields == {"operator_mode", "flux_exp_alpha"}

    def test_isinstance_of_base(self):
        cfg = BaryonOperatorConfig()
        assert isinstance(cfg, ChannelConfigBase)


# ---------------------------------------------------------------------------
# TestGlueballOperatorConfig
# ---------------------------------------------------------------------------


class TestGlueballOperatorConfig:
    """Tests for GlueballOperatorConfig."""

    def test_inherits_from_base(self):
        assert issubclass(GlueballOperatorConfig, ChannelConfigBase)

    def test_default_values(self):
        cfg = GlueballOperatorConfig()
        assert cfg.operator_mode is None
        assert cfg.use_action_form is False
        assert cfg.use_momentum_projection is False
        assert cfg.momentum_axis == 0
        assert cfg.momentum_mode_max == 3

    def test_inherits_base_defaults(self):
        cfg = GlueballOperatorConfig()
        assert cfg.warmup_fraction == 0.1
        assert cfg.end_fraction == 1.0
        assert cfg.eps == 1e-12

    def test_operator_mode_none_by_default(self):
        """Backward compatibility: operator_mode defaults to None."""
        cfg = GlueballOperatorConfig()
        assert cfg.operator_mode is None

    def test_set_operator_mode(self):
        cfg = GlueballOperatorConfig(operator_mode="plaquette")
        assert cfg.operator_mode == "plaquette"

    def test_action_form_flag(self):
        cfg = GlueballOperatorConfig(use_action_form=True)
        assert cfg.use_action_form is True

    def test_momentum_projection(self):
        cfg = GlueballOperatorConfig(
            use_momentum_projection=True,
            momentum_axis=2,
            momentum_mode_max=5,
        )
        assert cfg.use_momentum_projection is True
        assert cfg.momentum_axis == 2
        assert cfg.momentum_mode_max == 5

    def test_own_fields(self):
        all_fields = {f.name for f in dataclasses.fields(GlueballOperatorConfig)}
        base_fields = {f.name for f in dataclasses.fields(ChannelConfigBase)}
        own_fields = all_fields - base_fields
        assert own_fields == {
            "operator_mode",
            "use_action_form",
            "use_momentum_projection",
            "momentum_axis",
            "momentum_mode_max",
        }

    def test_isinstance_of_base(self):
        cfg = GlueballOperatorConfig()
        assert isinstance(cfg, ChannelConfigBase)


# ---------------------------------------------------------------------------
# TestTensorOperatorConfig
# ---------------------------------------------------------------------------


class TestTensorOperatorConfig:
    """Tests for TensorOperatorConfig."""

    def test_inherits_from_base(self):
        assert issubclass(TensorOperatorConfig, ChannelConfigBase)

    def test_default_values(self):
        cfg = TensorOperatorConfig()
        assert cfg.position_dims is None
        assert cfg.momentum_axis == 0
        assert cfg.momentum_mode_max == 4
        assert cfg.projection_length is None

    def test_inherits_base_defaults(self):
        cfg = TensorOperatorConfig()
        assert cfg.warmup_fraction == 0.1
        assert cfg.mass == 1.0
        assert cfg.pair_selection == "both"

    def test_override_position_dims(self):
        cfg = TensorOperatorConfig(position_dims=(0, 1, 2))
        assert cfg.position_dims == (0, 1, 2)

    def test_override_momentum_settings(self):
        cfg = TensorOperatorConfig(momentum_axis=1, momentum_mode_max=8)
        assert cfg.momentum_axis == 1
        assert cfg.momentum_mode_max == 8

    def test_override_projection_length(self):
        cfg = TensorOperatorConfig(projection_length=15.0)
        assert cfg.projection_length == 15.0

    def test_all_overrides(self):
        cfg = TensorOperatorConfig(
            warmup_fraction=0.2,
            mass=0.5,
            position_dims=(1, 2, 3),
            momentum_axis=2,
            momentum_mode_max=6,
            projection_length=20.0,
        )
        assert cfg.warmup_fraction == 0.2
        assert cfg.mass == 0.5
        assert cfg.position_dims == (1, 2, 3)
        assert cfg.momentum_axis == 2
        assert cfg.momentum_mode_max == 6
        assert cfg.projection_length == 20.0

    def test_own_fields(self):
        all_fields = {f.name for f in dataclasses.fields(TensorOperatorConfig)}
        base_fields = {f.name for f in dataclasses.fields(ChannelConfigBase)}
        own_fields = all_fields - base_fields
        assert own_fields == {
            "position_dims",
            "momentum_axis",
            "momentum_mode_max",
            "projection_length",
        }

    def test_isinstance_of_base(self):
        cfg = TensorOperatorConfig()
        assert isinstance(cfg, ChannelConfigBase)


# ---------------------------------------------------------------------------
# TestMultiscaleConfig
# ---------------------------------------------------------------------------


class TestMultiscaleConfig:
    """Tests for MultiscaleConfig (standalone, not a ChannelConfigBase subclass)."""

    def test_default_values(self):
        cfg = MultiscaleConfig()
        assert cfg.n_scales == 1
        assert cfg.mode == "companion"
        assert cfg.kernel_type == "gaussian"
        assert cfg.distance_method == "auto"
        assert cfg.distance_batch_size == 4
        assert cfg.scale_calibration_frames == 8
        assert cfg.scale_q_low == 0.05
        assert cfg.scale_q_high == 0.95
        assert cfg.max_scale_samples == 500_000
        assert cfg.min_scale == 1e-6
        assert cfg.scales is None
        assert cfg.edge_weight_mode == "riemannian_kernel_volume"

    def test_not_a_channel_config(self):
        """MultiscaleConfig should NOT inherit from ChannelConfigBase."""
        assert not issubclass(MultiscaleConfig, ChannelConfigBase)

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(MultiscaleConfig)

    def test_override_n_scales(self):
        cfg = MultiscaleConfig(n_scales=5)
        assert cfg.n_scales == 5

    def test_override_mode(self):
        for mode in ("companion", "kernel", "both"):
            cfg = MultiscaleConfig(mode=mode)
            assert cfg.mode == mode

    def test_override_kernel_type(self):
        for kt in ("gaussian", "exponential", "tophat", "shell"):
            cfg = MultiscaleConfig(kernel_type=kt)
            assert cfg.kernel_type == kt

    def test_override_distance_method(self):
        for dm in ("floyd-warshall", "tropical", "auto"):
            cfg = MultiscaleConfig(distance_method=dm)
            assert cfg.distance_method == dm

    def test_override_scale_quantiles(self):
        cfg = MultiscaleConfig(scale_q_low=0.1, scale_q_high=0.9)
        assert cfg.scale_q_low == 0.1
        assert cfg.scale_q_high == 0.9

    def test_user_specified_scales(self):
        scales = [0.5, 1.0, 2.0, 4.0]
        cfg = MultiscaleConfig(scales=scales)
        assert cfg.scales == [0.5, 1.0, 2.0, 4.0]

    def test_override_all_fields(self):
        cfg = MultiscaleConfig(
            n_scales=3,
            mode="both",
            kernel_type="exponential",
            distance_method="tropical",
            distance_batch_size=8,
            scale_calibration_frames=16,
            scale_q_low=0.01,
            scale_q_high=0.99,
            max_scale_samples=100_000,
            min_scale=1e-4,
            scales=[1.0, 2.0, 3.0],
            edge_weight_mode="kernel_only",
        )
        assert cfg.n_scales == 3
        assert cfg.mode == "both"
        assert cfg.kernel_type == "exponential"
        assert cfg.distance_method == "tropical"
        assert cfg.distance_batch_size == 8
        assert cfg.scale_calibration_frames == 16
        assert cfg.scale_q_low == 0.01
        assert cfg.scale_q_high == 0.99
        assert cfg.max_scale_samples == 100_000
        assert cfg.min_scale == 1e-4
        assert cfg.scales == [1.0, 2.0, 3.0]
        assert cfg.edge_weight_mode == "kernel_only"

    def test_fields_count(self):
        fields = dataclasses.fields(MultiscaleConfig)
        assert len(fields) == 12

    def test_equality(self):
        cfg1 = MultiscaleConfig()
        cfg2 = MultiscaleConfig()
        assert cfg1 == cfg2

    def test_inequality(self):
        cfg1 = MultiscaleConfig(n_scales=1)
        cfg2 = MultiscaleConfig(n_scales=3)
        assert cfg1 != cfg2


# ---------------------------------------------------------------------------
# TestCorrelatorConfig
# ---------------------------------------------------------------------------


class TestCorrelatorConfig:
    """Tests for CorrelatorConfig."""

    def test_default_values(self):
        cfg = CorrelatorConfig()
        assert cfg.max_lag == 80
        assert cfg.use_connected is True

    def test_not_a_channel_config(self):
        """CorrelatorConfig should NOT inherit from ChannelConfigBase."""
        assert not issubclass(CorrelatorConfig, ChannelConfigBase)

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(CorrelatorConfig)

    def test_override_max_lag(self):
        cfg = CorrelatorConfig(max_lag=200)
        assert cfg.max_lag == 200

    def test_override_use_connected(self):
        cfg = CorrelatorConfig(use_connected=False)
        assert cfg.use_connected is False

    def test_override_all(self):
        cfg = CorrelatorConfig(max_lag=50, use_connected=False)
        assert cfg.max_lag == 50
        assert cfg.use_connected is False

    def test_fields_count(self):
        fields = dataclasses.fields(CorrelatorConfig)
        assert len(fields) == 2

    def test_field_names(self):
        names = {f.name for f in dataclasses.fields(CorrelatorConfig)}
        assert names == {"max_lag", "use_connected"}

    def test_equality(self):
        cfg1 = CorrelatorConfig()
        cfg2 = CorrelatorConfig()
        assert cfg1 == cfg2

    def test_inequality(self):
        cfg1 = CorrelatorConfig(max_lag=80)
        cfg2 = CorrelatorConfig(max_lag=40)
        assert cfg1 != cfg2


# ---------------------------------------------------------------------------
# Cross-cutting / integration-style tests
# ---------------------------------------------------------------------------


class TestConfigCrossCutting:
    """Tests spanning multiple config classes."""

    @pytest.mark.parametrize(
        "cls",
        [
            MesonOperatorConfig,
            VectorOperatorConfig,
            BaryonOperatorConfig,
            GlueballOperatorConfig,
            TensorOperatorConfig,
        ],
    )
    def test_all_channel_configs_inherit_base(self, cls):
        assert issubclass(cls, ChannelConfigBase)
        cfg = cls()
        assert isinstance(cfg, ChannelConfigBase)

    @pytest.mark.parametrize(
        "cls",
        [
            MesonOperatorConfig,
            VectorOperatorConfig,
            BaryonOperatorConfig,
            GlueballOperatorConfig,
            TensorOperatorConfig,
        ],
    )
    def test_channel_configs_share_base_defaults(self, cls):
        cfg = cls()
        assert cfg.warmup_fraction == 0.1
        assert cfg.end_fraction == 1.0
        assert cfg.h_eff == 1.0
        assert cfg.mass == 1.0
        assert cfg.ell0 is None
        assert cfg.color_dims is None
        assert cfg.eps == 1e-12
        assert cfg.pair_selection == "both"

    @pytest.mark.parametrize(
        "cls",
        [
            MesonOperatorConfig,
            VectorOperatorConfig,
            BaryonOperatorConfig,
            GlueballOperatorConfig,
            TensorOperatorConfig,
        ],
    )
    def test_channel_configs_accept_base_overrides(self, cls):
        cfg = cls(warmup_fraction=0.5, eps=1e-6)
        assert cfg.warmup_fraction == 0.5
        assert cfg.eps == 1e-6

    @pytest.mark.parametrize(
        "cls",
        [
            ChannelConfigBase,
            MesonOperatorConfig,
            VectorOperatorConfig,
            BaryonOperatorConfig,
            GlueballOperatorConfig,
            TensorOperatorConfig,
            MultiscaleConfig,
            CorrelatorConfig,
        ],
    )
    def test_all_configs_are_dataclasses(self, cls):
        assert dataclasses.is_dataclass(cls)

    @pytest.mark.parametrize(
        "cls",
        [
            ChannelConfigBase,
            MesonOperatorConfig,
            VectorOperatorConfig,
            BaryonOperatorConfig,
            GlueballOperatorConfig,
            TensorOperatorConfig,
            MultiscaleConfig,
            CorrelatorConfig,
        ],
    )
    def test_all_configs_instantiate_with_no_args(self, cls):
        """Every config should have all-default fields (no required args)."""
        cfg = cls()
        assert cfg is not None

    @pytest.mark.parametrize(
        "cls",
        [
            ChannelConfigBase,
            MesonOperatorConfig,
            VectorOperatorConfig,
            BaryonOperatorConfig,
            GlueballOperatorConfig,
            TensorOperatorConfig,
            MultiscaleConfig,
            CorrelatorConfig,
        ],
    )
    def test_all_configs_support_equality(self, cls):
        assert cls() == cls()

    def test_standalone_configs_not_channel_subclasses(self):
        """MultiscaleConfig and CorrelatorConfig are standalone."""
        assert not issubclass(MultiscaleConfig, ChannelConfigBase)
        assert not issubclass(CorrelatorConfig, ChannelConfigBase)
