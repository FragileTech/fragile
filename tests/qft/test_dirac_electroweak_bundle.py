"""Tests for unified Dirac/Electroweak observable bundle."""

from __future__ import annotations

import torch

from fragile.fractalai.qft.dirac_electroweak import (
    compute_dirac_electroweak_bundle,
    DiracElectroweakConfig,
)
from fragile.fractalai.qft.dirac_spectrum import DiracSpectrumConfig
from fragile.fractalai.qft.electroweak_channels import ElectroweakChannelConfig


class _HistoryStub:
    def __init__(self, *, n_recorded: int = 8, n_walkers: int = 10, dim: int = 3):
        self.n_recorded = n_recorded
        self.n_steps = n_recorded
        self.N = n_walkers
        self.d = dim
        self.delta_t = 0.1
        self.record_every = 1
        self.pbc = False
        self.bounds = None
        self.params = {"companion_selection_clone": {"epsilon": 0.9}}
        self.recorded_steps = list(range(n_recorded))

        self.x_before_clone = torch.randn(n_recorded, n_walkers, dim, dtype=torch.float64)
        self.v_before_clone = torch.randn(n_recorded, n_walkers, dim, dtype=torch.float64)
        self.fitness = torch.rand(n_recorded - 1, n_walkers, dtype=torch.float64) + 0.1
        self.alive_mask = torch.ones(n_recorded - 1, n_walkers, dtype=torch.bool)
        self.force_viscous = torch.randn(n_recorded - 1, n_walkers, dim, dtype=torch.float64)
        self.will_clone = torch.rand(n_recorded - 1, n_walkers) > 0.5

        # Electroweak channel path reuses companion data when neighbor graph is absent.
        self.companions_distance = torch.randint(0, n_walkers, (n_recorded - 1, n_walkers))
        self.companions_clone = torch.randint(0, n_walkers, (n_recorded - 1, n_walkers))
        self.neighbor_edges = None
        self.edge_weights = None

    def get_step_index(self, step: int) -> int:
        return self.recorded_steps.index(step)


def test_compute_dirac_electroweak_bundle_runs():
    history = _HistoryStub()
    ew_cfg = ElectroweakChannelConfig(
        warmup_fraction=0.1,
        end_fraction=1.0,
        max_lag=10,
        neighbor_method="auto",
        fit_mode="aic",
        fit_start=2,
        min_fit_points=2,
    )
    dirac_cfg = DiracSpectrumConfig(
        kernel_mode="phase_space",
        epsilon_clone=0.02,
        epsilon_c=0.8,
        lambda_alg=0.5,
        h_eff=1.0,
        min_sector_size=1,
    )
    cfg = DiracElectroweakConfig(
        electroweak=ew_cfg,
        electroweak_channels=["u1_phase", "su2_phase", "su2_doublet"],
        dirac=dirac_cfg,
        sigma_max_lag=10,
    )

    bundle = compute_dirac_electroweak_bundle(history, cfg)

    assert bundle.electroweak_output.n_valid_frames > 0
    assert bundle.electroweak_output.channel_results
    assert bundle.electron_component_result.n_samples > 0
    assert bundle.higgs_sigma_result.n_samples > 0
    assert bundle.dirac_result.full_singular_values.size > 0
    assert bundle.color_singlet_spectrum is not None
    assert bundle.higgs_vev > 0.0


def test_compute_dirac_electroweak_bundle_companion_topology_both():
    history = _HistoryStub()
    ew_cfg = ElectroweakChannelConfig(
        warmup_fraction=0.1,
        end_fraction=1.0,
        max_lag=10,
        neighbor_method="companions",
        companion_topology="both",
        fit_mode="aic",
        fit_start=2,
        min_fit_points=2,
    )
    dirac_cfg = DiracSpectrumConfig(
        kernel_mode="phase_space",
        epsilon_clone=0.02,
        epsilon_c=0.8,
        lambda_alg=0.5,
        h_eff=1.0,
        min_sector_size=1,
    )
    cfg = DiracElectroweakConfig(
        electroweak=ew_cfg,
        electroweak_channels=["u1_phase", "su2_phase", "su2_doublet"],
        dirac=dirac_cfg,
        sigma_max_lag=10,
    )

    bundle = compute_dirac_electroweak_bundle(history, cfg)

    assert bundle.electroweak_output.n_valid_frames > 0
    assert bundle.electroweak_output.channel_results
    assert bundle.electroweak_output.avg_edges >= float(history.N)
    assert bundle.electron_component_result.n_samples > 0


def test_compute_dirac_electroweak_bundle_directional_walker_split_channels():
    history = _HistoryStub()
    ew_cfg = ElectroweakChannelConfig(
        warmup_fraction=0.1,
        end_fraction=1.0,
        max_lag=10,
        neighbor_method="companions",
        companion_topology="both",
        su2_operator_mode="score_directed",
        enable_walker_type_split=True,
        walker_type_scope="frame_global",
        fit_mode="aic",
        fit_start=2,
        min_fit_points=2,
    )
    dirac_cfg = DiracSpectrumConfig(
        kernel_mode="phase_space",
        epsilon_clone=0.02,
        epsilon_c=0.8,
        lambda_alg=0.5,
        h_eff=1.0,
        min_sector_size=1,
    )
    cfg = DiracElectroweakConfig(
        electroweak=ew_cfg,
        electroweak_channels=[
            "su2_phase",
            "su2_phase_directed",
            "su2_phase_cloner",
            "su2_phase_resister",
            "su2_phase_persister",
        ],
        dirac=dirac_cfg,
        sigma_max_lag=10,
    )

    bundle = compute_dirac_electroweak_bundle(history, cfg)
    channel_results = bundle.electroweak_output.channel_results

    assert bundle.electroweak_output.n_valid_frames > 0
    assert "su2_phase" in channel_results
    assert "su2_phase_directed" in channel_results
    assert "su2_phase_cloner" in channel_results
    assert "su2_phase_resister" in channel_results
    assert "su2_phase_persister" in channel_results
