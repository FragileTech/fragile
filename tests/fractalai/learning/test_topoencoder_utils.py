"""Unit tests for utility functions in topoencoder_mnist.py."""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest
import torch
from torch import nn

from fragile.learning.topoencoder_mnist import (
    TopoEncoderConfig,
    _benchmarks_compatible,
    _compute_perplexity_from_assignments,
    compute_ami,
    compute_matching_hidden_dim,
    count_parameters,
    load_benchmarks,
    load_checkpoint,
    save_benchmarks,
    save_checkpoint,
)


# ==========================================
# count_parameters
# ==========================================

class TestCountParameters:
    def test_linear(self):
        model = nn.Linear(10, 5)
        assert count_parameters(model) == 55  # 10*5 + 5

    def test_sequential(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        # 4*8+8 + 8*2+2 = 32+8+16+2 = 58
        assert count_parameters(model) == 58


# ==========================================
# compute_matching_hidden_dim
# ==========================================

class TestComputeMatchingHiddenDim:
    def test_returns_at_least_16(self):
        result = compute_matching_hidden_dim(target_params=10, input_dim=784)
        assert result >= 16

    def test_reasonable_for_known_target(self):
        from fragile.core.benchmarks import StandardVQ

        model = StandardVQ(input_dim=784, hidden_dim=32, latent_dim=2, num_codes=64)
        target = count_parameters(model)
        result = compute_matching_hidden_dim(target_params=target, input_dim=784)
        assert abs(result - 32) <= 5

    def test_fallback_on_tiny_target(self):
        result = compute_matching_hidden_dim(target_params=0, input_dim=784)
        assert isinstance(result, int)
        assert result >= 16


# ==========================================
# _compute_perplexity_from_assignments
# ==========================================

class TestComputePerplexity:
    def test_uniform_assignments(self):
        assignments = torch.tensor([i % 10 for i in range(100)])
        perp = _compute_perplexity_from_assignments(assignments, num_charts=10)
        assert abs(perp - 10.0) < 0.1

    def test_single_chart_collapse(self):
        assignments = torch.zeros(100, dtype=torch.long)
        perp = _compute_perplexity_from_assignments(assignments, num_charts=10)
        assert abs(perp - 1.0) < 0.1

    def test_empty_tensor(self):
        assignments = torch.tensor([], dtype=torch.long)
        perp = _compute_perplexity_from_assignments(assignments, num_charts=10)
        assert perp == 0.0

    def test_two_chart_equal(self):
        assignments = torch.tensor([0] * 50 + [1] * 50)
        perp = _compute_perplexity_from_assignments(assignments, num_charts=10)
        assert abs(perp - 2.0) < 0.1


# ==========================================
# compute_ami
# ==========================================

class TestComputeAMI:
    def test_perfect_match(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        ami = compute_ami(labels, labels)
        assert abs(ami - 1.0) < 0.01

    def test_random_labels(self):
        rng = np.random.RandomState(42)
        true = rng.randint(0, 5, size=200)
        pred = rng.randint(0, 5, size=200)
        ami = compute_ami(true, pred)
        assert ami < 0.5


# ==========================================
# _benchmarks_compatible
# ==========================================

class TestBenchmarksCompatible:
    def _make_config(self, **overrides) -> TopoEncoderConfig:
        return TopoEncoderConfig(**overrides)

    def _config_to_bench_dict(self, config: TopoEncoderConfig) -> dict:
        return {
            "dataset": config.dataset,
            "input_dim": config.input_dim,
            "latent_dim": config.latent_dim,
            "num_codes_standard": config.num_codes_standard,
            "baseline_vision_preproc": config.baseline_vision_preproc,
            "baseline_attn": config.baseline_attn,
            "baseline_attn_tokens": config.baseline_attn_tokens,
            "baseline_attn_dim": config.baseline_attn_dim,
            "baseline_attn_heads": config.baseline_attn_heads,
            "baseline_attn_dropout": config.baseline_attn_dropout,
        }

    def test_matching_config(self):
        config = self._make_config()
        bench = self._config_to_bench_dict(config)
        assert _benchmarks_compatible(bench, config) is True

    def test_empty_config(self):
        config = self._make_config()
        assert _benchmarks_compatible({}, config) is False

    def test_input_dim_mismatch(self):
        config = self._make_config()
        bench = self._config_to_bench_dict(config)
        bench["input_dim"] = 3072
        assert _benchmarks_compatible(bench, config) is False

    def test_latent_dim_mismatch(self):
        config = self._make_config()
        bench = self._config_to_bench_dict(config)
        bench["latent_dim"] = 10
        assert _benchmarks_compatible(bench, config) is False

    def test_baseline_attn_mismatch(self):
        config = self._make_config(baseline_attn=False)
        bench = self._config_to_bench_dict(config)
        bench["baseline_attn"] = True
        assert _benchmarks_compatible(bench, config) is False


# ==========================================
# Checkpoint save/load
# ==========================================

class TestCheckpointSaveLoad:
    def _make_minimal_model(self):
        return nn.Linear(10, 5)

    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        config = TopoEncoderConfig(epochs=5, device="cpu")
        model = self._make_minimal_model()
        jump = nn.Linear(2, 2)
        snapshot = {"X_train": torch.randn(10, 784), "X_test": torch.randn(5, 784)}

        save_checkpoint(
            path=path, config=config, model_atlas=model, jump_op=jump,
            metrics={"mse_atlas": 0.5}, data_snapshot=snapshot, epoch=3,
        )
        loaded = load_checkpoint(path)

        assert loaded["epoch"] == 3
        assert "atlas" in loaded["state"]
        assert set(loaded["state"]["atlas"].keys()) == set(model.state_dict().keys())

    def test_data_snapshot_preserved(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        config = TopoEncoderConfig(device="cpu")
        model = nn.Linear(10, 5)
        jump = nn.Linear(2, 2)
        snapshot = {"X_train": torch.randn(50, 784), "X_test": torch.randn(10, 784)}

        save_checkpoint(
            path=path, config=config, model_atlas=model, jump_op=jump,
            metrics={}, data_snapshot=snapshot, epoch=0,
        )
        loaded = load_checkpoint(path)
        assert loaded["data"]["X_train"].shape == (50, 784)


# ==========================================
# Benchmarks save/load
# ==========================================

class TestBenchmarksSaveLoad:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "bench.pt")
        config = TopoEncoderConfig(device="cpu")
        model_std = nn.Linear(784, 32)

        save_benchmarks(path, config, model_std=model_std, model_ae=None, std_hidden_dim=32)
        loaded = load_benchmarks(path)

        assert "state" in loaded
        assert loaded["dims"]["std_hidden_dim"] == 32

    def test_noop_when_both_none(self, tmp_path):
        path = str(tmp_path / "bench.pt")
        config = TopoEncoderConfig(device="cpu")
        save_benchmarks(path, config, model_std=None, model_ae=None)
        assert not os.path.exists(path)


# ==========================================
# TopoEncoderConfig
# ==========================================

class TestTopoEncoderConfig:
    def test_default_values(self):
        config = TopoEncoderConfig()
        assert config.dataset == "mnist"
        assert config.epochs == 1000
        assert config.hidden_dim == 32

    def test_asdict_roundtrip(self):
        config = TopoEncoderConfig()
        d = dataclasses.asdict(config)
        assert isinstance(d, dict)
        assert "dataset" in d
        assert "epochs" in d
        assert "hidden_dim" in d
        assert d["dataset"] == "mnist"
