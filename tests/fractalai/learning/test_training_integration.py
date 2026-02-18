"""Integration tests for the TopoEncoder training pipeline.

These tests use real MNIST data (auto-downloaded) at minimal scale.
Each test runs in ~5-15s on CPU.
"""

from __future__ import annotations

import os

import pytest
import torch

from fragile.learning.topoencoder_mnist import (
    load_checkpoint,
    TopoEncoderConfig,
    train_benchmark,
)


def _make_config(tmp_path, **overrides) -> TopoEncoderConfig:
    defaults = {
        "n_samples": 50,
        "epochs": 2,
        "hidden_dim": 16,
        "num_charts": 3,
        "codes_per_chart": 4,
        "batch_size": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "outputs"),
        "save_every": 0,
        "log_every": 1,
        # Disable all optional features
        "disable_ae": True,
        "disable_vq": True,
        "enable_supervised": False,
        "enable_classifier_head": False,
        "enable_cifar_backbone": False,
        "mlflow": False,
        "use_scheduler": False,
        "covariant_attn": False,
        "baseline_attn": False,
        "baseline_vision_preproc": False,
        "vision_preproc": False,
        # Disable expensive losses
        "orbit_weight": 0.0,
        "vicreg_inv_weight": 0.0,
        "orthogonality_weight": 0.0,
        "code_entropy_weight": 0.0,
    }
    defaults.update(overrides)
    return TopoEncoderConfig(**defaults)


# ==========================================
# train_benchmark
# ==========================================


class TestTrainBenchmark:
    def test_minimal_training(self, tmp_path):
        config = _make_config(tmp_path)
        result = train_benchmark(config)

        assert "ami_atlas" in result
        assert "mse_atlas" in result
        assert "atlas_perplexity" in result
        assert "checkpoint_path" in result
        assert isinstance(result["mse_atlas"], float)
        assert result["mse_atlas"] >= 0

    def test_training_with_baselines(self, tmp_path):
        config = _make_config(tmp_path, disable_ae=False, disable_vq=False)
        result = train_benchmark(config)

        assert isinstance(result["ami_std"], float)
        assert isinstance(result["mse_std"], float)
        assert isinstance(result["ami_ae"], float)
        assert isinstance(result["mse_ae"], float)
        assert result["mse_std"] >= 0
        assert result["mse_ae"] >= 0


# ==========================================
# Resume checkpoint
# ==========================================


class TestResumeCheckpoint:
    def test_save_then_resume(self, tmp_path):
        config = _make_config(tmp_path, epochs=1, save_every=1)
        result1 = train_benchmark(config)
        ckpt_path = result1["checkpoint_path"]
        assert os.path.exists(ckpt_path)

        config2 = _make_config(tmp_path, epochs=2, resume_checkpoint=ckpt_path)
        result2 = train_benchmark(config2)
        assert isinstance(result2["mse_atlas"], float)

    def test_resume_past_epochs(self, tmp_path):
        config = _make_config(tmp_path, epochs=2, save_every=1)
        result1 = train_benchmark(config)
        ckpt_path = result1["checkpoint_path"]

        config2 = _make_config(tmp_path, epochs=1, resume_checkpoint=ckpt_path)
        result2 = train_benchmark(config2)
        assert result2["checkpoint_path"] == ckpt_path


# ==========================================
# Checkpoint data integrity
# ==========================================


class TestCheckpointDataIntegrity:
    def test_checkpoint_data_shapes(self, tmp_path):
        config = _make_config(tmp_path, save_every=1)
        result = train_benchmark(config)
        ckpt = load_checkpoint(result["checkpoint_path"])
        assert ckpt["data"]["dataset_name"] == "MNIST"
        assert ckpt["data"]["X_train"].shape[1] == 784
