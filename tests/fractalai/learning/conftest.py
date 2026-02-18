"""Shared fixtures for learning tests."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
import torch

from fragile.learning.data import DataBundle, load_dataset


# ---------------------------------------------------------------------------
# Fake dataset helpers
# ---------------------------------------------------------------------------

def _make_fake_mnist(n_samples: int = 100) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    X = torch.from_numpy(rng.rand(n_samples, 784).astype(np.float32))
    labels = rng.randint(0, 10, size=n_samples).astype(np.int64)
    colors = rng.rand(n_samples).astype(np.float32)
    return X, labels, colors


def _make_fake_cifar10(n_samples: int = 100) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(43)
    X = torch.from_numpy(rng.rand(n_samples, 3072).astype(np.float32))
    labels = rng.randint(0, 10, size=n_samples).astype(np.int64)
    colors = rng.rand(n_samples).astype(np.float32)
    return X, labels, colors


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_mnist():
    with patch("fragile.learning.data.get_mnist_data", side_effect=_make_fake_mnist) as m:
        yield m


@pytest.fixture()
def mock_cifar10():
    with patch("fragile.learning.data.get_cifar10_data", side_effect=_make_fake_cifar10) as m:
        yield m


@pytest.fixture()
def mock_both_datasets(mock_mnist, mock_cifar10):
    yield mock_mnist, mock_cifar10


@pytest.fixture()
def mnist_bundle(mock_mnist) -> DataBundle:
    return load_dataset("mnist", n_samples=100, test_split=0.2)


@pytest.fixture()
def minimal_config(tmp_path):
    from fragile.learning.topoencoder_mnist import TopoEncoderConfig

    return TopoEncoderConfig(
        n_samples=100,
        epochs=2,
        hidden_dim=16,
        num_charts=3,
        codes_per_chart=4,
        batch_size=0,
        device="cpu",
        output_dir=str(tmp_path / "outputs"),
        # Disable all optional features
        disable_ae=True,
        disable_vq=True,
        enable_supervised=False,
        enable_classifier_head=False,
        enable_cifar_backbone=False,
        mlflow=False,
        use_scheduler=False,
        covariant_attn=False,
        baseline_attn=False,
        baseline_vision_preproc=False,
        vision_preproc=False,
        # Disable expensive losses
        orbit_weight=0.0,
        vicreg_inv_weight=0.0,
        orthogonality_weight=0.0,
        code_entropy_weight=0.0,
    )
