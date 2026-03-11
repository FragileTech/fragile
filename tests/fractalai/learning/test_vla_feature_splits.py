"""Tests for VLA feature-cache train/test splits and dashboard overlays."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import holoviews as hv
import numpy as np
import torch

from fragile.learning.plots import build_latent_scatter, plot_latent_3d
from fragile.learning.core.layers import TopoEncoderPrimitives
from fragile.learning.vla.dashboard import (
    build_symbol_options,
    infer_symbol_assignments,
    load_feature_cache,
)
from fragile.learning.vla.extract_features import VLAFeatureDataset, load_feature_cache_metadata


def _write_episode(cache_dir: Path, ep_id: int, n_frames: int = 3, feat_dim: int = 4) -> None:
    ep_dir = cache_dir / f"episode_{ep_id}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    base = torch.arange(n_frames * feat_dim, dtype=torch.float32).reshape(n_frames, feat_dim)
    torch.save(base + ep_id, ep_dir / "features.pt")
    torch.save(torch.zeros(n_frames, 2, dtype=torch.float32), ep_dir / "actions.pt")
    torch.save(torch.full((n_frames,), ep_id, dtype=torch.long), ep_dir / "task_indices.pt")


def test_feature_cache_metadata_backfills_train_test_split(tmp_path: Path) -> None:
    """Existing caches without split fields should derive a deterministic holdout split."""
    for ep_id in range(6):
        _write_episode(tmp_path, ep_id)

    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "dataset": "dummy",
                "feature_dim": 4,
                "episode_ids": list(range(6)),
                "held_out_test_episodes": 2,
            },
        ),
    )

    meta = load_feature_cache_metadata(tmp_path)

    assert meta["train_episode_ids"] == [0, 1, 2, 3]
    assert meta["test_episode_ids"] == [4, 5]
    assert meta["num_train_episodes"] == 4
    assert meta["num_test_episodes"] == 2


def test_vla_feature_dataset_respects_split_selection(tmp_path: Path) -> None:
    """The split-aware dataset should expose only the requested episodes."""
    for ep_id in range(5):
        _write_episode(tmp_path, ep_id, n_frames=4)

    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "episode_ids": list(range(5)),
                "held_out_test_episodes": 2,
                "train_episode_ids": [0, 1, 2],
                "test_episode_ids": [3, 4],
            },
        ),
    )

    train_ds = VLAFeatureDataset(tmp_path, sequence_length=1, split="train")
    test_ds = VLAFeatureDataset(tmp_path, sequence_length=1, split="test")

    assert train_ds.episode_ids == [0, 1, 2]
    assert test_ds.episode_ids == [3, 4]
    assert len(train_ds) == 12
    assert len(test_ds) == 8
    assert int(train_ds[0]["episode_id"]) == 0
    assert int(test_ds[0]["episode_id"]) == 3


def test_dashboard_cache_loader_emits_frame_split_labels(tmp_path: Path) -> None:
    """Dashboard cache loading should carry per-frame train/test labels."""
    for ep_id in range(4):
        _write_episode(tmp_path, ep_id, n_frames=2)

    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "episode_ids": [0, 1, 2, 3],
                "held_out_test_episodes": 1,
                "train_episode_ids": [0, 1, 2],
                "test_episode_ids": [3],
            },
        ),
    )

    cache = load_feature_cache(str(tmp_path))

    assert cache["features"].shape == (8, 4)
    assert cache["split_labels"].tolist() == [
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "test",
        "test",
    ]


def test_latent_plots_accept_split_marker_overlays() -> None:
    """2D and 3D latent views should render split markers without crashing."""
    z = np.array(
        [
            [0.1, 0.0, 0.2],
            [0.2, 0.1, 0.0],
            [0.0, 0.3, 0.1],
            [0.3, 0.2, 0.2],
        ],
        dtype=float,
    )
    labels = np.array([0, 1, 0, 1], dtype=int)
    charts = np.array([0, 1, 0, 1], dtype=int)
    correct = np.ones(4, dtype=int)
    split_labels = np.array(["train", "train", "test", "test"])

    scatter = build_latent_scatter(
        z,
        labels,
        charts,
        correct,
        color_by="chart",
        point_size=4,
        dim_i=0,
        dim_j=1,
        marker_groups=split_labels,
    )
    assert isinstance(scatter, hv.Overlay)

    fig = plot_latent_3d(
        z,
        labels,
        K_chart=charts,
        correct=correct,
        color_by="chart",
        point_size=4,
        marker_groups=split_labels,
    )
    trace_names = {trace.name for trace in fig.data}
    assert "train split" in trace_names
    assert "test split" in trace_names


def test_symbol_assignment_inference_and_option_building() -> None:
    """Dashboard symbol indexing should cover every frame and expose sorted options."""
    torch.manual_seed(7)
    model = TopoEncoderPrimitives(
        input_dim=4,
        hidden_dim=16,
        latent_dim=2,
        num_charts=3,
        codes_per_chart=4,
        soft_equiv_metric=True,
    )
    features = torch.randn(9, 4)

    symbol_index = infer_symbol_assignments(SimpleNamespace(encoder=model), features, batch_size=4)

    assert symbol_index["charts"].shape == (9,)
    assert symbol_index["codes"].shape == (9,)
    assert symbol_index["labels"].shape == (9,)
    assert all(label.startswith("c") and ":s" in label for label in symbol_index["labels"])

    options = build_symbol_options(symbol_index["labels"])
    assert options
    assert set(options.values()) == set(symbol_index["labels"])
