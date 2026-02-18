"""Unit tests for fragile.learning.data module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fragile.learning.data import (
    augment_inputs,
    create_data_snapshot,
    create_dataloaders,
    DataBundle,
    infer_vision_shape,
    load_dataset,
    restore_dataset,
)


# ==========================================
# infer_vision_shape
# ==========================================


class TestInferVisionShape:
    def test_mnist_auto(self):
        assert infer_vision_shape("MNIST", 0, 0, 0) == (1, 28, 28)

    def test_mnist_lowercase(self):
        assert infer_vision_shape("mnist", 0, 0, 0) == (1, 28, 28)

    def test_cifar10_auto(self):
        assert infer_vision_shape("CIFAR-10", 0, 0, 0) == (3, 32, 32)

    def test_cifar10_lowercase(self):
        assert infer_vision_shape("cifar10", 0, 0, 0) == (3, 32, 32)

    def test_explicit_dims_passthrough(self):
        assert infer_vision_shape("anything", 2, 64, 64) == (2, 64, 64)

    def test_unknown_dataset_zero_dims_raises(self):
        with pytest.raises(ValueError, match="vision_preproc requires valid"):
            infer_vision_shape("imagenet", 0, 0, 0)


# ==========================================
# load_dataset
# ==========================================


class TestLoadDataset:
    def test_mnist_shapes(self, mock_mnist):
        bundle = load_dataset("mnist", n_samples=100, test_split=0.2)
        assert bundle.X_train.shape == (80, 784)
        assert bundle.X_test.shape == (20, 784)
        assert len(bundle.labels_train) == 80
        assert len(bundle.labels_test) == 20
        assert len(bundle.colors_train) == 80
        assert len(bundle.colors_test) == 20
        assert bundle.input_dim == 784
        assert bundle.dataset_name == "MNIST"

    def test_cifar10_shapes(self, mock_cifar10):
        bundle = load_dataset("cifar10", n_samples=100, test_split=0.2)
        assert bundle.X_train.shape[1] == 3072
        assert bundle.dataset_name == "CIFAR-10"
        assert len(bundle.dataset_ids) == 10

    def test_labels_dtype_int64(self, mock_mnist):
        bundle = load_dataset("mnist", n_samples=50, test_split=0.2)
        assert bundle.labels_train.dtype == np.int64
        assert bundle.labels_test.dtype == np.int64

    def test_split_zero(self, mock_mnist):
        bundle = load_dataset("mnist", n_samples=100, test_split=0.0)
        assert len(bundle.X_train) == 100
        # When test_split=0, X_test is the full dataset
        assert len(bundle.X_test) == 100

    def test_split_near_one_capped(self, mock_mnist):
        bundle = load_dataset("mnist", n_samples=10, test_split=0.99)
        assert len(bundle.X_train) >= 1

    def test_invalid_split_raises(self, mock_mnist):
        with pytest.raises(ValueError, match="test_split must be"):
            load_dataset("mnist", n_samples=10, test_split=1.0)

    def test_negative_split_raises(self, mock_mnist):
        with pytest.raises(ValueError, match="test_split must be"):
            load_dataset("mnist", n_samples=10, test_split=-0.1)

    def test_unsupported_dataset_raises(self):
        with pytest.raises(ValueError, match="Unsupported dataset"):
            load_dataset("imagenet", n_samples=10, test_split=0.2)

    def test_vision_preproc_infers_shape(self, mock_mnist):
        bundle = load_dataset("mnist", n_samples=100, test_split=0.2, vision_preproc=True)
        assert bundle.vision_in_channels == 1
        assert bundle.vision_height == 28
        assert bundle.vision_width == 28

    def test_vision_mismatch_raises(self, mock_mnist):
        with pytest.raises(ValueError, match="vision_preproc shape does not match"):
            load_dataset(
                "mnist",
                n_samples=100,
                test_split=0.2,
                vision_preproc=True,
                vision_in_channels=3,
                vision_height=32,
                vision_width=32,
            )

    def test_no_vision_keeps_zeros(self, mock_mnist):
        bundle = load_dataset("mnist", n_samples=100, test_split=0.2, vision_preproc=False)
        assert bundle.vision_in_channels == 0
        assert bundle.vision_height == 0
        assert bundle.vision_width == 0


# ==========================================
# restore_dataset
# ==========================================


class TestRestoreDataset:
    def test_roundtrip_via_snapshot(self, mnist_bundle):
        snapshot = create_data_snapshot(mnist_bundle)
        restored = restore_dataset(snapshot, dataset_fallback="mnist")
        assert restored.X_train.shape == mnist_bundle.X_train.shape
        assert restored.X_test.shape == mnist_bundle.X_test.shape
        assert torch.equal(restored.X_train, mnist_bundle.X_train)
        assert torch.equal(restored.X_test, mnist_bundle.X_test)
        assert np.array_equal(restored.labels_train, mnist_bundle.labels_train)
        assert np.array_equal(restored.labels_test, mnist_bundle.labels_test)

    def test_fallback_dataset_name(self, mnist_bundle):
        snapshot = create_data_snapshot(mnist_bundle)
        del snapshot["dataset_name"]
        restored = restore_dataset(snapshot, dataset_fallback="custom_name")
        assert restored.dataset_name == "custom_name"

    def test_labels_full_concatenation(self, mnist_bundle):
        snapshot = create_data_snapshot(mnist_bundle)
        restored = restore_dataset(snapshot, dataset_fallback="mnist")
        expected_len = len(restored.labels_train) + len(restored.labels_test)
        assert len(restored.labels_full) == expected_len

    def test_empty_test_labels(self, mock_mnist):
        bundle = load_dataset("mnist", n_samples=100, test_split=0.0)
        snapshot = create_data_snapshot(bundle)
        # Force empty labels_test
        snapshot["labels_test"] = np.array([], dtype=np.int64)
        snapshot["X_test"] = torch.zeros(0, 784)
        snapshot["colors_test"] = np.array([], dtype=np.float32)
        restored = restore_dataset(snapshot, dataset_fallback="mnist")
        assert np.array_equal(restored.labels_full, restored.labels_train)


# ==========================================
# create_data_snapshot
# ==========================================


class TestCreateDataSnapshot:
    def test_snapshot_keys(self, mnist_bundle):
        snapshot = create_data_snapshot(mnist_bundle)
        expected = {
            "X_train",
            "X_test",
            "labels_train",
            "labels_test",
            "colors_train",
            "colors_test",
            "dataset_ids",
            "dataset_name",
        }
        assert set(snapshot.keys()) == expected

    def test_tensors_are_clones(self, mnist_bundle):
        snapshot = create_data_snapshot(mnist_bundle)
        assert snapshot["X_train"].data_ptr() != mnist_bundle.X_train.data_ptr()
        assert snapshot["X_test"].data_ptr() != mnist_bundle.X_test.data_ptr()

    def test_numpy_arrays_shared(self, mnist_bundle):
        snapshot = create_data_snapshot(mnist_bundle)
        assert snapshot["labels_train"] is mnist_bundle.labels_train
        assert snapshot["colors_train"] is mnist_bundle.colors_train


# ==========================================
# create_dataloaders
# ==========================================


class TestCreateDataloaders:
    def test_basic_batch_sizes(self, mnist_bundle):
        train_dl, test_dl = create_dataloaders(
            mnist_bundle,
            batch_size=32,
            eval_batch_size=16,
            device=torch.device("cpu"),
        )
        assert train_dl.batch_size == 32
        assert test_dl.batch_size == 16

    def test_full_batch_mode(self, mnist_bundle):
        train_dl, _ = create_dataloaders(
            mnist_bundle,
            batch_size=0,
            eval_batch_size=0,
            device=torch.device("cpu"),
        )
        assert train_dl.batch_size == len(mnist_bundle.X_train)

    def test_eval_auto_fallback(self, mnist_bundle):
        _train_dl, test_dl = create_dataloaders(
            mnist_bundle,
            batch_size=16,
            eval_batch_size=0,
            device=torch.device("cpu"),
        )
        # When eval_batch_size=0 and batch_size < len(X_test), uses batch_size
        assert test_dl.batch_size == 16

    def test_iteration_yields_three_tensors(self, mnist_bundle):
        train_dl, _ = create_dataloaders(
            mnist_bundle,
            batch_size=0,
            eval_batch_size=0,
            device=torch.device("cpu"),
        )
        batch = next(iter(train_dl))
        assert len(batch) == 3
        assert batch[0].dtype == torch.float32  # X
        assert batch[1].dtype == torch.int64  # labels
        assert batch[2].dtype == torch.float32  # colors

    @pytest.mark.parametrize("bs,ebs", [(0, 0), (32, 0), (0, 64), (32, 64)])
    def test_batch_size_combinations(self, mnist_bundle, bs, ebs):
        train_dl, test_dl = create_dataloaders(
            mnist_bundle,
            batch_size=bs,
            eval_batch_size=ebs,
            device=torch.device("cpu"),
        )
        # Just verify loaders are functional
        batch = next(iter(train_dl))
        assert batch[0].shape[0] > 0
        test_batch = next(iter(test_dl))
        assert test_batch[0].shape[0] > 0


# ==========================================
# augment_inputs
# ==========================================


class TestAugmentInputs:
    def test_output_shape(self):
        x = torch.randn(8, 784)
        aug = augment_inputs(x, "mnist")
        assert aug.shape == (8, 784)

    def test_noise_changes_values(self):
        torch.manual_seed(0)
        x = torch.randn(8, 784)
        aug = augment_inputs(x, "mnist", noise_std=0.1)
        assert not torch.equal(aug, x)

    def test_zero_noise_identity(self):
        x = torch.randn(8, 784)
        aug = augment_inputs(x, "mnist", noise_std=0.0)
        assert torch.equal(aug, x)

    def test_unsupported_dataset_raises(self):
        x = torch.randn(8, 784)
        with pytest.raises(ValueError, match="Unsupported dataset"):
            augment_inputs(x, "imagenet")
