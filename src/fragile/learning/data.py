"""Dataset loading, splitting, and DataLoader creation for training scripts.

Provides a self-contained data pipeline that can be used by any training script
without depending on specific config dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fragile.datasets import CIFAR10_CLASSES, get_cifar10_data, get_mnist_data


@dataclass
class DataBundle:
    """Container for train/test data, labels, colors, and metadata."""

    X_train: torch.Tensor
    X_test: torch.Tensor
    labels_train: np.ndarray
    labels_test: np.ndarray
    colors_train: np.ndarray
    colors_test: np.ndarray
    dataset_ids: dict
    dataset_name: str
    labels_full: np.ndarray
    input_dim: int
    # Vision shape (set when vision_preproc is active)
    vision_in_channels: int = 0
    vision_height: int = 0
    vision_width: int = 0


def infer_vision_shape(
    dataset_name: str,
    vision_in_channels: int,
    vision_height: int,
    vision_width: int,
) -> tuple[int, int, int]:
    """Infer vision dimensions from dataset name if not already set.

    Returns:
        Tuple of (channels, height, width).

    Raises:
        ValueError: If dimensions cannot be inferred or are invalid.
    """
    if vision_in_channels <= 0 or vision_height <= 0 or vision_width <= 0:
        if dataset_name in {"cifar10", "CIFAR-10"}:
            vision_in_channels = 3
            vision_height = 32
            vision_width = 32
        elif dataset_name in {"mnist", "MNIST"}:
            vision_in_channels = 1
            vision_height = 28
            vision_width = 28
    expected = vision_in_channels * vision_height * vision_width
    if expected <= 0:
        msg = "vision_preproc requires valid vision_* dimensions."
        raise ValueError(msg)
    return vision_in_channels, vision_height, vision_width


def load_dataset(
    dataset: str,
    n_samples: int,
    test_split: float,
    vision_preproc: bool = False,
    baseline_vision_preproc: bool = False,
    vision_in_channels: int = 0,
    vision_height: int = 0,
    vision_width: int = 0,
) -> DataBundle:
    """Load a dataset, perform train/test split, and return a DataBundle.

    Args:
        dataset: Dataset name ("mnist" or "cifar10").
        n_samples: Number of samples to load.
        test_split: Fraction of data to use for testing (in [0.0, 1.0)).
        vision_preproc: Whether vision preprocessing is active.
        baseline_vision_preproc: Whether baseline vision preprocessing is active.
        vision_in_channels: Number of input channels (0 = auto-infer).
        vision_height: Image height (0 = auto-infer).
        vision_width: Image width (0 = auto-infer).

    Returns:
        DataBundle with all data and metadata.
    """
    if dataset == "mnist":
        X, labels, colors = get_mnist_data(n_samples)
        dataset_ids = {str(i): i for i in range(10)}
        dataset_name = "MNIST"
    elif dataset == "cifar10":
        X, labels, colors = get_cifar10_data(n_samples)
        dataset_ids = {name: idx for idx, name in enumerate(CIFAR10_CLASSES)}
        dataset_name = "CIFAR-10"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    input_dim = X.shape[1]
    labels = labels.astype(np.int64)
    labels_full = labels
    print(f"Loaded {len(X)} samples from {dataset_name}")
    print(f"Input dim: {input_dim}")

    if not (0.0 <= test_split < 1.0):
        msg = "test_split must be in [0.0, 1.0)."
        raise ValueError(msg)

    n_total = X.shape[0]
    test_size = max(1, int(n_total * test_split)) if test_split > 0 else 0
    if test_size >= n_total:
        test_size = max(1, n_total - 1)
    train_size = n_total - test_size
    perm = torch.randperm(n_total)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]
    train_idx_np = train_idx.numpy()
    test_idx_np = test_idx.numpy()

    X_train = X[train_idx]
    X_test = X[test_idx] if test_size > 0 else X
    labels_train = labels[train_idx_np]
    labels_test = labels[test_idx_np] if test_size > 0 else labels
    colors_train = colors[train_idx_np]
    colors_test = colors[test_idx_np] if test_size > 0 else colors

    print(f"Train/test split: {len(X_train)}/{len(X_test)} (test={test_split:.2f})")

    # Infer vision shape if needed
    v_channels, v_height, v_width = 0, 0, 0
    if vision_preproc or baseline_vision_preproc:
        v_channels, v_height, v_width = infer_vision_shape(
            dataset_name,
            vision_in_channels,
            vision_height,
            vision_width,
        )
        expected = v_channels * v_height * v_width
        if input_dim != expected:
            raise ValueError(
                f"vision_preproc shape does not match input_dim ({input_dim} vs {expected})."
            )

    return DataBundle(
        X_train=X_train,
        X_test=X_test,
        labels_train=labels_train,
        labels_test=labels_test,
        colors_train=colors_train,
        colors_test=colors_test,
        dataset_ids=dataset_ids,
        dataset_name=dataset_name,
        labels_full=labels_full,
        input_dim=input_dim,
        vision_in_channels=v_channels,
        vision_height=v_height,
        vision_width=v_width,
    )


def restore_dataset(
    data_snapshot: dict,
    dataset_fallback: str,
    vision_preproc: bool = False,
    baseline_vision_preproc: bool = False,
    vision_in_channels: int = 0,
    vision_height: int = 0,
    vision_width: int = 0,
) -> DataBundle:
    """Restore a DataBundle from a checkpoint's data snapshot.

    Args:
        data_snapshot: The ``checkpoint["data"]`` dict.
        dataset_fallback: Fallback dataset name if not stored in snapshot.
        vision_preproc: Whether vision preprocessing is active.
        baseline_vision_preproc: Whether baseline vision preprocessing is active.
        vision_in_channels: Number of input channels (0 = auto-infer).
        vision_height: Image height (0 = auto-infer).
        vision_width: Image width (0 = auto-infer).

    Returns:
        DataBundle with all data and metadata.
    """
    X_train = data_snapshot["X_train"]
    X_test = data_snapshot["X_test"]
    labels_train = data_snapshot["labels_train"]
    labels_test = data_snapshot["labels_test"]
    colors_train = data_snapshot["colors_train"]
    colors_test = data_snapshot["colors_test"]
    dataset_ids = data_snapshot.get("dataset_ids", {})
    dataset_name = data_snapshot.get("dataset_name", dataset_fallback)

    labels_full = np.concatenate([labels_train, labels_test]) if len(labels_test) else labels_train
    input_dim = X_train.shape[1]

    # Infer vision shape if needed
    v_channels, v_height, v_width = 0, 0, 0
    if vision_preproc or baseline_vision_preproc:
        v_channels, v_height, v_width = infer_vision_shape(
            dataset_name,
            vision_in_channels,
            vision_height,
            vision_width,
        )
        expected = v_channels * v_height * v_width
        if input_dim != expected:
            raise ValueError(
                f"vision_preproc shape does not match input_dim ({input_dim} vs {expected})."
            )

    print(f"Loaded {len(X_train) + len(X_test)} samples from {dataset_name}")
    print(f"Input dim: {input_dim}")

    return DataBundle(
        X_train=X_train,
        X_test=X_test,
        labels_train=labels_train,
        labels_test=labels_test,
        colors_train=colors_train,
        colors_test=colors_test,
        dataset_ids=dataset_ids,
        dataset_name=dataset_name,
        labels_full=labels_full,
        input_dim=input_dim,
        vision_in_channels=v_channels,
        vision_height=v_height,
        vision_width=v_width,
    )


def create_data_snapshot(bundle: DataBundle) -> dict:
    """Create a checkpoint-ready dict from a DataBundle."""
    return {
        "X_train": bundle.X_train.clone(),
        "X_test": bundle.X_test.clone(),
        "labels_train": bundle.labels_train,
        "labels_test": bundle.labels_test,
        "colors_train": bundle.colors_train,
        "colors_test": bundle.colors_test,
        "dataset_ids": bundle.dataset_ids,
        "dataset_name": bundle.dataset_name,
    }


def create_dataloaders(
    bundle: DataBundle,
    batch_size: int,
    eval_batch_size: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders from a DataBundle.

    Args:
        bundle: The DataBundle containing train/test data.
        batch_size: Training batch size (0 = full batch).
        eval_batch_size: Evaluation batch size (0 = auto).
        device: Target device (used for pin_memory decision).

    Returns:
        Tuple of (train_dataloader, test_dataloader).
    """
    labels_train_t = torch.from_numpy(bundle.labels_train).long()
    labels_test_t = torch.from_numpy(bundle.labels_test).long()
    colors_train_t = torch.from_numpy(bundle.colors_train).float()
    colors_test_t = torch.from_numpy(bundle.colors_test).float()

    dataset = TensorDataset(bundle.X_train, labels_train_t, colors_train_t)
    effective_batch_size = batch_size if batch_size > 0 else len(bundle.X_train)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )

    if eval_batch_size > 0:
        effective_eval_batch_size = eval_batch_size
    elif 0 < effective_batch_size < len(bundle.X_test):
        effective_eval_batch_size = effective_batch_size
    else:
        effective_eval_batch_size = min(256, len(bundle.X_test)) if len(bundle.X_test) else 1
    effective_eval_batch_size = (
        min(effective_eval_batch_size, len(bundle.X_test))
        if len(bundle.X_test)
        else effective_eval_batch_size
    )

    test_dataset = TensorDataset(bundle.X_test, labels_test_t, colors_test_t)
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_eval_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def augment_inputs(
    x: torch.Tensor,
    dataset: str,
    noise_std: float = 0.1,
    _rotation_max: float = 0.3,
) -> torch.Tensor:
    """Apply dataset-aware augmentation.

    Args:
        x: Input tensor [B, D]
        dataset: Dataset name
        noise_std: Standard deviation of additive noise
        _rotation_max: Maximum rotation angle in radians (unused for images)

    Returns:
        Augmented tensor [B, D]
    """
    if dataset not in {"mnist", "cifar10"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return x + torch.randn_like(x) * noise_std
