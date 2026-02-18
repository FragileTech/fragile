"""Optional MLflow integration for TopoEncoder training."""

import math
from dataclasses import asdict

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

from fragile.learning.config import TopoEncoderConfig


def start_mlflow_run(
    config: TopoEncoderConfig,
    extra_params: dict[str, object] | None = None,
) -> bool:
    if not config.mlflow:
        return False
    if not MLFLOW_AVAILABLE:
        print("MLflow logging requested but mlflow is not installed. Skipping MLflow.")
        return False
    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    if config.mlflow_experiment:
        mlflow.set_experiment(config.mlflow_experiment)
    run_name = config.mlflow_run_name or f"topoencoder_{config.dataset}"
    mlflow.start_run(run_name=run_name)
    params = asdict(config)
    if extra_params:
        params.update(extra_params)
    safe_params: dict[str, object] = {}
    for key, value in params.items():
        if isinstance(value, int | float | str | bool):
            safe_params[key] = value
        else:
            safe_params[key] = str(value)
    if safe_params:
        mlflow.log_params(safe_params)
    return True


def log_mlflow_metrics(
    metrics: dict[str, float],
    step: int,
    enabled: bool,
) -> None:
    if not enabled:
        return
    safe_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            safe_metrics[key] = val
    if safe_metrics:
        mlflow.log_metrics(safe_metrics, step=step)


def end_mlflow_run(enabled: bool) -> None:
    if enabled and MLFLOW_AVAILABLE:
        mlflow.end_run()


def log_mlflow_params(params: dict[str, object], enabled: bool) -> None:
    """Log additional params to an active MLflow run."""
    if not enabled or not MLFLOW_AVAILABLE:
        return
    mlflow.log_params(params)
