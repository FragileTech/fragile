"""
Estimate lambda_gap from FractalSet spectral gaps.

This script builds a FractalSet from a RunHistory and estimates the spectral
gap across multiple timesteps. It reports a robust lambda_gap estimate along
with eta_time and kappa when rho is provided.

Usage:
    python src/experiments/estimate_lambda_gap.py --history-path outputs/..._history.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fragile.fractalai.core.history import RunHistory

try:
    from fragile.fractalai.core.fractal_set import FractalSet
    from fragile.fractalai.geometry.curvature import compute_graph_laplacian_eigenvalues

    HAS_FRACTAL_SET = True
    FRACTAL_SET_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    HAS_FRACTAL_SET = False
    FRACTAL_SET_ERROR = str(exc)


@dataclass
class LambdaGapConfig:
    fractal_set_stride: int = 20
    max_timesteps: int | None = 50
    k_eigenvalues: int = 5
    use_undirected: bool = True


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.ndarray, torch.Tensor)):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (value == float("inf") or value == float("-inf")):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, float) and value != value:
        return "nan"
    return value


def _downsample_history(history: RunHistory, stride: int) -> RunHistory:
    if stride <= 1 or history.n_recorded <= 1:
        return history

    indices = list(range(0, history.n_recorded, stride))
    if indices[-1] != history.n_recorded - 1:
        indices.append(history.n_recorded - 1)

    info_indices = [idx - 1 for idx in indices if idx > 0]

    data = history.to_dict()
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.shape[0] == history.n_recorded:
            data[key] = value[indices]
        elif value.shape[0] == history.n_recorded - 1:
            data[key] = value[info_indices]

    data["n_recorded"] = len(indices)
    data["record_every"] = history.record_every * stride
    data["recorded_steps"] = [history.recorded_steps[i] for i in indices]

    return RunHistory(**data)


def _select_timesteps(n_recorded: int, max_timesteps: int | None) -> list[int]:
    if n_recorded <= 0:
        return []
    if max_timesteps is None or max_timesteps <= 0 or max_timesteps >= n_recorded:
        return list(range(n_recorded))
    indices = np.linspace(0, n_recorded - 1, max_timesteps, dtype=int)
    return sorted(set(indices.tolist()))


def _build_neighbor_lists(fractal_set: FractalSet, timestep: int, undirected: bool) -> dict[int, list[int]]:
    alive_walkers = fractal_set.get_alive_walkers(timestep)
    alive_set = set(alive_walkers)
    neighbor_lists: dict[int, set[int]] = {walker_id: set() for walker_id in alive_walkers}

    ig_graph = fractal_set.get_ig_subgraph(timestep=timestep)
    for walker_id in alive_walkers:
        source_node = fractal_set.get_node_id(walker_id, timestep, stage="pre")
        if source_node not in ig_graph:
            continue
        for target_node in ig_graph.successors(source_node):
            target_walker = int(fractal_set.nodes["walker"][target_node].item())
            if target_walker not in alive_set:
                continue
            neighbor_lists[walker_id].add(target_walker)
            if undirected:
                neighbor_lists[target_walker].add(walker_id)

    return {k: sorted(v) for k, v in neighbor_lists.items()}


def estimate_lambda_gap(
    history: RunHistory,
    cfg: LambdaGapConfig,
) -> dict[str, Any]:
    if not HAS_FRACTAL_SET:
        raise RuntimeError(f"FractalSet unavailable: {FRACTAL_SET_ERROR}")

    history_small = _downsample_history(history, cfg.fractal_set_stride)
    fractal_set = FractalSet(history_small)

    timesteps = _select_timesteps(fractal_set.n_recorded, cfg.max_timesteps)
    spectral_gaps = []
    spectral_gaps_raw = []
    used_timesteps = []

    for t in timesteps:
        neighbor_lists = _build_neighbor_lists(fractal_set, t, cfg.use_undirected)
        if len(neighbor_lists) < 2:
            continue
        try:
            eigenvalues, _ = compute_graph_laplacian_eigenvalues(
                neighbor_lists, k=min(cfg.k_eigenvalues, len(neighbor_lists) - 1)
            )
        except Exception:
            continue
        if len(eigenvalues) < 2:
            continue
        gap = float(eigenvalues[1])
        spectral_gaps_raw.append(gap)
        spectral_gaps.append(max(0.0, gap))
        used_timesteps.append(t)

    if not spectral_gaps:
        lambda_gap = 0.0
        stats = {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p10": 0.0,
            "p90": 0.0,
        }
    else:
        arr = np.array(spectral_gaps, dtype=np.float64)
        stats = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }
        lambda_gap = stats["median"]

    return {
        "lambda_gap": float(lambda_gap),
        "spectral_gap_stats": stats,
        "spectral_gaps_raw": spectral_gaps_raw,
        "spectral_gaps": spectral_gaps,
        "timesteps": used_timesteps,
        "fractal_set_stride": cfg.fractal_set_stride,
        "n_recorded": fractal_set.n_recorded,
        "use_undirected": cfg.use_undirected,
    }


def _default_analysis_id(history_path: Path) -> str:
    name = history_path.stem
    if name.endswith("_history"):
        return name[: -len("_history")]
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate lambda_gap from FractalSet spectral gaps.")
    parser.add_argument("--history-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fractal_gas_potential_well_analysis"))
    parser.add_argument("--analysis-id", type=str, default=None)
    parser.add_argument("--fractal-set-stride", type=int, default=20)
    parser.add_argument("--max-timesteps", type=int, default=50)
    parser.add_argument("--k-eigenvalues", type=int, default=5)
    parser.add_argument("--directed", action="store_true", help="Use directed IG edges.")
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--hbar-eff", type=float, default=1.0)
    args = parser.parse_args()

    history = RunHistory.load(str(args.history_path))
    analysis_id = args.analysis_id or _default_analysis_id(args.history_path)
    cfg = LambdaGapConfig(
        fractal_set_stride=max(1, args.fractal_set_stride),
        max_timesteps=args.max_timesteps,
        k_eigenvalues=max(2, args.k_eigenvalues),
        use_undirected=not args.directed,
    )

    result = estimate_lambda_gap(history, cfg)

    tau = float(history.delta_t)
    lambda_gap = float(result["lambda_gap"])
    eta_time = tau * lambda_gap if lambda_gap > 0 else None
    kappa = None
    if args.rho is not None and lambda_gap > 0:
        kappa = 1.0 / (args.rho * args.hbar_eff * lambda_gap)

    result["tau"] = tau
    result["hbar_eff"] = float(args.hbar_eff)
    result["rho"] = float(args.rho) if args.rho is not None else None
    result["eta_time"] = eta_time
    result["kappa"] = kappa

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{analysis_id}_lambda_gap.json"
    output_path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True))

    print(f"lambda_gap: {result['lambda_gap']:.6g}")
    if eta_time is not None:
        print(f"eta_time: {eta_time:.6g}")
    if kappa is not None:
        print(f"kappa: {kappa:.6g}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
