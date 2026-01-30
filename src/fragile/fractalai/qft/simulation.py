"""
Fractal Gas potential well run for Fractal Set and lattice QFT analysis.

This script runs a simple Euclidean Gas in a quadratic potential well and saves
full RunHistory data for later FractalSet construction and analysis.

Defaults:
- N=1000 walkers
- n_steps=1000
- record_every=1 (record every step)
- Quadratic well U(x) = 0.5 * alpha * ||x||^2 with alpha=0.1
- Bounds: [-bounds_extent, bounds_extent]^d with bounds_extent=10
- Balanced phase-space distance (lambda_alg=1.0)
- Calibrated QFT parameters (epsilon_d, epsilon_c, epsilon_F, nu, rho, delta_t)

Usage:
    python src/experiments/fractal_gas_potential_well.py
    python src/experiments/fractal_gas_potential_well.py --n-steps 500 --record-every 5

Notes:
- This produces large output files. Increase record_every to reduce size.
- To enable local-gauge (rho) or Hessian data for advanced analysis, edit the
  config section below.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Callable

import torch

from fragile.fractalai.bounds import TorchBounds
from fragile.fractalai.core.cloning import CloneOperator
from fragile.fractalai.core.companion_selection import CompanionSelection
from fragile.fractalai.core.euclidean_gas import EuclideanGas
from fragile.fractalai.core.fitness import FitnessOperator
from fragile.fractalai.core.kinetic_operator import KineticOperator


@dataclass
class PotentialWellConfig:
    dims: int = 3
    alpha: float = 0.1
    bounds_extent: float = 10.0


@dataclass
class OperatorConfig:
    gamma: float = 1.0
    beta: float = 1.0
    delta_t: float = 0.01
    epsilon_F: float = 994.399
    use_fitness_force: bool = False
    use_potential_force: bool = False
    use_anisotropic_diffusion: bool = False
    diagonal_diffusion: bool = False
    epsilon_Sigma: float = 0.1
    nu: float = 0.948271
    use_viscous_coupling: bool = True
    viscous_length_scale: float = 0.00976705
    viscous_neighbor_mode: str = "all"
    viscous_neighbor_weighting: str = "kernel"
    viscous_neighbor_threshold: float | None = None
    viscous_neighbor_penalty: float = 1.1
    viscous_degree_cap: float | None = None
    beta_curl: float = 0.0
    use_velocity_squashing: bool = False
    V_alg: float = float("inf")
    companion_method: str = "uniform"
    companion_epsilon: float = 2.12029
    companion_epsilon_clone: float = 1.68419
    lambda_alg: float = 1.0
    exclude_self: bool = True
    p_max: float = 1.0
    epsilon_clone: float = 0.01
    sigma_x: float = 0.1
    alpha_restitution: float = 0.5
    fitness_alpha: float = 1.0
    fitness_beta: float = 1.0
    fitness_eta: float = 0.1
    fitness_A: float = 2.0
    fitness_sigma_min: float = 1e-8
    fitness_epsilon_dist: float = 1e-8
    fitness_rho: float | None = None
    fitness_grad_mode: str = "exact"
    fitness_detach_stats: bool = False
    fitness_detach_companions: bool = False
    diffusion_mode: str = "hessian"
    diffusion_grad_scale: float = 1.0


@dataclass
class RunConfig:
    N: int = 2000
    n_steps: int = 3000
    record_every: int = 1
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"
    pbc: bool = False
    record_rng_state: bool = True
    neighbor_graph_method: str = "delaunay"
    neighbor_graph_update_every: int = 1
    neighbor_graph_record: bool = True


class QuadraticPotential:
    def __init__(self, alpha: float, dims: int, bounds_extent: float) -> None:
        self.alpha = float(alpha)
        self.dims = int(dims)
        self.bounds = TorchBounds.from_tuples([(-bounds_extent, bounds_extent)] * self.dims)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.alpha * (x**2).sum(dim=-1)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (value == float("inf") or value == float("-inf")):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, float) and value != value:
        return "nan"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def build_gas(
    potential_cfg: PotentialWellConfig,
    operator_cfg: OperatorConfig,
    run_cfg: RunConfig,
    reward_1form: Callable[[torch.Tensor], torch.Tensor] | None = None,
    curl_field: Callable[[torch.Tensor], torch.Tensor] | None = None,
    reward_mode: str = "potential",
    voronoi_reward_update_every: int = 1,
) -> tuple[EuclideanGas, QuadraticPotential]:
    if reward_mode == "voronoi_volume":
        from fragile.fractalai.core.benchmarks import VoronoiCellVolume

        bounds = TorchBounds.from_tuples(
            [(-potential_cfg.bounds_extent, potential_cfg.bounds_extent)] * potential_cfg.dims
        )
        potential = VoronoiCellVolume(
            dims=potential_cfg.dims,
            bounds=bounds,
            update_every=voronoi_reward_update_every,
        )
    else:
        potential = QuadraticPotential(
            alpha=potential_cfg.alpha,
            dims=potential_cfg.dims,
            bounds_extent=potential_cfg.bounds_extent,
        )

    companion = CompanionSelection(
        method=operator_cfg.companion_method,
        epsilon=operator_cfg.companion_epsilon,
        lambda_alg=operator_cfg.lambda_alg,
        exclude_self=operator_cfg.exclude_self,
    )
    companion_clone = CompanionSelection(
        method=operator_cfg.companion_method,
        epsilon=operator_cfg.companion_epsilon_clone,
        lambda_alg=operator_cfg.lambda_alg,
        exclude_self=operator_cfg.exclude_self,
    )

    fitness_op = FitnessOperator(
        alpha=operator_cfg.fitness_alpha,
        beta=operator_cfg.fitness_beta,
        eta=operator_cfg.fitness_eta,
        lambda_alg=operator_cfg.lambda_alg,
        sigma_min=operator_cfg.fitness_sigma_min,
        epsilon_dist=operator_cfg.fitness_epsilon_dist,
        A=operator_cfg.fitness_A,
        rho=operator_cfg.fitness_rho,
        grad_mode=operator_cfg.fitness_grad_mode,
        detach_stats=operator_cfg.fitness_detach_stats,
        detach_companions=operator_cfg.fitness_detach_companions,
    )

    kinetic_op = KineticOperator(
        gamma=operator_cfg.gamma,
        beta=operator_cfg.beta,
        delta_t=operator_cfg.delta_t,
        epsilon_F=operator_cfg.epsilon_F,
        use_fitness_force=operator_cfg.use_fitness_force,
        use_potential_force=operator_cfg.use_potential_force,
        use_anisotropic_diffusion=operator_cfg.use_anisotropic_diffusion,
        diagonal_diffusion=operator_cfg.diagonal_diffusion,
        epsilon_Sigma=operator_cfg.epsilon_Sigma,
        diffusion_mode=operator_cfg.diffusion_mode,
        diffusion_grad_scale=operator_cfg.diffusion_grad_scale,
        nu=operator_cfg.nu,
        use_viscous_coupling=operator_cfg.use_viscous_coupling,
        viscous_length_scale=operator_cfg.viscous_length_scale,
        viscous_neighbor_mode=operator_cfg.viscous_neighbor_mode,
        viscous_neighbor_weighting=operator_cfg.viscous_neighbor_weighting,
        viscous_neighbor_threshold=operator_cfg.viscous_neighbor_threshold,
        viscous_neighbor_penalty=operator_cfg.viscous_neighbor_penalty,
        viscous_degree_cap=operator_cfg.viscous_degree_cap,
        beta_curl=operator_cfg.beta_curl,
        curl_field=curl_field,
        use_velocity_squashing=operator_cfg.use_velocity_squashing,
        V_alg=operator_cfg.V_alg,
        potential=potential,
    )

    cloning = CloneOperator(
        p_max=operator_cfg.p_max,
        epsilon_clone=operator_cfg.epsilon_clone,
        sigma_x=operator_cfg.sigma_x,
        alpha_restitution=operator_cfg.alpha_restitution,
    )

    gas = EuclideanGas(
        N=run_cfg.N,
        d=potential_cfg.dims,
        potential=potential,
        reward_1form=reward_1form,
        companion_selection=companion,
        companion_selection_clone=companion_clone,
        kinetic_op=kinetic_op,
        cloning=cloning,
        fitness_op=fitness_op,
        bounds=potential.bounds,
        device=torch.device(run_cfg.device),
        dtype=run_cfg.dtype,
        enable_cloning=True,
        enable_kinetic=True,
        pbc=run_cfg.pbc,
        neighbor_graph_method=run_cfg.neighbor_graph_method,
        neighbor_graph_update_every=run_cfg.neighbor_graph_update_every,
        neighbor_graph_record=run_cfg.neighbor_graph_record,
    )

    return gas, potential


def run_simulation(
    potential_cfg: PotentialWellConfig,
    operator_cfg: OperatorConfig,
    run_cfg: RunConfig,
    show_progress: bool = True,
    reward_1form: Callable[[torch.Tensor], torch.Tensor] | None = None,
    curl_field: Callable[[torch.Tensor], torch.Tensor] | None = None,
    reward_mode: str = "potential",
    voronoi_reward_update_every: int = 1,
) -> tuple[Any, QuadraticPotential]:
    gas, potential = build_gas(
        potential_cfg,
        operator_cfg,
        run_cfg,
        reward_1form=reward_1form,
        curl_field=curl_field,
        reward_mode=reward_mode,
        voronoi_reward_update_every=voronoi_reward_update_every,
    )
    history = gas.run(
        n_steps=run_cfg.n_steps,
        record_every=run_cfg.record_every,
        seed=run_cfg.seed,
        record_rng_state=run_cfg.record_rng_state,
        show_progress=show_progress,
    )
    return history, potential


def save_outputs(
    history: Any,
    output_dir: Path,
    run_id: str,
    potential_cfg: PotentialWellConfig,
    operator_cfg: OperatorConfig,
    run_cfg: RunConfig,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / f"{run_id}_history.pt"
    summary_path = output_dir / f"{run_id}_summary.txt"
    metadata_path = output_dir / f"{run_id}_config.json"

    history.save(str(history_path))
    summary_path.write_text(history.summary(), encoding="utf-8")

    metadata = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "potential": asdict(potential_cfg),
        "operators": asdict(operator_cfg),
        "run": asdict(run_cfg),
        "history_path": str(history_path),
    }
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2, sort_keys=True))

    return {
        "history": history_path,
        "summary": summary_path,
        "metadata": metadata_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/fractal_gas_potential_well")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--N", type=int, default=RunConfig.N)
    parser.add_argument("--n-steps", type=int, default=RunConfig.n_steps)
    parser.add_argument("--dims", type=int, default=PotentialWellConfig.dims)
    parser.add_argument("--record-every", type=int, default=RunConfig.record_every)
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument("--device", default=RunConfig.device)
    parser.add_argument("--dtype", choices=["float32", "float64"], default=RunConfig.dtype)
    parser.add_argument("--alpha", type=float, default=PotentialWellConfig.alpha)
    parser.add_argument("--bounds-extent", type=float, default=PotentialWellConfig.bounds_extent)
    parser.add_argument("--gamma", type=float, default=OperatorConfig.gamma)
    parser.add_argument("--beta", type=float, default=OperatorConfig.beta)
    parser.add_argument("--delta-t", type=float, default=OperatorConfig.delta_t)
    parser.add_argument("--epsilon-F", type=float, default=OperatorConfig.epsilon_F)
    parser.add_argument("--nu", type=float, default=OperatorConfig.nu)
    parser.add_argument(
        "--viscous-length-scale", type=float, default=OperatorConfig.viscous_length_scale
    )
    parser.add_argument(
        "--viscous-neighbor-mode",
        choices=["all", "nearest"],
        default=OperatorConfig.viscous_neighbor_mode,
        help="Neighbor mode for viscous coupling (all or nearest)",
    )
    parser.add_argument(
        "--viscous-neighbor-weighting",
        choices=["kernel", "uniform"],
        default=OperatorConfig.viscous_neighbor_weighting,
        help="Weighting for viscous neighbors (kernel or uniform)",
    )
    parser.add_argument(
        "--viscous-neighbor-threshold",
        type=float,
        default=OperatorConfig.viscous_neighbor_threshold,
        help="Kernel threshold for strong neighbor penalty (0-1)",
    )
    parser.add_argument(
        "--viscous-neighbor-penalty",
        type=float,
        default=OperatorConfig.viscous_neighbor_penalty,
        help="Penalty strength for extra strong neighbors",
    )
    parser.add_argument(
        "--viscous-degree-cap",
        type=float,
        default=OperatorConfig.viscous_degree_cap,
        help="Optional cap on viscous degree (saturates multi-neighbor coupling)",
    )
    parser.add_argument("--fitness-rho", type=float, default=OperatorConfig.fitness_rho)
    parser.add_argument(
        "--neighbor-graph-method",
        choices=["none", "delaunay", "voronoi"],
        default=RunConfig.neighbor_graph_method,
        help="Neighbor graph backend for viscous coupling/recording",
    )
    parser.add_argument(
        "--neighbor-graph-update-every",
        type=int,
        default=RunConfig.neighbor_graph_update_every,
        help="Recompute neighbor graph every k steps",
    )
    parser.add_argument(
        "--neighbor-graph-record",
        action="store_true",
        default=RunConfig.neighbor_graph_record,
        help="Record neighbor graph and Voronoi regions in RunHistory",
    )
    parser.add_argument(
        "--no-neighbor-graph-record",
        action="store_false",
        dest="neighbor_graph_record",
        help="Disable recording neighbor graph/Voronoi regions",
    )
    parser.add_argument(
        "--companion-epsilon",
        type=float,
        default=OperatorConfig.companion_epsilon,
        help="Diversity companion epsilon (epsilon_d)",
    )
    parser.add_argument(
        "--companion-epsilon-clone",
        type=float,
        default=OperatorConfig.companion_epsilon_clone,
        help="Cloning companion epsilon (epsilon_c)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    potential_cfg = PotentialWellConfig(
        dims=args.dims,
        alpha=args.alpha,
        bounds_extent=args.bounds_extent,
    )
    operator_cfg = OperatorConfig(
        gamma=args.gamma,
        beta=args.beta,
        delta_t=args.delta_t,
        epsilon_F=args.epsilon_F,
        nu=args.nu,
        viscous_length_scale=args.viscous_length_scale,
        viscous_neighbor_mode=args.viscous_neighbor_mode,
        viscous_neighbor_weighting=args.viscous_neighbor_weighting,
        viscous_neighbor_threshold=args.viscous_neighbor_threshold,
        viscous_neighbor_penalty=args.viscous_neighbor_penalty,
        viscous_degree_cap=args.viscous_degree_cap,
        fitness_rho=args.fitness_rho,
        companion_epsilon=args.companion_epsilon,
        companion_epsilon_clone=args.companion_epsilon_clone,
    )
    run_cfg = RunConfig(
        N=args.N,
        n_steps=args.n_steps,
        record_every=args.record_every,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        neighbor_graph_method=args.neighbor_graph_method,
        neighbor_graph_update_every=args.neighbor_graph_update_every,
        neighbor_graph_record=args.neighbor_graph_record,
    )

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)

    history, _potential = run_simulation(
        potential_cfg, operator_cfg, run_cfg, show_progress=not args.no_progress
    )
    paths = save_outputs(history, output_dir, run_id, potential_cfg, operator_cfg, run_cfg)

    print(history.summary())
    print("Saved:")
    print(f"  history: {paths['history']}")
    print(f"  summary: {paths['summary']}")
    print(f"  metadata: {paths['metadata']}")


if __name__ == "__main__":
    main()
