"""Convert a recorded Algorithmic Gas run with a geometry stage into a ``RunHistory``.

The input is the engine's ``RunArchive`` as written by

    gas-benchmark --einstein-hilbert --steps N --record run.json --record-graph

(``.cbor`` archives need ``cbor2``). The archive already holds everything the
QFT analyzers read: walker state before and after every step with the geometry
fields, rewards and fitness, companions and the clone plan, the recorded graph
forces and noise, and the tessellation graph with its edge arrays. This tool
only renames and reshapes them into
``fragile.physics.fractal_gas.history.RunHistory``.

Post-cloning positions are reconstructed from the clone plan (exact without
position jitter); post-cloning velocities come from the recorded collision
output. Fields with no counterpart in the archive are zero-filled and listed
under ``params["native_run"]["zero_filled"]``.

Usage:
    uv run python algorithmic-gas/tools/eh_archive_to_history.py run.json history.pt
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

from fragile.physics.fractal_gas.history import RunHistory


CURVATURE_PREFIX = "geometry.curvature."


def _load(path: Path) -> dict:
    if path.suffix == ".json":
        return json.loads(path.read_text())
    try:
        import cbor2
    except ImportError as exc:  # pragma: no cover - optional dependency
        msg = "reading .cbor needs cbor2; record JSON with `--record run.json` instead"
        raise SystemExit(msg) from exc
    with path.open("rb") as handle:
        return cbor2.load(handle)


def _field(population: dict, name: str, dtype: torch.dtype) -> torch.Tensor:
    field = population["observations"]["fields"][name]
    return torch.tensor(field["values"], dtype=dtype).reshape(field["rows"], *field["item_shape"])


def _evaluation(step: dict, stage: str, name: str, dtype: torch.dtype) -> torch.Tensor | None:
    for entry in step["field_evaluations"]:
        if entry["stage"] == stage and entry["field"] == name:
            return torch.tensor(entry["values"], dtype=dtype).reshape(
                entry["rows"], *entry["item_shape"]
            )
    return None


def _companions(batch: dict, sources: list[dict], n: int) -> torch.Tensor:
    """Slot of the first companion of every walker (itself when unmatched)."""
    out = torch.arange(n, dtype=torch.long)
    for i in range(n):
        if batch["valid"][i * batch["count"]]:
            out[i] = sources[batch["indices"][i * batch["count"]]]["slot"]
    return out


def convert(archive: dict, curvature: str | None = None) -> RunHistory:
    if archive.get("schema_version") != 2:
        msg = f"unsupported archive schema: {archive.get('schema_version')!r}"
        raise ValueError(msg)
    steps, gas = archive["steps"], archive["gas_config"]
    if not steps:
        msg = "the archive holds no recorded steps"
        raise ValueError(msg)
    if gas.get("geometry") is None:
        msg = "the recorded gas has no geometry stage"
        raise ValueError(msg)
    dtype = torch.float32 if gas["precision"] == "f32" else torch.float64
    integrator = gas["kinetic"]["integrator"]
    positions, velocities = integrator["positions"], integrator["velocities"]
    first = steps[0]["before"]
    names = [k for k in first["observations"]["fields"] if k.startswith(CURVATURE_PREFIX)]
    curvature_field = CURVATURE_PREFIX + curvature if curvature else names[0]
    x0 = _field(first, positions, dtype)
    n, d = x0.shape

    def stack(items: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(items)

    x_before = [_field(s["before"], positions, dtype) for s in steps]
    v_before = [_field(s["before"], velocities, dtype) for s in steps]
    x_final = [_field(s["final_population"], positions, dtype) for s in steps]
    v_final = [_field(s["final_population"], velocities, dtype) for s in steps]
    x_after, v_after, will_clone, probs = [], [], [], []
    for s, xb, vb in zip(steps, x_before, v_before):
        plan = s["report"]["clone_plan"]
        accepted = torch.tensor([c["accepted"] for c in plan["choices"]], dtype=torch.bool)
        donors = torch.tensor(
            [
                plan["sources"][c["donors"][0]["pool_index"]]["slot"] if c["donors"] else i
                for i, c in enumerate(plan["choices"])
            ],
            dtype=torch.long,
        )
        x_after.append(torch.where(accepted[:, None], xb[donors], xb))
        collided = _evaluation(s, "component_collision", "collision_output_velocity", dtype)
        v_after.append(collided if collided is not None else vb)
        will_clone.append(accepted)
        probs.append(torch.tensor([c["probability"] or 0.0 for c in plan["choices"]], dtype=dtype))

    def report(path: tuple[str, ...]) -> torch.Tensor:
        rows = []
        for s in steps:
            value = s["report"]
            for key in path:
                value = value[key]
            rows.append(torch.tensor(value, dtype=dtype))
        return stack(rows)

    def recorded(stage: str, name: str, shape: tuple[int, ...]) -> tuple[torch.Tensor, bool]:
        rows = [_evaluation(s, stage, name, dtype) for s in steps]
        if any(r is None for r in rows):
            return torch.zeros((len(steps), n, *shape), dtype=dtype), False
        return stack(rows), True

    info = (len(steps), n)
    zeros, zeros_vec = torch.zeros(info, dtype=dtype), torch.zeros((*info, d), dtype=dtype)
    force_viscous, has_viscous = recorded("B1", "viscous_force", (d,))
    force_total, has_total = recorded("B1", "total_force", (d,))
    noise, has_noise = recorded("O", "executed_noise", (d,))
    friction = integrator["friction"]
    zero_filled = [
        "cloning_scores",
        "clone_jitter",
        "pos_squared_differences",
        "vel_squared_differences",
        "rescaled_rewards",
        "rescaled_distances",
        "force_adapt",
        "step_times",
    ]
    zero_filled += [
        name
        for name, present in (
            ("force_viscous", has_viscous),
            ("force_total", has_total),
            ("noise", has_noise),
        )
        if not present
    ]
    graphs = [s.get("graph") for s in steps]
    has_graph = all(g is not None for g in graphs)

    def edges(graph: dict) -> torch.Tensor:
        offsets = torch.tensor(graph["graph"]["offsets"], dtype=torch.long)
        sources = torch.repeat_interleave(torch.arange(n), offsets[1:] - offsets[:-1])
        return torch.stack([sources, torch.tensor(graph["graph"]["neighbors"], dtype=torch.long)], 1)

    recorded_steps = [s["report"]["step"] for s in steps]
    x_clone_delta = stack(x_after) - stack(x_before)
    rewards_before = report(("pre_clone_rewards", "raw"))
    rewards_final = report(("final_rewards", "raw"))
    return RunHistory(
        N=n,
        d=d,
        n_steps=recorded_steps[-1],
        n_recorded=len(steps) + 1,
        record_every=1,
        terminated_early=False,
        final_step=recorded_steps[-1],
        recorded_steps=[recorded_steps[0] - 1, *recorded_steps],
        delta_t=integrator["dt"],
        params={
            "gas_config": gas,
            "native_run": {
                "engine": "algorithmic-gas",
                "providers": archive["providers"],
                "curvature_field": curvature_field,
                "zero_filled": zero_filled,
            },
        },
        rng_seed=gas["seed"],
        x_before_clone=torch.cat([x0[None], stack(x_before)]),
        v_before_clone=torch.cat([v_before[0][None], stack(v_before)]),
        x_after_clone=torch.cat([x0[None], stack(x_after)]),
        v_after_clone=torch.cat([v_before[0][None], stack(v_after)]),
        x_final=torch.cat([x0[None], stack(x_final)]),
        v_final=torch.cat([v_before[0][None], stack(v_final)]),
        U_before=-rewards_before,
        U_after_clone=-rewards_final,
        U_final=-rewards_final,
        n_alive=torch.tensor(
            [n, *[s["report"]["eligible"] for s in steps]], dtype=torch.long
        ),
        num_cloned=torch.tensor([0, *[s["report"]["clones"] for s in steps]], dtype=torch.long),
        step_times=torch.zeros(len(steps) + 1, dtype=dtype),
        fitness=report(("pre_clone_fitness", "fitness")),
        rewards=rewards_before,
        cloning_scores=zeros,
        cloning_probs=stack(probs),
        will_clone=stack(will_clone),
        alive_mask=stack(
            [torch.tensor(s["report"]["pre_clone_eligible"], dtype=torch.bool) for s in steps]
        ),
        companions_distance=stack(
            [
                _companions(s["report"]["distance_companions"], s["report"]["distance_sources"], n)
                for s in steps
            ]
        ),
        companions_clone=stack(
            [
                _companions(
                    s["report"]["cloning_companions"], s["report"]["clone_plan"]["sources"], n
                )
                for s in steps
            ]
        ),
        clone_jitter=zeros_vec,
        clone_delta_x=x_clone_delta,
        clone_delta_v=stack(v_after) - stack(v_before),
        distances=report(("pre_clone_fitness", "separation")),
        z_rewards=report(("pre_clone_fitness", "reward_z")),
        z_distances=report(("pre_clone_fitness", "diversity_z")),
        pos_squared_differences=zeros,
        vel_squared_differences=zeros,
        rescaled_rewards=zeros,
        rescaled_distances=zeros,
        mu_rewards=rewards_before.mean(dim=1),
        sigma_rewards=rewards_before.std(dim=1),
        mu_distances=report(("pre_clone_fitness", "separation")).mean(dim=1),
        sigma_distances=report(("pre_clone_fitness", "separation")).std(dim=1),
        force_stable=-recorded("B1", "potential_gradient", (d,))[0],
        force_adapt=zeros_vec,
        force_viscous=force_viscous,
        force_friction=-friction * stack(v_after),
        force_total=force_total,
        noise=noise,
        riemannian_volume_weights=stack(
            [_field(s["final_population"], "geometry.volume_element", dtype) for s in steps]
        ),
        ricci_scalar_proxy=stack(
            [_field(s["final_population"], curvature_field, dtype) for s in steps]
        ),
        diffusion_tensors_full=(
            stack([_field(s["final_population"], "geometry.diffusion", dtype) for s in steps])
            if "geometry.diffusion" in first["observations"]["fields"]
            else None
        ),
        geodesic_edge_distances=(
            [torch.tensor(g["geodesic_length"], dtype=dtype) for g in graphs]
            if has_graph
            else None
        ),
        neighbor_edges=[edges(g) for g in graphs] if has_graph else None,
        edge_weights=(
            [{k: torch.tensor(w, dtype=dtype) for k, w in g["weights"].items()} for g in graphs]
            if has_graph
            else None
        ),
        total_time=0.0,
        init_time=0.0,
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    history = convert(_load(Path(sys.argv[1])))
    history.save(sys.argv[2])
    print(history.summary())


if __name__ == "__main__":
    main()
