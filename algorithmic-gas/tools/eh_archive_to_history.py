"""Convert a native Einstein-Hilbert gas history into a ``RunHistory``.

``gas-eh --output run.json`` (or ``.cbor`` when ``cbor2`` is installed) writes
the walker state, the Einstein-Hilbert potential, the curvature and volume
fields, clone events and the tessellation graph of every recorded step. This
tool maps them onto the field names of
``fragile.physics.fractal_gas.history.RunHistory`` so the QFT analyzers
(Einstein equations, spectral gap, smeared operators, the qft-analysis skill)
read a native run unchanged.

Quantities the native history does not carry are zero-filled and listed under
``params["native_run"]["zero_filled"]``; the intermediate post-cloning state is
not exported, so ``x_after_clone`` repeats ``x_before_clone`` and is listed
under ``approximated``.

Usage:
    uv run python algorithmic-gas/tools/eh_archive_to_history.py run.json history.pt
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

from fragile.physics.fractal_gas.history import RunHistory


def _load(path: Path) -> dict:
    if path.suffix == ".json":
        return json.loads(path.read_text())
    try:
        import cbor2
    except ImportError as exc:  # pragma: no cover - optional dependency
        msg = "reading .cbor needs cbor2; write JSON with `gas-eh --output run.json` instead"
        raise SystemExit(msg) from exc
    with path.open("rb") as handle:
        return cbor2.load(handle)


def convert(native: dict) -> RunHistory:
    if native.get("schema") != "einstein-hilbert-gas-history/v1":
        msg = f"unsupported native history schema: {native.get('schema')!r}"
        raise ValueError(msg)
    config, frames = native["config"], native["frames"]
    gas = config["gas"]
    n, d = config["walkers"], config["dimensions"]
    dtype = torch.float32 if config["precision"] == "f32" else torch.float64
    steps = [frame["step"] for frame in frames]
    n_recorded = len(frames) + 1

    def per_walker(key: str, shape: tuple[int, ...] = ()) -> torch.Tensor:
        return torch.tensor([frame[key] for frame in frames], dtype=dtype).reshape(
            len(frames), n, *shape
        )

    def trajectory(key: str) -> torch.Tensor:
        initial = torch.tensor(native[f"initial_{key}"], dtype=dtype).reshape(1, n, d)
        return torch.cat([initial, per_walker(key, (d,))])

    x_final, v_final = trajectory("positions"), trajectory("velocities")
    # State entering step t is the state leaving step t-1.
    x_before = torch.cat([x_final[:1], x_final[:-1]])
    v_before = torch.cat([v_final[:1], v_final[:-1]])
    info = (len(frames), n)
    zeros = torch.zeros(info, dtype=dtype)
    zeros_vec = torch.zeros((*info, d), dtype=dtype)
    potential = per_walker("potential")
    will_clone = torch.tensor([frame["will_clone"] for frame in frames], dtype=torch.bool)
    zero_filled = [
        "cloning_scores",
        "cloning_probs",
        "clone_jitter",
        "clone_delta_x",
        "clone_delta_v",
        "distances",
        "pos_squared_differences",
        "vel_squared_differences",
        "rescaled_rewards",
        "rescaled_distances",
        "mu_rewards",
        "sigma_rewards",
        "mu_distances",
        "sigma_distances",
        "force_stable",
        "force_adapt",
        "force_viscous",
        "force_friction",
        "force_total",
        "noise",
        "step_times",
    ]
    graph = bool(frames and frames[0]["neighbor_edges"])
    return RunHistory(
        N=n,
        d=d,
        n_steps=config["steps"],
        n_recorded=n_recorded,
        record_every=config["record_every"],
        terminated_early=False,
        final_step=steps[-1] if steps else 0,
        recorded_steps=[0, *steps],
        delta_t=gas["dt"],
        params={
            "gas": {"N": n, "d": d, "eh_scale": gas["eh_scale"], "dtype": config["precision"]},
            "kinetic": {
                key: gas[key] for key in ("gamma", "temperature", "dt", "nu", "beta_curl")
            },
            "cloning": {
                key: gas[key]
                for key in ("clone_every", "p_max", "epsilon_clone", "sigma_x", "restitution")
            },
            "fitness": {key: gas[key] for key in ("alpha", "beta", "eta", "amplitude")},
            "native_run": {
                "engine": "algorithmic-gas",
                "summary": native["summary"],
                "zero_filled": zero_filled,
                "approximated": ["x_after_clone", "v_after_clone"],
            },
        },
        rng_seed=config["seed"],
        x_before_clone=x_before,
        v_before_clone=v_before,
        x_after_clone=x_before.clone(),
        v_after_clone=v_before.clone(),
        x_final=x_final,
        v_final=v_final,
        U_before=potential,
        U_after_clone=potential,
        U_final=potential,
        n_alive=torch.full((n_recorded,), n, dtype=torch.long),
        num_cloned=torch.cat([torch.zeros(1, dtype=torch.long), will_clone.sum(dim=1)]),
        step_times=torch.zeros(n_recorded, dtype=dtype),
        fitness=per_walker("fitness"),
        rewards=-potential,
        cloning_scores=zeros,
        cloning_probs=zeros,
        will_clone=will_clone,
        alive_mask=torch.ones(info, dtype=torch.bool),
        companions_distance=torch.tensor(
            [frame["companions_distance"] for frame in frames], dtype=torch.long
        ),
        companions_clone=torch.tensor(
            [frame["companions_clone"] for frame in frames], dtype=torch.long
        ),
        clone_jitter=zeros_vec,
        clone_delta_x=zeros_vec,
        clone_delta_v=zeros_vec,
        distances=zeros,
        z_rewards=per_walker("z_rewards"),
        z_distances=per_walker("z_distances"),
        pos_squared_differences=zeros,
        vel_squared_differences=zeros,
        rescaled_rewards=zeros,
        rescaled_distances=zeros,
        mu_rewards=torch.zeros(len(frames), dtype=dtype),
        sigma_rewards=torch.zeros(len(frames), dtype=dtype),
        mu_distances=torch.zeros(len(frames), dtype=dtype),
        sigma_distances=torch.zeros(len(frames), dtype=dtype),
        force_stable=zeros_vec,
        force_adapt=zeros_vec,
        force_viscous=zeros_vec,
        force_friction=zeros_vec,
        force_total=zeros_vec,
        noise=zeros_vec,
        riemannian_volume_weights=per_walker("volume_element"),
        ricci_scalar_proxy=per_walker("ricci_scalar"),
        diffusion_tensors_full=per_walker("diffusion", (d, d)),
        geodesic_edge_distances=(
            [torch.tensor(f["geodesic_edge_distances"], dtype=dtype) for f in frames]
            if graph
            else None
        ),
        neighbor_edges=(
            [torch.tensor(f["neighbor_edges"], dtype=torch.long).reshape(-1, 2) for f in frames]
            if graph
            else None
        ),
        edge_weights=(
            [
                {mode: torch.tensor(w, dtype=dtype) for mode, w in f["edge_weights"].items()}
                for f in frames
            ]
            if graph
            else None
        ),
        total_time=native["summary"]["seconds"],
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
