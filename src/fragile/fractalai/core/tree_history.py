"""TreeHistory: graph-backed history recorder compatible with RunHistory.

This module provides a ``TreeHistory`` class that wraps a ``DataTree`` from
``fragile.fractalai.core.tree`` and exposes the same field-access and recording
API as ``VectorizedHistoryRecorder`` / ``RunHistory``.

Each recorded timestep is stored as N graph nodes (one per walker) under a
shared root.  Node IDs are deterministic:

    node_id = step * (N + 1) + walker + 1

Dense tensors are lazily reconstructed from graph nodes and cached until the
next ``record_step()`` call invalidates the cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from fragile.fractalai.core.tree import DataTree, DEFAULT_ROOT_ID


if TYPE_CHECKING:
    from fragile.fractalai.bounds import TorchBounds
    from fragile.fractalai.core.euclidean_gas import SwarmState


# ---------------------------------------------------------------------------
# Field category constants – mirrors VectorizedHistoryRecorder
# ---------------------------------------------------------------------------

# Node attributes stored with the *full* time axis (present at step 0 too)
_FULL_NODE_FIELDS = (
    "x_before_clone",
    "v_before_clone",
    "U_before",
    "x_final",
    "v_final",
    "U_final",
)

# Node attributes stored only from step 1 onward (minus-one axis)
_MINUS_ONE_NODE_FIELDS = (
    "x_after_clone",
    "v_after_clone",
    "U_after_clone",
    "fitness",
    "rewards",
    "cloning_scores",
    "cloning_probs",
    "will_clone",
    "alive_mask",
    "companions_distance",
    "companions_clone",
    "clone_jitter",
    "clone_delta_x",
    "clone_delta_v",
    "distances",
    "z_rewards",
    "z_distances",
    "pos_squared_differences",
    "vel_squared_differences",
    "rescaled_rewards",
    "rescaled_distances",
)

# Optional node attributes (may be absent)
_OPTIONAL_NODE_FIELDS = (
    "fitness_gradients",
    "fitness_hessians_diag",
    "fitness_hessians_full",
    "sigma_reg_diag",
    "sigma_reg_full",
    "riemannian_volume_weights",
    "ricci_scalar_proxy",
    "diffusion_tensors_full",
)

# Step-level metadata fields stored in _step_metadata (full axis)
_FULL_STEP_META = ("n_alive",)

# Step-level metadata fields stored in _step_metadata (minus-one axis)
_MINUS_ONE_STEP_META = (
    "num_cloned",
    "step_time",
    "mu_rewards",
    "sigma_rewards",
    "mu_distances",
    "sigma_distances",
)

# Force / kinetic fields stored in _force_data (minus-one axis)
_FORCE_FIELDS = (
    "force_stable",
    "force_adapt",
    "force_viscous",
    "force_friction",
    "force_total",
    "noise",
)


def _node_id(step: int, walker: int, N: int) -> int:
    """Deterministic node ID for a (step, walker) pair."""
    return step * (N + 1) + walker + 1


class TreeHistory:
    """Graph-backed history recorder with the same API as RunHistory.

    Parameters
    ----------
    N : int
        Number of walkers.
    d : int
        Spatial dimension.
    device : str or torch.device
        Torch device for tensor storage.
    dtype : torch.dtype
        Float dtype for tensor storage.
    """

    def __init__(
        self,
        N: int,
        d: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.N = N
        self.d = d
        self.device = torch.device(device)
        self.dtype = dtype

        # Use empty node_names/edge_names so DataTree.append_leaf skips
        # key-presence validation (initial-state nodes don't have minus-one fields).
        self._tree = DataTree(
            names=[],
            node_names=[],
            edge_names=[],
        )

        self._n_recorded: int = 0  # number of recorded timesteps so far
        self._step_metadata: list[dict] = []  # one dict per recorded step
        self._force_data: list[dict] = []  # one dict per recorded step (minus-one)
        self._cache: dict[str, Tensor] = {}

        # Build metadata (filled by build())
        self._build_meta: dict | None = None

    # ------------------------------------------------------------------
    # Deterministic node helpers
    # ------------------------------------------------------------------

    def _nid(self, step: int, walker: int) -> int:
        return _node_id(step, walker, self.N)

    def _parent_nid(self, step: int, walker: int) -> int:
        if step == 0:
            return int(DEFAULT_ROOT_ID)
        return _node_id(step - 1, walker, self.N)

    # ------------------------------------------------------------------
    # Recording API (matches VectorizedHistoryRecorder)
    # ------------------------------------------------------------------

    def record_initial_state(
        self,
        state: SwarmState,
        n_alive: int,
        U_before: Tensor | None = None,
        U_final: Tensor | None = None,
    ) -> None:
        """Record initial state at t=0."""
        self._invalidate_cache()
        step = 0
        for w in range(self.N):
            nid = self._nid(step, w)
            node_data = {
                "x_before_clone": state.x[w].detach().cpu().numpy(),
                "v_before_clone": state.v[w].detach().cpu().numpy(),
                "U_before": (U_before[w].detach().cpu().item() if U_before is not None else 0.0),
                "x_final": state.x[w].detach().cpu().numpy(),
                "v_final": state.v[w].detach().cpu().numpy(),
                "U_final": (U_final[w].detach().cpu().item() if U_final is not None else 0.0),
            }
            parent_id = self._parent_nid(step, w)
            self._tree.append_leaf(
                leaf_id=nid,
                parent_id=parent_id,
                node_data=node_data,
                edge_data={},
                epoch=step,
            )

        self._step_metadata.append({"n_alive": n_alive})
        self._n_recorded = 1

    def record_step(
        self,
        state_before: SwarmState,
        state_cloned: SwarmState,
        state_final: SwarmState,
        info: dict,
        step_time: float,
        grad_fitness: Tensor | None = None,
        hess_fitness: Tensor | None = None,
        is_diagonal_hessian: bool = False,
        kinetic_info: dict | None = None,
    ) -> None:
        """Record a single step in-place (same signature as VectorizedHistoryRecorder)."""
        self._invalidate_cache()
        step = self._n_recorded
        alive_mask = info["alive_mask"]

        def _to_np(t: Tensor) -> object:
            if isinstance(t, Tensor):
                return t.detach().cpu().numpy()
            return t

        def _reduce_stat(value: Tensor | float, alive: Tensor) -> float:
            if isinstance(value, Tensor):
                if value.ndim > 0:
                    if bool(alive.any()):
                        value = value[alive]
                    if value.numel() == 0:
                        return 0.0
                    return float(value.mean().item())
                return float(value.item())
            return float(value)

        for w in range(self.N):
            nid = self._nid(step, w)
            parent_id = self._parent_nid(step, w)

            node_data: dict = {
                # Full-axis fields
                "x_before_clone": _to_np(state_before.x[w]),
                "v_before_clone": _to_np(state_before.v[w]),
                "U_before": float(info["U_before"][w].item())
                if isinstance(info["U_before"], Tensor)
                else float(info["U_before"]),
                "x_final": _to_np(state_final.x[w]),
                "v_final": _to_np(state_final.v[w]),
                "U_final": float(info["U_final"][w].item())
                if isinstance(info["U_final"], Tensor)
                else float(info["U_final"]),
                # Minus-one fields
                "x_after_clone": _to_np(state_cloned.x[w]),
                "v_after_clone": _to_np(state_cloned.v[w]),
                "U_after_clone": float(info["U_after_clone"][w].item())
                if isinstance(info["U_after_clone"], Tensor)
                else float(info["U_after_clone"]),
                "fitness": float(info["fitness"][w].item()),
                "rewards": float(info["rewards"][w].item()),
                "cloning_scores": float(info["cloning_scores"][w].item()),
                "cloning_probs": float(info["cloning_probs"][w].item()),
                "will_clone": bool(info["will_clone"][w].item()),
                "alive_mask": bool(alive_mask[w].item()),
                "companions_distance": int(info["companions_distance"][w].item()),
                "companions_clone": int(info["companions_clone"][w].item()),
                "clone_jitter": _to_np(info["clone_jitter"][w]),
                "clone_delta_x": _to_np(info["clone_delta_x"][w]),
                "clone_delta_v": _to_np(info["clone_delta_v"][w]),
                "distances": float(info["distances"][w].item()),
                "z_rewards": float(info["z_rewards"][w].item()),
                "z_distances": float(info["z_distances"][w].item()),
                "pos_squared_differences": float(info["pos_squared_differences"][w].item()),
                "vel_squared_differences": float(info["vel_squared_differences"][w].item()),
                "rescaled_rewards": float(info["rescaled_rewards"][w].item()),
                "rescaled_distances": float(info["rescaled_distances"][w].item()),
            }

            # Optional gradient/Hessian data
            if grad_fitness is not None:
                node_data["fitness_gradients"] = _to_np(grad_fitness[w])
            if hess_fitness is not None:
                if is_diagonal_hessian:
                    node_data["fitness_hessians_diag"] = _to_np(hess_fitness[w])
                else:
                    node_data["fitness_hessians_full"] = _to_np(hess_fitness[w])

            # Kinetic / force data stored on nodes
            if kinetic_info is not None:
                for fname in _FORCE_FIELDS:
                    if fname in kinetic_info:
                        node_data[fname] = _to_np(kinetic_info[fname][w])
                # Optional per-walker kinetic fields
                for opt_name in (
                    "sigma_reg_diag",
                    "sigma_reg_full",
                    "riemannian_volume_weights",
                    "ricci_scalar_proxy",
                    "diffusion_tensors_full",
                ):
                    if kinetic_info.get(opt_name) is not None:
                        val = kinetic_info[opt_name]
                        if isinstance(val, Tensor) and val.ndim >= 1:
                            node_data[opt_name] = _to_np(val[w])

            self._tree.append_leaf(
                leaf_id=nid,
                parent_id=parent_id,
                node_data=node_data,
                edge_data={},
                epoch=step,
            )

        # Step-level metadata
        self._step_metadata.append({
            "n_alive": int(alive_mask.sum().item()),
            "num_cloned": int(info["num_cloned"])
            if not isinstance(info["num_cloned"], Tensor)
            else int(info["num_cloned"].item()),
            "step_time": float(step_time),
            "mu_rewards": _reduce_stat(info["mu_rewards"], alive_mask),
            "sigma_rewards": _reduce_stat(info["sigma_rewards"], alive_mask),
            "mu_distances": _reduce_stat(info["mu_distances"], alive_mask),
            "sigma_distances": _reduce_stat(info["sigma_distances"], alive_mask),
        })

        # Force data (step-level, redundant with node-level but kept for metadata)
        force_entry: dict = {}
        if kinetic_info is not None:
            for fname in _FORCE_FIELDS:
                if fname in kinetic_info:
                    force_entry[fname] = True  # marker that data exists
        self._force_data.append(force_entry)

        self._n_recorded += 1

    def build(
        self,
        n_steps: int,
        record_every: int,
        terminated_early: bool,
        final_step: int,
        total_time: float,
        init_time: float,
        bounds: TorchBounds | None = None,
        recorded_steps: list[int] | None = None,
        delta_t: float | None = None,
        pbc: bool = False,
        params: dict | None = None,
        rng_seed: int | None = None,
        rng_state: dict | None = None,
    ) -> TreeHistory:
        """Store build metadata and return self (same pattern as VectorizedHistoryRecorder)."""
        self._build_meta = {
            "n_steps": n_steps,
            "record_every": record_every,
            "terminated_early": terminated_early,
            "final_step": final_step,
            "total_time": total_time,
            "init_time": init_time,
            "bounds": bounds,
            "recorded_steps": recorded_steps or [],
            "delta_t": delta_t or 0.0,
            "pbc": pbc,
            "params": params,
            "rng_seed": rng_seed,
            "rng_state": rng_state,
        }
        return self

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _invalidate_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------
    # Dense tensor reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_node_tensor(
        self,
        attr: str,
        shape_suffix: tuple[int, ...],
        start_step: int = 0,
        tensor_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Build a dense tensor from graph node attributes.

        Parameters
        ----------
        attr : str
            Node attribute name.
        shape_suffix : tuple[int, ...]
            Trailing shape dimensions after [n_steps, N].
        start_step : int
            First recorded step to include (0 or 1).
        tensor_dtype : torch.dtype or None
            Override dtype (defaults to self.dtype).
        """
        cache_key = f"node:{attr}:{start_step}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        n_steps = self._n_recorded - start_step
        if n_steps <= 0:
            t = torch.zeros(
                0, self.N, *shape_suffix, device=self.device, dtype=tensor_dtype or self.dtype
            )
            self._cache[cache_key] = t
            return t

        out = torch.zeros(
            n_steps, self.N, *shape_suffix, device=self.device, dtype=tensor_dtype or self.dtype
        )
        nodes = self._tree.data.nodes
        for si, step in enumerate(range(start_step, self._n_recorded)):
            for w in range(self.N):
                nid = self._nid(step, w)
                if nid in nodes and attr in nodes[nid]:
                    val = nodes[nid][attr]
                    out[si, w] = torch.as_tensor(val, dtype=tensor_dtype or self.dtype)

        self._cache[cache_key] = out
        return out

    def _reconstruct_step_meta(
        self, attr: str, start_step: int = 0, tensor_dtype: torch.dtype | None = None
    ) -> Tensor:
        """Build a 1-D tensor from step metadata."""
        cache_key = f"meta:{attr}:{start_step}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if start_step == 0:
            entries = self._step_metadata
        else:
            entries = self._step_metadata[start_step:]

        vals = [e.get(attr, 0) for e in entries]
        dt = tensor_dtype or self.dtype
        t = torch.tensor(vals, device=self.device, dtype=dt)
        self._cache[cache_key] = t
        return t

    # ------------------------------------------------------------------
    # Read API: properties matching RunHistory fields
    # ------------------------------------------------------------------

    # --- Full-axis state fields [n_recorded, N, d] ---

    @property
    def n_recorded(self) -> int:
        return self._n_recorded

    @property
    def x_before_clone(self) -> Tensor:
        return self._reconstruct_node_tensor("x_before_clone", (self.d,), start_step=0)

    @property
    def v_before_clone(self) -> Tensor:
        return self._reconstruct_node_tensor("v_before_clone", (self.d,), start_step=0)

    @property
    def U_before(self) -> Tensor:
        return self._reconstruct_node_tensor("U_before", (), start_step=0)

    @property
    def x_final(self) -> Tensor:
        return self._reconstruct_node_tensor("x_final", (self.d,), start_step=0)

    @property
    def v_final(self) -> Tensor:
        return self._reconstruct_node_tensor("v_final", (self.d,), start_step=0)

    @property
    def U_final(self) -> Tensor:
        return self._reconstruct_node_tensor("U_final", (), start_step=0)

    # --- Minus-one state fields [n_recorded-1, N, d] ---

    @property
    def x_after_clone(self) -> Tensor:
        return self._reconstruct_node_tensor("x_after_clone", (self.d,), start_step=1)

    @property
    def v_after_clone(self) -> Tensor:
        return self._reconstruct_node_tensor("v_after_clone", (self.d,), start_step=1)

    @property
    def U_after_clone(self) -> Tensor:
        return self._reconstruct_node_tensor("U_after_clone", (), start_step=1)

    # --- Minus-one per-walker scalar fields [n_recorded-1, N] ---

    @property
    def fitness(self) -> Tensor:
        return self._reconstruct_node_tensor("fitness", (), start_step=1)

    @property
    def rewards(self) -> Tensor:
        return self._reconstruct_node_tensor("rewards", (), start_step=1)

    @property
    def cloning_scores(self) -> Tensor:
        return self._reconstruct_node_tensor("cloning_scores", (), start_step=1)

    @property
    def cloning_probs(self) -> Tensor:
        return self._reconstruct_node_tensor("cloning_probs", (), start_step=1)

    @property
    def will_clone(self) -> Tensor:
        return self._reconstruct_node_tensor(
            "will_clone", (), start_step=1, tensor_dtype=torch.bool
        )

    @property
    def alive_mask(self) -> Tensor:
        return self._reconstruct_node_tensor(
            "alive_mask", (), start_step=1, tensor_dtype=torch.bool
        )

    @property
    def companions_distance(self) -> Tensor:
        return self._reconstruct_node_tensor(
            "companions_distance", (), start_step=1, tensor_dtype=torch.long
        )

    @property
    def companions_clone(self) -> Tensor:
        return self._reconstruct_node_tensor(
            "companions_clone", (), start_step=1, tensor_dtype=torch.long
        )

    @property
    def clone_jitter(self) -> Tensor:
        return self._reconstruct_node_tensor("clone_jitter", (self.d,), start_step=1)

    @property
    def clone_delta_x(self) -> Tensor:
        return self._reconstruct_node_tensor("clone_delta_x", (self.d,), start_step=1)

    @property
    def clone_delta_v(self) -> Tensor:
        return self._reconstruct_node_tensor("clone_delta_v", (self.d,), start_step=1)

    @property
    def distances(self) -> Tensor:
        return self._reconstruct_node_tensor("distances", (), start_step=1)

    @property
    def z_rewards(self) -> Tensor:
        return self._reconstruct_node_tensor("z_rewards", (), start_step=1)

    @property
    def z_distances(self) -> Tensor:
        return self._reconstruct_node_tensor("z_distances", (), start_step=1)

    @property
    def pos_squared_differences(self) -> Tensor:
        return self._reconstruct_node_tensor("pos_squared_differences", (), start_step=1)

    @property
    def vel_squared_differences(self) -> Tensor:
        return self._reconstruct_node_tensor("vel_squared_differences", (), start_step=1)

    @property
    def rescaled_rewards(self) -> Tensor:
        return self._reconstruct_node_tensor("rescaled_rewards", (), start_step=1)

    @property
    def rescaled_distances(self) -> Tensor:
        return self._reconstruct_node_tensor("rescaled_distances", (), start_step=1)

    # --- Step-level metadata ---

    @property
    def n_alive(self) -> Tensor:
        return self._reconstruct_step_meta("n_alive", start_step=0, tensor_dtype=torch.long)

    @property
    def num_cloned(self) -> Tensor:
        return self._reconstruct_step_meta("num_cloned", start_step=1, tensor_dtype=torch.long)

    @property
    def step_times(self) -> Tensor:
        return self._reconstruct_step_meta("step_time", start_step=1, tensor_dtype=torch.float32)

    @property
    def mu_rewards(self) -> Tensor:
        return self._reconstruct_step_meta("mu_rewards", start_step=1)

    @property
    def sigma_rewards(self) -> Tensor:
        return self._reconstruct_step_meta("sigma_rewards", start_step=1)

    @property
    def mu_distances(self) -> Tensor:
        return self._reconstruct_step_meta("mu_distances", start_step=1)

    @property
    def sigma_distances(self) -> Tensor:
        return self._reconstruct_step_meta("sigma_distances", start_step=1)

    # --- Force / kinetic fields [n_recorded-1, N, d] ---

    @property
    def force_stable(self) -> Tensor:
        return self._reconstruct_node_tensor("force_stable", (self.d,), start_step=1)

    @property
    def force_adapt(self) -> Tensor:
        return self._reconstruct_node_tensor("force_adapt", (self.d,), start_step=1)

    @property
    def force_viscous(self) -> Tensor:
        return self._reconstruct_node_tensor("force_viscous", (self.d,), start_step=1)

    @property
    def force_friction(self) -> Tensor:
        return self._reconstruct_node_tensor("force_friction", (self.d,), start_step=1)

    @property
    def force_total(self) -> Tensor:
        return self._reconstruct_node_tensor("force_total", (self.d,), start_step=1)

    @property
    def noise(self) -> Tensor:
        return self._reconstruct_node_tensor("noise", (self.d,), start_step=1)

    # --- Optional adaptive kinetics ---

    @property
    def fitness_gradients(self) -> Tensor | None:
        t = self._reconstruct_node_tensor("fitness_gradients", (self.d,), start_step=1)
        return t if t.numel() > 0 and t.abs().sum() > 0 else None

    @property
    def fitness_hessians_diag(self) -> Tensor | None:
        t = self._reconstruct_node_tensor("fitness_hessians_diag", (self.d,), start_step=1)
        return t if t.numel() > 0 and t.abs().sum() > 0 else None

    @property
    def fitness_hessians_full(self) -> Tensor | None:
        t = self._reconstruct_node_tensor("fitness_hessians_full", (self.d, self.d), start_step=1)
        return t if t.numel() > 0 and t.abs().sum() > 0 else None

    @property
    def sigma_reg_diag(self) -> Tensor | None:
        t = self._reconstruct_node_tensor("sigma_reg_diag", (self.d,), start_step=1)
        return t if t.numel() > 0 and t.abs().sum() > 0 else None

    @property
    def sigma_reg_full(self) -> Tensor | None:
        t = self._reconstruct_node_tensor("sigma_reg_full", (self.d, self.d), start_step=1)
        return t if t.numel() > 0 and t.abs().sum() > 0 else None

    # ------------------------------------------------------------------
    # Query API (matches RunHistory)
    # ------------------------------------------------------------------

    @property
    def recorded_steps(self) -> list[int]:
        if self._build_meta is not None:
            return self._build_meta["recorded_steps"]
        return list(range(self._n_recorded))

    def get_step_index(self, step: int) -> int:
        rs = self.recorded_steps
        if step not in rs:
            msg = f"Step {step} was not recorded"
            raise ValueError(msg)
        return rs.index(step)

    def get_walker_trajectory(self, walker_idx: int, stage: str = "final") -> dict:
        if stage == "before_clone":
            return {
                "x": self.x_before_clone[:, walker_idx, :],
                "v": self.v_before_clone[:, walker_idx, :],
            }
        if stage == "after_clone":
            return {
                "x": self.x_after_clone[:, walker_idx, :],
                "v": self.v_after_clone[:, walker_idx, :],
            }
        if stage == "final":
            return {
                "x": self.x_final[:, walker_idx, :],
                "v": self.v_final[:, walker_idx, :],
            }
        msg = f"Unknown stage: {stage}. Must be 'before_clone', 'after_clone', or 'final'"
        raise ValueError(msg)

    def get_clone_events(self) -> list[tuple[int, int, int]]:
        events = []
        wc = self.will_clone
        cc = self.companions_clone
        rs = self.recorded_steps
        for t in range(wc.shape[0]):
            cloners = torch.where(wc[t])[0]
            for i in cloners:
                companion = cc[t, i].item()
                step = rs[t + 1] if t + 1 < len(rs) else t + 1
                events.append((step, i.item(), companion))
        return events

    def get_alive_walkers(self, step: int) -> Tensor:
        idx = self.get_step_index(step)
        if idx == 0:
            pbc = self._build_meta.get("pbc", False) if self._build_meta else False
            if pbc:
                return torch.arange(self.N, device=self.device)
            bounds = self._build_meta.get("bounds") if self._build_meta else None
            if bounds is not None:
                return torch.where(bounds.contains(self.x_before_clone[0]))[0]
            return torch.arange(self.N, device=self.device)
        return torch.where(self.alive_mask[idx - 1])[0]

    def summary(self) -> str:
        meta = self._build_meta or {}
        n_steps = meta.get("n_steps", self._n_recorded - 1)
        record_every = meta.get("record_every", 1)
        terminated_early = meta.get("terminated_early", False)
        final_step = meta.get("final_step", n_steps)
        total_time = meta.get("total_time", 0.0)

        lines = [
            f"TreeHistory: {n_steps} steps, {self.N} walkers, {self.d}D",
            f"  Recorded: {self._n_recorded} timesteps (every {record_every} steps)",
            f"  Final step: {final_step} (terminated_early={terminated_early})",
            f"  Total cloning events: {self.will_clone.sum().item() if self._n_recorded > 1 else 0}",
            f"  Timing: {total_time:.3f}s total"
            + (f", {total_time / n_steps:.4f}s/step" if n_steps > 0 else ""),
            f"  Graph nodes: {self._tree.data.number_of_nodes()}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Conversion to RunHistory
    # ------------------------------------------------------------------

    def to_run_history(self):
        """Materialize all fields and construct a proper RunHistory."""
        from fragile.fractalai.core.history import RunHistory

        meta = self._build_meta or {}
        return RunHistory(
            N=self.N,
            d=self.d,
            n_steps=meta.get("n_steps", self._n_recorded - 1),
            n_recorded=self._n_recorded,
            record_every=meta.get("record_every", 1),
            terminated_early=meta.get("terminated_early", False),
            final_step=meta.get("final_step", self._n_recorded - 1),
            recorded_steps=self.recorded_steps,
            delta_t=meta.get("delta_t", 0.0),
            pbc=meta.get("pbc", False),
            params=meta.get("params"),
            rng_seed=meta.get("rng_seed"),
            rng_state=meta.get("rng_state"),
            bounds=meta.get("bounds"),
            x_before_clone=self.x_before_clone,
            v_before_clone=self.v_before_clone,
            U_before=self.U_before,
            x_after_clone=self.x_after_clone,
            v_after_clone=self.v_after_clone,
            U_after_clone=self.U_after_clone,
            x_final=self.x_final,
            v_final=self.v_final,
            U_final=self.U_final,
            n_alive=self.n_alive,
            num_cloned=self.num_cloned,
            step_times=self.step_times,
            fitness=self.fitness,
            rewards=self.rewards,
            cloning_scores=self.cloning_scores,
            cloning_probs=self.cloning_probs,
            will_clone=self.will_clone,
            alive_mask=self.alive_mask,
            companions_distance=self.companions_distance,
            companions_clone=self.companions_clone,
            clone_jitter=self.clone_jitter,
            clone_delta_x=self.clone_delta_x,
            clone_delta_v=self.clone_delta_v,
            distances=self.distances,
            z_rewards=self.z_rewards,
            z_distances=self.z_distances,
            pos_squared_differences=self.pos_squared_differences,
            vel_squared_differences=self.vel_squared_differences,
            rescaled_rewards=self.rescaled_rewards,
            rescaled_distances=self.rescaled_distances,
            mu_rewards=self.mu_rewards,
            sigma_rewards=self.sigma_rewards,
            mu_distances=self.mu_distances,
            sigma_distances=self.sigma_distances,
            fitness_gradients=self.fitness_gradients,
            fitness_hessians_diag=self.fitness_hessians_diag,
            fitness_hessians_full=self.fitness_hessians_full,
            force_stable=self.force_stable,
            force_adapt=self.force_adapt,
            force_viscous=self.force_viscous,
            force_friction=self.force_friction,
            force_total=self.force_total,
            noise=self.noise,
            sigma_reg_diag=self.sigma_reg_diag,
            sigma_reg_full=self.sigma_reg_full,
            total_time=meta.get("total_time", 0.0),
            init_time=meta.get("init_time", 0.0),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        d: dict = {
            "N": self.N,
            "d": self.d,
            "n_recorded": self._n_recorded,
            "step_metadata": self._step_metadata,
            "force_data": self._force_data,
            "build_meta": self._build_meta,
            # Store the raw graph node data
            "graph_nodes": dict(self._tree.data.nodes(data=True)),
            "graph_edges": list(self._tree.data.edges()),
        }
        return d

    def save(self, path: str) -> None:
        """Save to disk using torch.save."""
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str) -> TreeHistory:
        """Load from disk."""
        data = torch.load(path, weights_only=False)
        hist = cls(N=data["N"], d=data["d"])
        hist._n_recorded = data["n_recorded"]
        hist._step_metadata = data["step_metadata"]
        hist._force_data = data["force_data"]
        hist._build_meta = data["build_meta"]
        # Restore graph
        hist._tree.data.clear()
        for nid, attrs in data["graph_nodes"].items():
            hist._tree.data.add_node(nid, **attrs)
        for u, v in data["graph_edges"]:
            hist._tree.data.add_edge(u, v)
        return hist
