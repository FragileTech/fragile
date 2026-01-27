"""Fractal Set data structure for representing Fractal Gas execution traces.

This implementation stores an array-backed directed 2-complex with three edge types:
- CST edges: temporal evolution per recorded step
- IG edges: companion-selected spatial coupling per recorded step
- IA edges: influence attribution edges closing IG/CST triangles

Auxiliary clone edges are stored separately to represent cloning jitter at t+0.5.
Vector data is stored as raw psi_* tensors for later spinor construction; node
attributes remain scalar-only per the Volume 3 specification.

Reference: docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import torch
from torch import Tensor

from fragile.fractalai.core.companion_selection import compute_algorithmic_distance_matrix
from fragile.fractalai.core.distance import compute_periodic_distance_matrix
from fragile.fractalai.core.history import RunHistory


STAGE_PRE = 0
STAGE_CLONE = 1
STAGE_FINAL = 2
STAGE_NAMES = {STAGE_PRE: "pre", STAGE_CLONE: "clone", STAGE_FINAL: "final"}
STAGE_BY_NAME = {"pre": STAGE_PRE, "clone": STAGE_CLONE, "final": STAGE_FINAL}


class FractalSet:
    """Complete data structure for a Fractal Gas run.

    Nodes are stage-aware spacetime points (pre, clone, final). CST/IG/IA edges
    encode evolution, coupling, and attribution. Clone edges are auxiliary.
    """

    def __init__(
        self,
        history: RunHistory,
        epsilon_c: float | None = None,
        epsilon_d: float | None = None,
        lambda_alg: float | None = None,
        hbar_eff: float = 1.0,
    ):
        self.history = history
        self.N = history.N
        self.d = history.d
        self.n_steps = history.n_steps
        self.n_recorded = history.n_recorded
        self.record_every = history.record_every
        self.recorded_steps = history.recorded_steps
        self.delta_t = history.delta_t
        self.pbc = history.pbc
        self.bounds = history.bounds
        self.params = history.params or {}

        selection_params = self.params.get("companion_selection", {})
        self.epsilon_c = epsilon_c if epsilon_c is not None else selection_params.get("epsilon")
        self.epsilon_d = epsilon_d if epsilon_d is not None else self.epsilon_c
        self.lambda_alg = (
            lambda_alg if lambda_alg is not None else selection_params.get("lambda_alg", 0.0)
        )
        self.hbar_eff = hbar_eff

        kinetic_params = self.params.get("kinetic", {})
        self.nu = kinetic_params.get("nu", 0.0)
        self.viscous_length_scale = kinetic_params.get("viscous_length_scale", 1.0)
        self.use_viscous_coupling = kinetic_params.get("use_viscous_coupling", False)
        self.viscous_neighbor_mode = kinetic_params.get("viscous_neighbor_mode", "all")
        self.viscous_neighbor_threshold = kinetic_params.get("viscous_neighbor_threshold", None)
        self.viscous_neighbor_penalty = kinetic_params.get("viscous_neighbor_penalty", 0.0)
        self.viscous_degree_cap = kinetic_params.get("viscous_degree_cap", None)

        self._graph_cache: nx.DiGraph | None = None
        self._graph_cache_aux: nx.DiGraph | None = None

        self._build_nodes()
        self._build_edges()
        self._build_triangles()

    # =====================================================================
    # Node Construction
    # =====================================================================

    def _compute_pre_step(self, time_idx: int) -> float:
        if time_idx == 0:
            return 0.0
        return float(self.recorded_steps[time_idx] - 1)

    def _compute_delta(self, x_from: Tensor, x_to: Tensor) -> Tensor:
        delta = x_to - x_from
        if not self.pbc or self.bounds is None:
            return delta
        high = self.bounds.high.to(x_to)
        low = self.bounds.low.to(x_to)
        span = high - low
        return delta - span * torch.round(delta / span)

    def _build_nodes(self) -> None:
        device = self.history.x_final.device
        dtype = self.history.x_final.dtype
        n_pre = self.n_recorded
        n_clone = max(self.n_recorded - 1, 0)
        n_final = self.n_recorded
        total_nodes = self.N * (n_pre + n_clone + n_final)

        node_id = torch.arange(total_nodes, device=device, dtype=torch.long)
        node_stage = torch.empty(total_nodes, device=device, dtype=torch.int8)
        node_time_index = torch.empty(total_nodes, device=device, dtype=torch.long)
        node_walker = torch.empty(total_nodes, device=device, dtype=torch.long)
        node_abs_step = torch.empty(total_nodes, device=device, dtype=dtype)
        node_tau = torch.empty(total_nodes, device=device, dtype=dtype)
        node_alive = torch.empty(total_nodes, device=device, dtype=torch.bool)
        node_clone_source = torch.full((total_nodes,), -1, device=device, dtype=torch.long)

        node_E_kin = torch.empty(total_nodes, device=device, dtype=dtype)
        node_U = torch.empty(total_nodes, device=device, dtype=dtype)
        node_E_total = torch.empty(total_nodes, device=device, dtype=dtype)
        node_fitness = torch.full((total_nodes,), float("nan"), device=device, dtype=dtype)
        node_V_fit = torch.full((total_nodes,), float("nan"), device=device, dtype=dtype)
        node_reward = torch.full((total_nodes,), float("nan"), device=device, dtype=dtype)
        node_cloning_score = torch.full((total_nodes,), float("nan"), device=device, dtype=dtype)
        node_cloning_prob = torch.full((total_nodes,), float("nan"), device=device, dtype=dtype)
        node_will_clone = torch.zeros(total_nodes, device=device, dtype=torch.bool)
        node_d_alg = torch.full((total_nodes,), float("nan"), device=device, dtype=dtype)
        node_z_rewards = torch.full((total_nodes,), float("nan"), device=device, dtype=dtype)
        node_z_distances = torch.full((total_nodes,), float("nan"), device=device, dtype=dtype)
        node_rescaled_rewards = torch.full(
            (total_nodes,), float("nan"), device=device, dtype=dtype
        )
        node_rescaled_distances = torch.full(
            (total_nodes,), float("nan"), device=device, dtype=dtype
        )
        node_pos_sq = torch.full((total_nodes,), float("nan"), device=device, dtype=dtype)
        node_vel_sq = torch.full((total_nodes,), float("nan"), device=device, dtype=dtype)
        node_mu_rewards = torch.full((total_nodes,), float("nan"), device=device, dtype=dtype)
        node_sigma_rewards = torch.full((total_nodes,), float("nan"), device=device, dtype=dtype)
        node_mu_distances = torch.full((total_nodes,), float("nan"), device=device, dtype=dtype)
        node_sigma_distances = torch.full((total_nodes,), float("nan"), device=device, dtype=dtype)
        node_companion_distance = torch.full((total_nodes,), -1, device=device, dtype=torch.long)
        node_companion_clone = torch.full((total_nodes,), -1, device=device, dtype=torch.long)

        pre_offset = 0
        clone_offset = pre_offset + n_pre * self.N
        final_offset = clone_offset + n_clone * self.N

        pre_ids = (
            torch.arange(n_pre * self.N, device=device, dtype=torch.long).reshape(n_pre, self.N)
            + pre_offset
        )
        final_ids = (
            torch.arange(n_final * self.N, device=device, dtype=torch.long).reshape(
                n_final, self.N
            )
            + final_offset
        )
        if n_clone > 0:
            clone_ids = (
                torch.arange(n_clone * self.N, device=device, dtype=torch.long).reshape(
                    n_clone, self.N
                )
                + clone_offset
            )
        else:
            clone_ids = torch.empty((0, self.N), device=device, dtype=torch.long)

        clone_ids_full = torch.full((n_pre, self.N), -1, device=device, dtype=torch.long)
        if n_clone > 0:
            clone_ids_full[1:] = clone_ids

        self.node_index = {"pre": pre_ids, "clone": clone_ids_full, "final": final_ids}

        # Pre nodes
        for t_idx in range(n_pre):
            abs_step = self._compute_pre_step(t_idx)
            row_ids = pre_ids[t_idx]
            node_stage[row_ids] = STAGE_PRE
            node_time_index[row_ids] = t_idx
            node_walker[row_ids] = torch.arange(self.N, device=device)
            node_abs_step[row_ids] = abs_step
            node_tau[row_ids] = abs_step * self.delta_t

            x_pre = self.history.x_before_clone[t_idx]
            v_pre = self.history.v_before_clone[t_idx]
            node_E_kin[row_ids] = 0.5 * torch.sum(v_pre**2, dim=-1)
            node_U[row_ids] = self.history.U_before[t_idx]
            node_E_total[row_ids] = node_E_kin[row_ids] + node_U[row_ids]

            if t_idx == 0:
                if self.bounds is not None:
                    node_alive[row_ids] = self.bounds.contains(x_pre)
                else:
                    node_alive[row_ids] = torch.ones(self.N, device=device, dtype=torch.bool)
            else:
                node_alive[row_ids] = self.history.alive_mask[t_idx - 1]

            if t_idx > 0:
                info_idx = t_idx - 1
                node_fitness[row_ids] = self.history.fitness[info_idx]
                node_V_fit[row_ids] = self.history.fitness[info_idx]
                node_reward[row_ids] = self.history.rewards[info_idx]
                node_cloning_score[row_ids] = self.history.cloning_scores[info_idx]
                node_cloning_prob[row_ids] = self.history.cloning_probs[info_idx]
                node_will_clone[row_ids] = self.history.will_clone[info_idx]
                node_d_alg[row_ids] = self.history.distances[info_idx]
                node_z_rewards[row_ids] = self.history.z_rewards[info_idx]
                node_z_distances[row_ids] = self.history.z_distances[info_idx]
                node_rescaled_rewards[row_ids] = self.history.rescaled_rewards[info_idx]
                node_rescaled_distances[row_ids] = self.history.rescaled_distances[info_idx]
                node_pos_sq[row_ids] = self.history.pos_squared_differences[info_idx]
                node_vel_sq[row_ids] = self.history.vel_squared_differences[info_idx]
                node_mu_rewards[row_ids] = self.history.mu_rewards[info_idx]
                node_sigma_rewards[row_ids] = self.history.sigma_rewards[info_idx]
                node_mu_distances[row_ids] = self.history.mu_distances[info_idx]
                node_sigma_distances[row_ids] = self.history.sigma_distances[info_idx]
                node_companion_distance[row_ids] = self.history.companions_distance[info_idx]
                node_companion_clone[row_ids] = self.history.companions_clone[info_idx]

        # Clone nodes (auxiliary)
        for t_idx in range(1, n_pre):
            clone_idx = t_idx - 1
            row_ids = clone_ids_full[t_idx]
            abs_step = self._compute_pre_step(t_idx) + 0.5
            node_stage[row_ids] = STAGE_CLONE
            node_time_index[row_ids] = t_idx
            node_walker[row_ids] = torch.arange(self.N, device=device)
            node_abs_step[row_ids] = abs_step
            node_tau[row_ids] = abs_step * self.delta_t

            self.history.x_after_clone[clone_idx]
            v_clone = self.history.v_after_clone[clone_idx]
            node_E_kin[row_ids] = 0.5 * torch.sum(v_clone**2, dim=-1)
            node_U[row_ids] = self.history.U_after_clone[clone_idx]
            node_E_total[row_ids] = node_E_kin[row_ids] + node_U[row_ids]
            node_alive[row_ids] = torch.ones(self.N, device=device, dtype=torch.bool)

            clone_sources = self.history.companions_clone[clone_idx]
            cloned_mask = self.history.will_clone[clone_idx]
            if cloned_mask.any():
                node_clone_source[row_ids[cloned_mask]] = clone_sources[cloned_mask]

        # Final nodes
        for t_idx in range(n_final):
            row_ids = final_ids[t_idx]
            abs_step = float(self.recorded_steps[t_idx])
            node_stage[row_ids] = STAGE_FINAL
            node_time_index[row_ids] = t_idx
            node_walker[row_ids] = torch.arange(self.N, device=device)
            node_abs_step[row_ids] = abs_step
            node_tau[row_ids] = abs_step * self.delta_t

            x_final = self.history.x_final[t_idx]
            v_final = self.history.v_final[t_idx]
            node_E_kin[row_ids] = 0.5 * torch.sum(v_final**2, dim=-1)
            node_U[row_ids] = self.history.U_final[t_idx]
            node_E_total[row_ids] = node_E_kin[row_ids] + node_U[row_ids]

            if self.bounds is not None:
                node_alive[row_ids] = self.bounds.contains(x_final)
            else:
                node_alive[row_ids] = torch.ones(self.N, device=device, dtype=torch.bool)

            if t_idx > 0:
                info_idx = t_idx - 1
                clone_sources = self.history.companions_clone[info_idx]
                cloned_mask = self.history.will_clone[info_idx]
                if cloned_mask.any():
                    node_clone_source[row_ids[cloned_mask]] = clone_sources[cloned_mask]

        self.nodes = {
            "id": node_id,
            "stage": node_stage,
            "time_index": node_time_index,
            "walker": node_walker,
            "abs_step": node_abs_step,
            "tau": node_tau,
            "alive": node_alive,
            "clone_source": node_clone_source,
            "E_kin": node_E_kin,
            "U": node_U,
            "E_total": node_E_total,
            "fitness": node_fitness,
            "V_fit": node_V_fit,
            "reward": node_reward,
            "cloning_score": node_cloning_score,
            "cloning_prob": node_cloning_prob,
            "will_clone": node_will_clone,
            "algorithmic_distance": node_d_alg,
            "z_rewards": node_z_rewards,
            "z_distances": node_z_distances,
            "rescaled_rewards": node_rescaled_rewards,
            "rescaled_distances": node_rescaled_distances,
            "pos_sq_diff": node_pos_sq,
            "vel_sq_diff": node_vel_sq,
            "mu_rewards": node_mu_rewards,
            "sigma_rewards": node_sigma_rewards,
            "mu_distances": node_mu_distances,
            "sigma_distances": node_sigma_distances,
            "companion_distance_id": node_companion_distance,
            "companion_clone_id": node_companion_clone,
        }

    # =====================================================================
    # Edge Construction
    # =====================================================================

    def _compute_companion_weights(self, x: Tensor, v: Tensor, alive: Tensor) -> dict[str, Tensor]:
        if self.epsilon_d is None:
            eps = 1.0
        else:
            eps = float(self.epsilon_d)

        d_alg_sq = compute_algorithmic_distance_matrix(
            x,
            v,
            lambda_alg=float(self.lambda_alg or 0.0),
            bounds=self.bounds,
            pbc=self.pbc,
        )
        kernel = torch.exp(-d_alg_sq / (2.0 * eps**2))
        kernel.fill_diagonal_(0.0)
        alive_2d = alive.unsqueeze(0) & alive.unsqueeze(1)
        kernel = kernel * alive_2d.float()
        row_sums = torch.clamp(kernel.sum(dim=1, keepdim=True), min=1e-12)
        weights = kernel / row_sums

        if self.pbc and self.bounds is not None:
            dist_matrix = compute_periodic_distance_matrix(x, y=None, bounds=self.bounds, pbc=True)
        else:
            diff = x[:, None, :] - x[None, :, :]
            dist_matrix = torch.linalg.vector_norm(diff, dim=-1)

        return {
            "d_alg_sq": d_alg_sq,
            "kernel": kernel,
            "weights": weights,
            "distance": dist_matrix,
        }

    def _compute_viscous_weights(self, x: Tensor, alive: Tensor) -> dict[str, Tensor]:
        if not self.use_viscous_coupling or self.nu == 0.0:
            zero = torch.zeros(
                (alive.shape[0], alive.shape[0]),
                device=alive.device,
                dtype=x.dtype,
            )
            return {
                "kernel": zero,
                "weights": zero,
            }

        if self.pbc and self.bounds is not None:
            dist = compute_periodic_distance_matrix(x, y=None, bounds=self.bounds, pbc=True)
        else:
            diff = x[:, None, :] - x[None, :, :]
            dist = torch.linalg.vector_norm(diff, dim=-1)

        l_sq = float(self.viscous_length_scale) ** 2
        kernel = torch.exp(-(dist**2) / (2.0 * l_sq))
        kernel.fill_diagonal_(0.0)
        alive_2d = alive.unsqueeze(0) & alive.unsqueeze(1)
        kernel = kernel * alive_2d.float()
        if self.viscous_neighbor_mode == "nearest":
            nearest_dist = dist.clone()
            nearest_dist = nearest_dist.masked_fill(~alive_2d, float("inf"))
            nearest_dist.fill_diagonal_(float("inf"))
            nn_idx = nearest_dist.argmin(dim=1)
            mask = torch.zeros_like(kernel)
            mask.scatter_(1, nn_idx.unsqueeze(1), 1.0)
            kernel = kernel * mask
        deg = torch.clamp(kernel.sum(dim=1, keepdim=True), min=1e-12)
        weights = kernel / deg
        if self.viscous_neighbor_threshold is not None and self.viscous_neighbor_penalty > 0:
            threshold = float(self.viscous_neighbor_threshold)
            if threshold > 0:
                strong = kernel >= threshold
                strong_count = strong.sum(dim=1, keepdim=True).to(weights.dtype)
                excess = torch.clamp(strong_count - 1.0, min=0.0)
                penalty_scale = 1.0 / (1.0 + self.viscous_neighbor_penalty * excess)
                weights = weights * penalty_scale
        if self.viscous_degree_cap is not None:
            cap = float(self.viscous_degree_cap)
            if cap <= 0:
                weights = torch.zeros_like(weights)
            else:
                scale = torch.clamp(cap / deg, max=1.0)
                weights = weights * scale
        return {"kernel": kernel, "weights": weights}

    def _build_edges(self) -> None:
        self.edges = {}
        self._build_cst_edges()
        self._build_ig_edges()
        self._build_ia_edges()
        self._build_clone_edges()

    def _build_cst_edges(self) -> None:
        cst = {
            k: []
            for k in [
                "source",
                "target",
                "walker",
                "time_index",
                "abs_step",
                "delta_t",
                "v_pre",
                "v_final",
                "delta_v",
                "delta_x",
                "x_clone",
                "v_clone",
                "force_stable",
                "force_adapt",
                "force_viscous",
                "force_friction",
                "force_total",
                "noise",
                "sigma_reg_diag",
                "sigma_reg_full",
                "phi_cst",
                "norm_delta_v",
                "norm_delta_x",
            ]
        }

        for t_idx in range(1, self.n_recorded):
            info_idx = t_idx - 1
            pre_ids = self.node_index["pre"][t_idx]
            final_ids = self.node_index["final"][t_idx]
            alive = self.nodes["alive"][pre_ids]
            if not alive.any():
                continue

            x_pre = self.history.x_before_clone[t_idx]
            v_pre = self.history.v_before_clone[t_idx]
            x_final = self.history.x_final[t_idx]
            v_final = self.history.v_final[t_idx]
            x_clone = self.history.x_after_clone[info_idx]
            v_clone = self.history.v_after_clone[info_idx]

            delta_x = self._compute_delta(x_pre, x_final)
            delta_v = v_final - v_pre

            for walker_id in torch.where(alive)[0].tolist():
                cst["source"].append(int(pre_ids[walker_id]))
                cst["target"].append(int(final_ids[walker_id]))
                cst["walker"].append(walker_id)
                cst["time_index"].append(t_idx)
                cst["abs_step"].append(self._compute_pre_step(t_idx))
                cst["delta_t"].append(self.delta_t)
                cst["v_pre"].append(v_pre[walker_id].clone())
                cst["v_final"].append(v_final[walker_id].clone())
                cst["delta_v"].append(delta_v[walker_id].clone())
                cst["delta_x"].append(delta_x[walker_id].clone())
                cst["x_clone"].append(x_clone[walker_id].clone())
                cst["v_clone"].append(v_clone[walker_id].clone())
                cst["force_stable"].append(self.history.force_stable[info_idx, walker_id])
                cst["force_adapt"].append(self.history.force_adapt[info_idx, walker_id])
                cst["force_viscous"].append(self.history.force_viscous[info_idx, walker_id])
                cst["force_friction"].append(self.history.force_friction[info_idx, walker_id])
                cst["force_total"].append(self.history.force_total[info_idx, walker_id])
                cst["noise"].append(self.history.noise[info_idx, walker_id])
                if self.history.sigma_reg_diag is not None:
                    cst["sigma_reg_diag"].append(self.history.sigma_reg_diag[info_idx, walker_id])
                    cst["sigma_reg_full"].append(None)
                elif self.history.sigma_reg_full is not None:
                    cst["sigma_reg_diag"].append(None)
                    cst["sigma_reg_full"].append(self.history.sigma_reg_full[info_idx, walker_id])
                else:
                    cst["sigma_reg_diag"].append(None)
                    cst["sigma_reg_full"].append(None)
                cst["phi_cst"].append(0.0)
                cst["norm_delta_v"].append(float(torch.linalg.vector_norm(delta_v[walker_id])))
                cst["norm_delta_x"].append(float(torch.linalg.vector_norm(delta_x[walker_id])))

        self.edges["cst"] = cst

    def _build_ig_edges(self) -> None:
        ig = {
            k: []
            for k in [
                "source",
                "target",
                "source_walker",
                "target_walker",
                "time_index",
                "abs_step",
                "x_i",
                "x_j",
                "v_i",
                "v_j",
                "delta_x",
                "delta_v",
                "viscous_force",
                "kernel_companion",
                "weight_companion",
                "kernel_viscous",
                "weight_viscous",
                "distance",
                "d_alg",
                "theta_ij",
                "fitness_i",
                "fitness_j",
                "V_clone",
                "psi_amp",
                "psi_real",
                "psi_imag",
                "companion_kind",
            ]
        }

        for t_idx in range(1, self.n_recorded):
            info_idx = t_idx - 1
            pre_ids = self.node_index["pre"][t_idx]
            alive = self.nodes["alive"][pre_ids]
            if not alive.any():
                continue

            x_pre = self.history.x_before_clone[t_idx]
            v_pre = self.history.v_before_clone[t_idx]
            fitness = self.history.fitness[info_idx]

            pairwise = self._compute_companion_weights(x_pre, v_pre, alive)
            d_alg_sq = pairwise["d_alg_sq"]
            kernel = pairwise["kernel"]
            weights = pairwise["weights"]
            distances = pairwise["distance"]

            viscous = self._compute_viscous_weights(x_pre, alive)
            kernel_visc = viscous["kernel"]
            weights_visc = viscous["weights"]

            comp_dist = self.history.companions_distance[info_idx]
            comp_clone = self.history.companions_clone[info_idx]

            for walker_id in torch.where(alive)[0].tolist():
                companions = []
                dist_comp = int(comp_dist[walker_id].item())
                clone_comp = int(comp_clone[walker_id].item())
                if dist_comp == clone_comp:
                    companions.append((dist_comp, "both"))
                else:
                    companions.append((dist_comp, "distance"))
                    companions.append((clone_comp, "clone"))

                for target_id, kind in companions:
                    if target_id == walker_id:
                        continue
                    if not bool(alive[target_id].item()):
                        continue

                    source_node = int(pre_ids[walker_id])
                    target_node = int(pre_ids[target_id])
                    delta_x = self._compute_delta(x_pre[walker_id], x_pre[target_id])
                    delta_v = v_pre[target_id] - v_pre[walker_id]

                    d_alg = math.sqrt(float(d_alg_sq[walker_id, target_id].item()))
                    dist = float(distances[walker_id, target_id].item())
                    k_comp = float(kernel[walker_id, target_id].item())
                    w_comp = float(weights[walker_id, target_id].item())
                    k_visc = float(kernel_visc[walker_id, target_id].item())
                    w_visc = float(weights_visc[walker_id, target_id].item())

                    fitness_i = float(fitness[walker_id].item())
                    fitness_j = float(fitness[target_id].item())
                    V_clone = fitness_j - fitness_i
                    theta_ij = -V_clone / self.hbar_eff
                    psi_amp = math.sqrt(w_comp) if w_comp > 0.0 else 0.0
                    psi_real = psi_amp * math.cos(theta_ij)
                    psi_imag = psi_amp * math.sin(theta_ij)

                    nu_effective = self.nu if self.use_viscous_coupling else 0.0
                    viscous_force = nu_effective * w_visc * delta_v

                    ig["source"].append(source_node)
                    ig["target"].append(target_node)
                    ig["source_walker"].append(walker_id)
                    ig["target_walker"].append(target_id)
                    ig["time_index"].append(t_idx)
                    ig["abs_step"].append(self._compute_pre_step(t_idx))
                    ig["x_i"].append(x_pre[walker_id].clone())
                    ig["x_j"].append(x_pre[target_id].clone())
                    ig["v_i"].append(v_pre[walker_id].clone())
                    ig["v_j"].append(v_pre[target_id].clone())
                    ig["delta_x"].append(delta_x.clone())
                    ig["delta_v"].append(delta_v.clone())
                    ig["viscous_force"].append(viscous_force.clone())
                    ig["kernel_companion"].append(k_comp)
                    ig["weight_companion"].append(w_comp)
                    ig["kernel_viscous"].append(k_visc)
                    ig["weight_viscous"].append(w_visc)
                    ig["distance"].append(dist)
                    ig["d_alg"].append(d_alg)
                    ig["theta_ij"].append(theta_ij)
                    ig["fitness_i"].append(fitness_i)
                    ig["fitness_j"].append(fitness_j)
                    ig["V_clone"].append(V_clone)
                    ig["psi_amp"].append(psi_amp)
                    ig["psi_real"].append(psi_real)
                    ig["psi_imag"].append(psi_imag)
                    ig["companion_kind"].append(kind)

        self.edges["ig"] = ig

    def _build_ia_edges(self) -> None:
        ia = {
            k: []
            for k in [
                "source",
                "target",
                "effect_walker",
                "cause_walker",
                "time_index",
                "abs_step",
                "kernel_viscous",
                "weight_viscous",
                "w_ia",
                "clone_indicator",
                "clone_source",
                "phi_ia",
            ]
        }

        for t_idx in range(1, self.n_recorded):
            info_idx = t_idx - 1
            pre_ids = self.node_index["pre"][t_idx]
            final_ids = self.node_index["final"][t_idx]
            alive = self.nodes["alive"][pre_ids]
            if not alive.any():
                continue

            x_pre = self.history.x_before_clone[t_idx]
            viscous = self._compute_viscous_weights(x_pre, alive)
            kernel_visc = viscous["kernel"]
            weights_visc = viscous["weights"]

            comp_dist = self.history.companions_distance[info_idx]
            comp_clone = self.history.companions_clone[info_idx]
            will_clone = self.history.will_clone[info_idx]

            for walker_id in torch.where(alive)[0].tolist():
                companions = []
                dist_comp = int(comp_dist[walker_id].item())
                clone_comp = int(comp_clone[walker_id].item())
                if dist_comp == clone_comp:
                    companions.append(dist_comp)
                else:
                    companions.extend([dist_comp, clone_comp])

                for cause_id in companions:
                    if cause_id == walker_id:
                        continue
                    if not bool(alive[cause_id].item()):
                        continue

                    w_visc = float(weights_visc[walker_id, cause_id].item())
                    k_visc = float(kernel_visc[walker_id, cause_id].item())
                    cloned = bool(will_clone[walker_id].item())
                    clone_source = int(comp_clone[walker_id].item()) if cloned else -1
                    clone_indicator = 1 if cloned and cause_id == clone_source else 0
                    w_ia = float(clone_indicator) if cloned else w_visc

                    ia["source"].append(int(final_ids[walker_id]))
                    ia["target"].append(int(pre_ids[cause_id]))
                    ia["effect_walker"].append(walker_id)
                    ia["cause_walker"].append(cause_id)
                    ia["time_index"].append(t_idx)
                    ia["abs_step"].append(self._compute_pre_step(t_idx))
                    ia["kernel_viscous"].append(k_visc)
                    ia["weight_viscous"].append(w_visc)
                    ia["w_ia"].append(w_ia)
                    ia["clone_indicator"].append(clone_indicator)
                    ia["clone_source"].append(clone_source)
                    ia["phi_ia"].append(0.0)

        self.edges["ia"] = ia

    def _build_clone_edges(self) -> None:
        clone = {
            k: []
            for k in [
                "source",
                "target",
                "walker",
                "time_index",
                "abs_step",
                "delta_x",
                "delta_v",
                "clone_jitter",
            ]
        }

        if self.n_recorded <= 1:
            self.edges["clone"] = clone
            return

        for t_idx in range(1, self.n_recorded):
            info_idx = t_idx - 1
            pre_ids = self.node_index["pre"][t_idx]
            clone_ids = self.node_index["clone"][t_idx]
            for walker_id in range(self.N):
                clone["source"].append(int(pre_ids[walker_id]))
                clone["target"].append(int(clone_ids[walker_id]))
                clone["walker"].append(walker_id)
                clone["time_index"].append(t_idx)
                clone["abs_step"].append(self._compute_pre_step(t_idx) + 0.5)
                clone["delta_x"].append(self.history.clone_delta_x[info_idx, walker_id])
                clone["delta_v"].append(self.history.clone_delta_v[info_idx, walker_id])
                clone["clone_jitter"].append(self.history.clone_jitter[info_idx, walker_id])

        self.edges["clone"] = clone

    # =====================================================================
    # Triangles
    # =====================================================================

    def _build_triangles(self) -> None:
        triangles = {
            k: []
            for k in [
                "time_index",
                "source_walker",
                "influencer_walker",
                "node_pre",
                "node_influencer",
                "node_final",
                "edge_cst",
                "edge_ig",
                "edge_ia",
            ]
        }

        ig_edges = self.edges["ig"]
        ia_edges = self.edges["ia"]
        cst_edges = self.edges["cst"]

        cst_lookup = {}
        for idx, (src, dst, t_idx, walker) in enumerate(
            zip(
                cst_edges["source"],
                cst_edges["target"],
                cst_edges["time_index"],
                cst_edges["walker"],
            )
        ):
            cst_lookup[int(walker), int(t_idx)] = idx

        ia_lookup = {}
        for idx, (src, dst, t_idx, effect, cause) in enumerate(
            zip(
                ia_edges["source"],
                ia_edges["target"],
                ia_edges["time_index"],
                ia_edges["effect_walker"],
                ia_edges["cause_walker"],
            )
        ):
            ia_lookup[int(effect), int(cause), int(t_idx)] = idx

        for idx, (src, dst, t_idx, source_walker, target_walker) in enumerate(
            zip(
                ig_edges["source"],
                ig_edges["target"],
                ig_edges["time_index"],
                ig_edges["source_walker"],
                ig_edges["target_walker"],
            )
        ):
            cst_idx = cst_lookup.get((int(source_walker), int(t_idx)))
            ia_idx = ia_lookup.get((int(source_walker), int(target_walker), int(t_idx)))
            if cst_idx is None or ia_idx is None:
                continue
            pre_node = int(src)
            influencer_node = int(dst)
            final_node = int(self.node_index["final"][t_idx][source_walker])

            triangles["time_index"].append(int(t_idx))
            triangles["source_walker"].append(int(source_walker))
            triangles["influencer_walker"].append(int(target_walker))
            triangles["node_pre"].append(pre_node)
            triangles["node_influencer"].append(influencer_node)
            triangles["node_final"].append(final_node)
            triangles["edge_cst"].append(cst_idx)
            triangles["edge_ig"].append(idx)
            triangles["edge_ia"].append(ia_idx)

        self.triangles = triangles

    # =====================================================================
    # Graph View
    # =====================================================================

    def _build_graph(self, include_aux: bool = False) -> nx.DiGraph:
        g = nx.DiGraph()
        node_count = int(self.nodes["id"].numel())
        for nid in range(node_count):
            g.add_node(
                nid,
                node_id=nid,
                stage=STAGE_NAMES[int(self.nodes["stage"][nid])],
                walker_id=int(self.nodes["walker"][nid].item()),
                time_index=int(self.nodes["time_index"][nid].item()),
                absolute_step=float(self.nodes["abs_step"][nid].item()),
                tau=float(self.nodes["tau"][nid].item()),
                alive=bool(self.nodes["alive"][nid].item()),
                clone_source=int(self.nodes["clone_source"][nid].item()),
                E_kin=float(self.nodes["E_kin"][nid].item()),
                U=float(self.nodes["U"][nid].item()),
                E_total=float(self.nodes["E_total"][nid].item()),
                fitness=float(self.nodes["fitness"][nid].item()),
                V_fit=float(self.nodes["V_fit"][nid].item()),
                reward=float(self.nodes["reward"][nid].item()),
                cloning_score=float(self.nodes["cloning_score"][nid].item()),
                cloning_prob=float(self.nodes["cloning_prob"][nid].item()),
                will_clone=bool(self.nodes["will_clone"][nid].item()),
                algorithmic_distance=float(self.nodes["algorithmic_distance"][nid].item()),
                z_rewards=float(self.nodes["z_rewards"][nid].item()),
                z_distances=float(self.nodes["z_distances"][nid].item()),
                rescaled_rewards=float(self.nodes["rescaled_rewards"][nid].item()),
                rescaled_distances=float(self.nodes["rescaled_distances"][nid].item()),
                pos_sq_diff=float(self.nodes["pos_sq_diff"][nid].item()),
                vel_sq_diff=float(self.nodes["vel_sq_diff"][nid].item()),
                mu_rewards=float(self.nodes["mu_rewards"][nid].item()),
                sigma_rewards=float(self.nodes["sigma_rewards"][nid].item()),
                mu_distances=float(self.nodes["mu_distances"][nid].item()),
                sigma_distances=float(self.nodes["sigma_distances"][nid].item()),
                companion_distance_id=int(self.nodes["companion_distance_id"][nid].item()),
                companion_clone_id=int(self.nodes["companion_clone_id"][nid].item()),
            )

        def add_edges(edge_type: str) -> None:
            edges = self.edges[edge_type]
            for edge_id, (src, dst) in enumerate(zip(edges["source"], edges["target"])):
                attrs = {"edge_type": edge_type, "edge_id": edge_id}
                for key, values in edges.items():
                    if key in {"source", "target"}:
                        continue
                    attrs[key] = values[edge_id]
                g.add_edge(int(src), int(dst), **attrs)

        add_edges("cst")
        add_edges("ig")
        add_edges("ia")
        if include_aux:
            add_edges("clone")

        g.graph["triangles"] = self.triangles
        g.graph["recorded_steps"] = self.recorded_steps
        g.graph["delta_t"] = self.delta_t
        g.graph["schema_version"] = "fractal-set-v3"
        return g

    @property
    def graph(self) -> nx.DiGraph:
        if self._graph_cache is None:
            self._graph_cache = self._build_graph(include_aux=False)
        return self._graph_cache

    def graph_with_aux(self) -> nx.DiGraph:
        if self._graph_cache_aux is None:
            self._graph_cache_aux = self._build_graph(include_aux=True)
        return self._graph_cache_aux

    # =====================================================================
    # Query Methods
    # =====================================================================

    def get_walker_trajectory(self, walker_id: int, stage: str = "final") -> dict[str, Tensor]:
        if stage == "pre":
            return {
                "x": self.history.x_before_clone[:, walker_id, :],
                "v": self.history.v_before_clone[:, walker_id, :],
            }
        if stage == "clone":
            return {
                "x": self.history.x_after_clone[:, walker_id, :],
                "v": self.history.v_after_clone[:, walker_id, :],
            }
        return self.history.get_walker_trajectory(walker_id, stage="final")

    def get_node_id(self, walker_id: int, timestep: int, stage: str = "final") -> int:
        stage_key = STAGE_BY_NAME[stage]
        if stage_key == STAGE_PRE:
            return int(self.node_index["pre"][timestep, walker_id])
        if stage_key == STAGE_CLONE:
            node_id = int(self.node_index["clone"][timestep, walker_id])
            if node_id < 0:
                msg = f"No clone node for timestep={timestep}"
                raise ValueError(msg)
            return node_id
        return int(self.node_index["final"][timestep, walker_id])

    def get_node_data(self, walker_id: int, timestep: int, stage: str = "final") -> dict[str, Any]:
        node_id = self.get_node_id(walker_id, timestep, stage=stage)
        return {key: self.nodes[key][node_id] for key in self.nodes}

    def get_alive_walkers(self, timestep: int, stage: str = "final") -> list[int]:
        if stage == "clone" and timestep == 0:
            return []
        node_ids = self.node_index[stage][timestep]
        alive = self.nodes["alive"][node_ids]
        return torch.where(alive)[0].tolist()

    def get_cst_subgraph(self) -> nx.DiGraph:
        edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("edge_type") == "cst"]
        return self.graph.edge_subgraph(edges).copy()

    def get_ig_subgraph(
        self, timestep: int | None = None, companion_type: str | None = None
    ) -> nx.DiGraph:
        edges = []
        for u, v, d in self.graph.edges(data=True):
            if d.get("edge_type") != "ig":
                continue
            if timestep is not None and int(d.get("time_index")) != timestep:
                continue
            kind = d.get("companion_kind")
            if companion_type is not None:
                if companion_type == "distance" and kind not in {"distance", "both"}:
                    continue
                if companion_type == "clone" and kind not in {"clone", "both"}:
                    continue
                if companion_type == "both" and kind != "both":
                    continue
            edges.append((u, v))
        return self.graph.edge_subgraph(edges).copy()

    def get_ia_subgraph(
        self, timestep: int | None = None, clone_only: bool | None = None
    ) -> nx.DiGraph:
        edges = []
        for u, v, d in self.graph.edges(data=True):
            if d.get("edge_type") != "ia":
                continue
            if timestep is not None and int(d.get("time_index")) != timestep:
                continue
            clone_indicator = int(d.get("clone_indicator", 0))
            if clone_only is True and clone_indicator == 0:
                continue
            if clone_only is False and clone_indicator != 0:
                continue
            edges.append((u, v))
        return self.graph.edge_subgraph(edges).copy()

    def get_triangles(self, timestep: int | None = None) -> list[dict[str, Any]]:
        triangles = []
        for idx in range(len(self.triangles["time_index"])):
            t_idx = int(self.triangles["time_index"][idx])
            if timestep is not None and t_idx != timestep:
                continue
            triangles.append({
                "triangle_id": idx,
                "time_index": t_idx,
                "source_walker": int(self.triangles["source_walker"][idx]),
                "influencer_walker": int(self.triangles["influencer_walker"][idx]),
                "nodes": {
                    "pre": int(self.triangles["node_pre"][idx]),
                    "influencer": int(self.triangles["node_influencer"][idx]),
                    "final": int(self.triangles["node_final"][idx]),
                },
                "edges": {
                    "cst": int(self.triangles["edge_cst"][idx]),
                    "ig": int(self.triangles["edge_ig"][idx]),
                    "ia": int(self.triangles["edge_ia"][idx]),
                },
            })
        return triangles

    def get_cloning_events(self) -> list[tuple[int, int, int]]:
        return self.history.get_clone_events()

    # =====================================================================
    # Properties
    # =====================================================================

    @property
    def num_cst_edges(self) -> int:
        return len(self.edges["cst"]["source"])

    @property
    def num_ig_edges(self) -> int:
        return len(self.edges["ig"]["source"])

    @property
    def num_ia_edges(self) -> int:
        return len(self.edges["ia"]["source"])

    @property
    def num_clone_edges(self) -> int:
        return len(self.edges["clone"]["source"])

    @property
    def total_nodes(self) -> int:
        return int(self.nodes["id"].numel())

    @property
    def num_triangles(self) -> int:
        return len(self.triangles["time_index"])

    # =====================================================================
    # Serialization
    # =====================================================================

    def save(self, path: str) -> None:
        data = {
            "nodes": self.nodes,
            "edges": self.edges,
            "triangles": self.triangles,
            "meta": {
                "recorded_steps": self.recorded_steps,
                "delta_t": self.delta_t,
                "epsilon_c": self.epsilon_c,
                "epsilon_d": self.epsilon_d,
                "lambda_alg": self.lambda_alg,
                "hbar_eff": self.hbar_eff,
                "schema_version": "fractal-set-v3",
            },
        }
        torch.save(data, path)

    @classmethod
    def load(cls, path: str, history: RunHistory) -> FractalSet:
        data = torch.load(path, weights_only=False)
        fs = cls.__new__(cls)
        fs.history = history
        fs.N = history.N
        fs.d = history.d
        fs.n_steps = history.n_steps
        fs.n_recorded = history.n_recorded
        fs.record_every = history.record_every
        fs.recorded_steps = history.recorded_steps
        fs.delta_t = history.delta_t
        fs.pbc = history.pbc
        fs.bounds = history.bounds
        fs.params = history.params or {}

        meta = data.get("meta", {})
        fs.epsilon_c = meta.get("epsilon_c")
        fs.epsilon_d = meta.get("epsilon_d")
        fs.lambda_alg = meta.get("lambda_alg")
        fs.hbar_eff = meta.get("hbar_eff", 1.0)
        fs.nu = fs.params.get("kinetic", {}).get("nu", 0.0)
        fs.viscous_length_scale = fs.params.get("kinetic", {}).get("viscous_length_scale", 1.0)

        fs.nodes = data["nodes"]
        fs.edges = data["edges"]
        fs.triangles = data["triangles"]
        fs.node_index = {
            "pre": fs._rebuild_node_index(STAGE_PRE),
            "clone": fs._rebuild_node_index(STAGE_CLONE),
            "final": fs._rebuild_node_index(STAGE_FINAL),
        }

        fs._graph_cache = None
        fs._graph_cache_aux = None
        return fs

    def _rebuild_node_index(self, stage: int) -> Tensor:
        mask = self.nodes["stage"] == stage
        node_ids = self.nodes["id"][mask]
        time_idx = self.nodes["time_index"][mask]
        walker = self.nodes["walker"][mask]
        max_time = int(time_idx.max().item()) + 1 if node_ids.numel() else 0
        index = torch.full((max_time, self.N), -1, dtype=torch.long, device=node_ids.device)
        index[time_idx, walker] = node_ids
        return index
