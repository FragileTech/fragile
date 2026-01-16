"""Fractal Set data structure for representing Fractal Gas execution traces.

This module implements the Fractal Set as a directed 2-complex with simplicial support,
providing a complete graph-based representation of algorithm dynamics with three edge types:

- **CST (Causal Spacetime Tree)**: Temporal edges connecting walker states across timesteps.
- **IG (Information Graph)**: Directed spatial edges representing selection coupling.
- **IA (Influence Attribution)**: Retrocausal edges attributing effects to causes.

Interaction triangles are stored as explicit 2-simplices that close CST/IG/IA loops.
The Fractal Set is constructed from a RunHistory object and stores the execution trace
as a NetworkX directed graph with rich node and edge attributes plus a triangle list.

Reference: docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md
"""

from __future__ import annotations

from typing import Any

import math
import numpy as np

import networkx as nx
import torch
from torch import Tensor

from fragile.fractalai.core.history import RunHistory


class FractalSet:
    """Complete graph representation of a Fractal Gas run with CST/IG/IA structure.

    The Fractal Set encodes the full execution trace of a Fractal Gas run as a directed
    2-complex where:

    - **Nodes** represent individual walkers at specific timesteps (spacetime points)
    - **CST edges** connect the same walker across consecutive timesteps (temporal evolution)
    - **IG edges** connect different walkers at the same timestep (selection coupling)
    - **IA edges** connect effects to causes across time (influence attribution)
    - **Interaction triangles** close each IG/CST/IA causal loop

    Nodes store scalar quantities like fitness, energy, and status. Edge attributes
    record directional quantities in a doc-aligned schema; vector data is stored with
    spinor-style keys for downstream reconstruction experiments.

    Attributes:
        history: The RunHistory object containing the execution trace data
        graph: NetworkX directed graph storing nodes and edges
        triangles: List of interaction triangle records (2-simplices)
        N: Number of walkers
        d: Spatial dimension
        n_steps: Total number of steps executed
        n_recorded: Number of recorded timesteps
        record_every: Recording interval

    Example:
        >>> history = gas.run(n_steps=100, record_every=10)
        >>> fractal_set = FractalSet(history)
        >>> print(f"Nodes: {fractal_set.graph.number_of_nodes()}")
        >>> print(f"CST edges: {fractal_set.num_cst_edges}")
        >>> print(f"IG edges: {fractal_set.num_ig_edges}")
        >>> print(f"IA edges: {fractal_set.num_ia_edges}")
        >>> triangle = fractal_set.triangles[0]

    Reference: docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md
    """

    def __init__(
        self,
        history: RunHistory,
        epsilon_c: float | None = None,
        hbar_eff: float = 1.0,
        epsilon_d: float | None = None,
        lambda_alg: float = 0.0,
    ):
        """Initialize FractalSet from a RunHistory object.

        Args:
            history: RunHistory object from EuclideanGas.run()
            epsilon_c: Cloning interaction range (kept for backward compatibility).
            hbar_eff: Effective Planck constant for fitness-phase encoding (default: 1.0)
            epsilon_d: Diversity interaction range for companion amplitude computation.
                If None, falls back to epsilon_c or defaults to 1.0.
            lambda_alg: Velocity weight in algorithmic distance (default: 0.0)

        The constructor immediately builds the complete graph structure by:
        1. Creating nodes for all (walker, timestep) pairs
        2. Adding CST edges for temporal evolution
        3. Adding IG edges for selection coupling with fitness phase potentials
        4. Adding IA edges for influence attribution
        5. Building interaction triangles that close IG/CST/IA loops
        """
        self.history = history
        self.graph = nx.DiGraph()

        # Store metadata for convenience
        self.N = history.N
        self.d = history.d
        self.n_steps = history.n_steps
        self.n_recorded = history.n_recorded
        self.record_every = history.record_every
        self.delta_t = float(self.record_every)
        self.recorded_steps = self._compute_recorded_steps()

        # Store parameters for IG edge phase/amplitude computation
        self.epsilon_c = epsilon_c
        self.epsilon_d = epsilon_d
        self.lambda_alg = lambda_alg
        self.hbar_eff = hbar_eff

        # Build graph structure
        self._build_nodes()
        self._build_cst_edges()
        self._build_ig_edges()
        self._build_ia_edges()
        self._build_triangles()

    # ========================================================================
    # Construction Methods
    # ========================================================================

    def _compute_recorded_steps(self) -> list[int]:
        recorded_steps = list(range(0, self.n_steps + 1, self.record_every))
        if self.n_steps not in recorded_steps:
            recorded_steps.append(self.n_steps)
        if len(recorded_steps) != self.n_recorded:
            recorded_steps = [t_idx * self.record_every for t_idx in range(self.n_recorded)]
            recorded_steps[-1] = self.n_steps
        self.graph.graph["recorded_steps"] = recorded_steps
        self.graph.graph["delta_t"] = self.delta_t
        return recorded_steps

    def _info_index(self, t_idx: int) -> int | None:
        if t_idx < self.n_recorded - 1:
            return t_idx
        return None

    def _alive_mask_at(self, t_idx: int) -> Tensor:
        if t_idx < self.n_recorded - 1:
            return self.history.alive_mask[t_idx]
        if self.history.bounds is not None:
            return self.history.bounds.contains(self.history.x_final[t_idx])
        return torch.ones(self.N, dtype=torch.bool, device=self.history.x_final.device)

    def _pairwise_weights(
        self, t_idx: int, alive_indices: list[int]
    ) -> dict[str, Tensor | dict[int, int]]:
        x_alive = self.history.x_final[t_idx, alive_indices, :]
        v_alive = self.history.v_final[t_idx, alive_indices, :]
        delta_x = x_alive[:, None, :] - x_alive[None, :, :]
        delta_v = v_alive[:, None, :] - v_alive[None, :, :]
        d_alg_sq = torch.sum(delta_x**2, dim=2) + self.lambda_alg * torch.sum(
            delta_v**2, dim=2
        )
        epsilon_d = self.epsilon_d if self.epsilon_d is not None else self.epsilon_c
        if epsilon_d is None:
            epsilon_d = 1.0
        weights = torch.exp(-d_alg_sq / (2.0 * epsilon_d**2))
        weights.fill_diagonal_(0.0)
        weight_sums = torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-12)
        p_comp = weights / weight_sums
        alive_index = {walker_id: idx for idx, walker_id in enumerate(alive_indices)}
        return {
            "x_alive": x_alive,
            "v_alive": v_alive,
            "delta_x": delta_x,
            "delta_v": delta_v,
            "d_alg_sq": d_alg_sq,
            "weights": weights,
            "p_comp": p_comp,
            "alive_index": alive_index,
        }

    def _build_nodes(self):
        """Construct all nodes in the Fractal Set.

        Creates one node for each (walker_id, timestep) pair, storing scalar attributes:
        - Identity: walker_id, timestep, node_id
        - Temporal: continuous time t
        - Status: alive flag
        - Energy: kinetic energy, potential U
        - Fitness: Φ, V_fit
        - Per-step info data (from history.alive_mask indices)

        Nodes are indexed by (walker_id, timestep) tuples where timestep is the
        recorded index (not the absolute step number).

        Reference: docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md § 2
        """
        for t_idx in range(self.n_recorded):
            step = self.recorded_steps[t_idx] if t_idx < len(self.recorded_steps) else 0
            info_idx = self._info_index(t_idx)
            alive_mask = self._alive_mask_at(t_idx)

            for walker_id in range(self.N):
                node_id = (walker_id, t_idx)

                # Extract position and velocity for energy computation
                x = self.history.x_final[t_idx, walker_id, :]
                v = self.history.v_final[t_idx, walker_id, :]

                # Compute kinetic energy
                E_kin = 0.5 * torch.sum(v**2).item()

                # Base node attributes (always available)
                attrs = {
                    "node_id": t_idx * self.N + walker_id,
                    "walker_id": walker_id,
                    "timestep": t_idx,
                    "absolute_step": step,
                    "t": float(step),
                    "delta_t": self.delta_t,
                    "x": x.clone(),
                    "v": v.clone(),
                    "E_kin": E_kin,
                    "alive": alive_mask[walker_id].item(),
                }

                if info_idx is not None:
                    will_clone = self.history.will_clone[info_idx, walker_id].item()
                    clone_source = (
                        self.history.companions_clone[info_idx, walker_id].item()
                        if will_clone
                        else None
                    )
                    attrs.update(
                        {
                            "fitness": self.history.fitness[info_idx, walker_id].item(),
                            "V_fit": self.history.fitness[info_idx, walker_id].item(),
                            "reward": self.history.rewards[info_idx, walker_id].item(),
                            "cloning_score": self.history.cloning_scores[
                                info_idx, walker_id
                            ].item(),
                            "cloning_prob": self.history.cloning_probs[
                                info_idx, walker_id
                            ].item(),
                            "will_clone": will_clone,
                            "clone_source": clone_source,
                            "companion_distance_id": self.history.companions_distance[
                                info_idx, walker_id
                            ].item(),
                            "companion_clone_id": self.history.companions_clone[
                                info_idx, walker_id
                            ].item(),
                            # Intermediate fitness computation scalars (from RunHistory)
                            "z_rewards": self.history.z_rewards[info_idx, walker_id].item(),
                            "z_distances": self.history.z_distances[info_idx, walker_id].item(),
                            "rescaled_rewards": self.history.rescaled_rewards[
                                info_idx, walker_id
                            ].item(),
                            "rescaled_distances": self.history.rescaled_distances[
                                info_idx, walker_id
                            ].item(),
                            "pos_squared_diff": self.history.pos_squared_differences[
                                info_idx, walker_id
                            ].item(),
                            "vel_squared_diff": self.history.vel_squared_differences[
                                info_idx, walker_id
                            ].item(),
                            "algorithmic_distance": self.history.distances[
                                info_idx, walker_id
                            ].item(),
                            # Localized statistics (per-step, global case rho → ∞)
                            "mu_rewards": self.history.mu_rewards[info_idx].item(),
                            "sigma_rewards": self.history.sigma_rewards[info_idx].item(),
                            "mu_distances": self.history.mu_distances[info_idx].item(),
                            "sigma_distances": self.history.sigma_distances[info_idx].item(),
                        }
                    )

                self.graph.add_node(node_id, **attrs)

    def _build_cst_edges(self):
        """Construct CST (Causal Spacetime Tree) edges.

        Creates directed temporal edges (i, t) → (i, t+1) for each walker's evolution
        across consecutive timesteps. Only creates edges for alive walkers.

        Each CST edge stores:
        - Velocity at source and target: v_t, v_{t+1}
        - Velocity increment: Δv = v_{t+1} - v_t
        - Position displacement: Δx = x_{t+1} - x_t
        - Spinor-aligned aliases for the vector quantities
        - Derived scalars: ||Δv||, ||Δx|| and Δt

        Reference: docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md § 3
        """
        edge_id = 0
        for t_idx in range(self.n_recorded - 1):
            alive_mask = self._alive_mask_at(t_idx)
            if t_idx + 1 < len(self.recorded_steps):
                delta_t = float(self.recorded_steps[t_idx + 1] - self.recorded_steps[t_idx])
            else:
                delta_t = self.delta_t
            for walker_id in range(self.N):
                # Check if walker is alive at this timestep
                if not alive_mask[walker_id].item():
                    continue

                source = (walker_id, t_idx)
                target = (walker_id, t_idx + 1)

                # Extract velocities and positions (final states after all operators)
                v_t = self.history.v_final[t_idx, walker_id, :]
                v_t1 = self.history.v_final[t_idx + 1, walker_id, :]
                x_t = self.history.x_final[t_idx, walker_id, :]
                x_t1 = self.history.x_final[t_idx + 1, walker_id, :]

                # Compute increments
                Delta_v = v_t1 - v_t
                Delta_x = x_t1 - x_t
                v_t_data = v_t.clone()
                v_t1_data = v_t1.clone()
                delta_v_data = Delta_v.clone()
                delta_x_data = Delta_x.clone()

                attrs = {
                    "edge_type": "cst",
                    "edge_id": edge_id,
                    "walker_id": walker_id,
                    "timestep": t_idx,
                    "delta_t": delta_t,
                    "omega_cst": delta_t,
                    # Final states (after all operators)
                    "v_t": v_t_data,
                    "v_t1": v_t1_data,
                    "Delta_v": delta_v_data,
                    "Delta_x": delta_x_data,
                    # Spinor-aligned aliases for vector data
                    "psi_v_t": v_t_data,
                    "psi_v_t1": v_t1_data,
                    "psi_Delta_v": delta_v_data,
                    "psi_Delta_x": delta_x_data,
                    "norm_Delta_v": torch.norm(Delta_v).item(),
                    "norm_Delta_x": torch.norm(Delta_x).item(),
                }

                # Add before/after cloning states (t_idx+1 has cloning data from step t_idx)
                # before_clone: state before cloning operator at next timestep
                # after_clone: state after cloning, before kinetic operator
                attrs["x_before_clone"] = self.history.x_before_clone[
                    t_idx + 1, walker_id, :
                ].clone()
                attrs["v_before_clone"] = self.history.v_before_clone[
                    t_idx + 1, walker_id, :
                ].clone()

                # After cloning states (available for t_idx > 0, since t_idx+1 > 0)
                if t_idx < self.n_recorded - 1:  # Ensure we don't go out of bounds
                    attrs["x_after_clone"] = self.history.x_after_clone[
                        t_idx, walker_id, :
                    ].clone()
                    attrs["v_after_clone"] = self.history.v_after_clone[
                        t_idx, walker_id, :
                    ].clone()

                # Add gradient/Hessian data if available (from adaptive kinetics)
                if self.history.fitness_gradients is not None:
                    grad_V_fit = self.history.fitness_gradients[t_idx, walker_id, :]
                    attrs["grad_V_fit"] = grad_V_fit.clone()
                    attrs["psi_nabla_V_fit"] = grad_V_fit.clone()
                    attrs["norm_grad_V_fit"] = torch.norm(grad_V_fit).item()

                if self.history.fitness_hessians_diag is not None:
                    attrs["hess_V_fit_diag"] = self.history.fitness_hessians_diag[
                        t_idx, walker_id, :
                    ].clone()
                elif self.history.fitness_hessians_full is not None:
                    attrs["hess_V_fit_full"] = self.history.fitness_hessians_full[
                        t_idx, walker_id, :, :
                    ].clone()

                self.graph.add_edge(source, target, **attrs)
                edge_id += 1

    def _build_ig_edges(self):
        """Construct IG (Information Graph) edges.

        Creates directed spatial edges (i, t) → (j, t) representing selection coupling
        between different walkers at the same timestep. These edges are DIRECTED and
        encode the antisymmetric cloning potential.

        For each pair of alive walkers (i, j) at timestep t, creates directed edge
        i → j storing:
        - Relative position: Δx_ij = x_j - x_i
        - Relative velocity: Δv_ij = v_j - v_i
        - Antisymmetric cloning potential: V_clone(i→j) = Φ_j - Φ_i
        - Distance: ||x_i - x_j||
        - Fitness phase: θ_ij = -(Φ_j - Φ_i)/ħ_eff
        - Companion amplitude: √P_comp(i,j) from algorithmic distance
        - Companion type flags: is_distance_companion, is_clone_companion

        Note: Creates a complete directed graph (tournament) among alive walkers.
        For k alive walkers, creates k(k-1) directed IG edges.

        Reference: docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md § 4
        """
        edge_id = 0
        for t_idx in range(self.n_recorded):
            alive_mask = self._alive_mask_at(t_idx)
            alive_indices = torch.where(alive_mask)[0].tolist()
            if len(alive_indices) < 2:
                continue

            pairwise = self._pairwise_weights(t_idx, alive_indices)
            x_alive = pairwise["x_alive"]
            v_alive = pairwise["v_alive"]
            delta_x = pairwise["delta_x"]
            delta_v = pairwise["delta_v"]
            d_alg_sq = pairwise["d_alg_sq"]
            weights = pairwise["weights"]
            p_comp = pairwise["p_comp"]
            alive_index = pairwise["alive_index"]

            info_idx = self._info_index(t_idx)
            if info_idx is not None:
                companion_distance = self.history.companions_distance[info_idx]
                companion_clone = self.history.companions_clone[info_idx]
                fitness = self.history.fitness[info_idx]

            for i in alive_indices:
                i_idx = alive_index[i]
                for j in alive_indices:
                    if i == j:
                        continue

                    j_idx = alive_index[j]
                    source = (i, t_idx)
                    target = (j, t_idx)

                    x_i = x_alive[i_idx]
                    x_j = x_alive[j_idx]
                    v_i = v_alive[i_idx]
                    v_j = v_alive[j_idx]

                    Delta_x_ij = delta_x[i_idx, j_idx]
                    Delta_v_ij = delta_v[i_idx, j_idx]
                    distance = torch.norm(Delta_x_ij).item()

                    d_alg_ij = math.sqrt(d_alg_sq[i_idx, j_idx].item())
                    kernel_weight = weights[i_idx, j_idx].item()
                    w_ij = p_comp[i_idx, j_idx].item()

                    x_i_data = x_i.clone()
                    x_j_data = x_j.clone()
                    v_i_data = v_i.clone()
                    v_j_data = v_j.clone()
                    delta_x_data = Delta_x_ij.clone()
                    delta_v_data = Delta_v_ij.clone()

                    attrs = {
                        "edge_type": "ig",
                        "edge_id": edge_id,
                        "source_walker": i,
                        "target_walker": j,
                        "timestep": t_idx,
                        "x_i": x_i_data,
                        "x_j": x_j_data,
                        "v_i": v_i_data,
                        "v_j": v_j_data,
                        "Delta_x_ij": delta_x_data,
                        "Delta_v_ij": delta_v_data,
                        # Spinor-aligned aliases for vector data
                        "psi_x_i": x_i_data,
                        "psi_x_j": x_j_data,
                        "psi_Delta_x_ij": delta_x_data,
                        "psi_v_i": v_i_data,
                        "psi_v_j": v_j_data,
                        "psi_Delta_v_ij": delta_v_data,
                        "distance": distance,
                        "kernel_weight": kernel_weight,
                        "w_ij": w_ij,
                        "d_alg_ij": d_alg_ij,
                    }

                    if info_idx is not None:
                        fitness_i = fitness[i].item()
                        fitness_j = fitness[j].item()
                        V_clone = fitness_j - fitness_i
                        theta_ij = -V_clone / self.hbar_eff
                        psi_amp = math.sqrt(w_ij) if w_ij > 0.0 else 0.0
                        psi_ij = psi_amp * np.exp(1j * theta_ij)

                        attrs.update(
                            {
                                "V_clone": V_clone,
                                "fitness_i": fitness_i,
                                "fitness_j": fitness_j,
                                "Phi_i": fitness_i,
                                "Phi_j": fitness_j,
                                "is_distance_companion": j == companion_distance[i].item(),
                                "is_clone_companion": j == companion_clone[i].item(),
                                "d_alg_i": self.history.distances[info_idx, i].item(),
                                "d_alg_j": self.history.distances[info_idx, j].item(),
                                "theta_ij": theta_ij,
                                "psi_ij_real": float(psi_ij.real),
                                "psi_ij_imag": float(psi_ij.imag),
                                "psi_ij_amp": float(psi_amp),
                                "p_comp_ij": w_ij,
                                "z_rewards_i": self.history.z_rewards[info_idx, i].item(),
                                "z_rewards_j": self.history.z_rewards[info_idx, j].item(),
                                "z_distances_i": self.history.z_distances[info_idx, i].item(),
                                "z_distances_j": self.history.z_distances[info_idx, j].item(),
                                "rescaled_rewards_i": self.history.rescaled_rewards[
                                    info_idx, i
                                ].item(),
                                "rescaled_rewards_j": self.history.rescaled_rewards[
                                    info_idx, j
                                ].item(),
                                "rescaled_distances_i": self.history.rescaled_distances[
                                    info_idx, i
                                ].item(),
                                "rescaled_distances_j": self.history.rescaled_distances[
                                    info_idx, j
                                ].item(),
                                "pos_sq_diff_i": self.history.pos_squared_differences[
                                    info_idx, i
                                ].item(),
                                "pos_sq_diff_j": self.history.pos_squared_differences[
                                    info_idx, j
                                ].item(),
                                "vel_sq_diff_i": self.history.vel_squared_differences[
                                    info_idx, i
                                ].item(),
                                "vel_sq_diff_j": self.history.vel_squared_differences[
                                    info_idx, j
                                ].item(),
                            }
                        )

                    self.graph.add_edge(source, target, **attrs)
                    edge_id += 1

    def _build_ia_edges(self):
        """Construct IA (Influence Attribution) edges.

        Creates directed retrocausal edges (i, t+1) → (j, t) for each ordered
        pair of distinct alive walkers at timestep t. These edges attribute
        the update of walker i to walker j and close IG/CST/IA triangles.

        Reference: docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md § 5.3
        """
        edge_id = 0
        for t_idx in range(self.n_recorded - 1):
            alive_mask = self._alive_mask_at(t_idx)
            alive_indices = torch.where(alive_mask)[0].tolist()
            if len(alive_indices) < 2:
                continue

            pairwise = self._pairwise_weights(t_idx, alive_indices)
            weights = pairwise["weights"]
            p_comp = pairwise["p_comp"]
            alive_index = pairwise["alive_index"]

            will_clone = self.history.will_clone[t_idx]
            companions_clone = self.history.companions_clone[t_idx]

            for i in alive_indices:
                i_idx = alive_index[i]
                cloned = bool(will_clone[i].item())
                clone_source = companions_clone[i].item() if cloned else None

                for j in alive_indices:
                    if i == j:
                        continue

                    j_idx = alive_index[j]
                    source = (i, t_idx + 1)
                    target = (j, t_idx)
                    w_viscous = float(p_comp[i_idx, j_idx].item())
                    clone_indicator = 1 if cloned and j == clone_source else 0
                    w_ia = float(clone_indicator) if cloned else w_viscous

                    attrs = {
                        "edge_type": "ia",
                        "edge_id": edge_id,
                        "effect_walker": i,
                        "cause_walker": j,
                        "timestep": t_idx,
                        "kernel_weight": float(weights[i_idx, j_idx].item()),
                        "w_ia": w_ia,
                        "w_ia_viscous": w_viscous,
                        "clone_indicator": clone_indicator,
                        "clone_source": clone_source,
                        "phi_ia": 0.0,
                        "omega_ia": w_ia,
                    }

                    self.graph.add_edge(source, target, **attrs)
                    edge_id += 1

    def _build_triangles(self):
        """Construct interaction triangles from IG/CST/IA edges.

        Each triangle stores its vertices and boundary edges, matching the
        definition of the interaction 2-simplices in the Fractal Set.
        """
        triangles: list[dict[str, Any]] = []
        triangle_id = 0
        for t_idx in range(self.n_recorded - 1):
            alive_mask = self._alive_mask_at(t_idx)
            alive_indices = torch.where(alive_mask)[0].tolist()
            if len(alive_indices) < 2:
                continue

            for i in alive_indices:
                for j in alive_indices:
                    if i == j:
                        continue

                    ig_edge = ((i, t_idx), (j, t_idx))
                    cst_edge = ((i, t_idx), (i, t_idx + 1))
                    ia_edge = ((i, t_idx + 1), (j, t_idx))

                    if not (
                        self.graph.has_edge(*ig_edge)
                        and self.graph.has_edge(*cst_edge)
                        and self.graph.has_edge(*ia_edge)
                    ):
                        continue

                    triangles.append(
                        {
                            "triangle_id": triangle_id,
                            "timestep": t_idx,
                            "source_walker": i,
                            "influencer_walker": j,
                            "nodes": {
                                "influencer": (j, t_idx),
                                "influenced": (i, t_idx),
                                "effect": (i, t_idx + 1),
                            },
                            "edges": {"ig": ig_edge, "cst": cst_edge, "ia": ia_edge},
                        }
                    )
                    triangle_id += 1

        self.triangles = triangles
        self.graph.graph["triangles"] = triangles

    # ========================================================================
    # Query Methods
    # ========================================================================

    def get_walker_trajectory(
        self,
        walker_id: int,
        stage: str = "final",
    ) -> dict[str, Tensor]:
        """Extract trajectory for a single walker from node positions.

        Args:
            walker_id: Walker index (0 to N-1)
            stage: Which state to extract - delegates to RunHistory

        Returns:
            Dict with 'x' [n_recorded, d] and 'v' [n_recorded, d] tensors

        Note: This delegates to RunHistory.get_walker_trajectory() for consistency.
        """
        return self.history.get_walker_trajectory(walker_id, stage=stage)

    def get_cst_subgraph(self) -> nx.DiGraph:
        """Extract CST (temporal evolution) subgraph.

        Returns:
            Directed graph containing only CST edges (temporal transitions)
        """
        edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d["edge_type"] == "cst"]
        return self.graph.edge_subgraph(edges).copy()

    def get_ig_subgraph(
        self,
        timestep: int | None = None,
        companion_type: str | None = None,
    ) -> nx.DiGraph:
        """Extract IG (selection coupling) subgraph.

        Args:
            timestep: If provided, only extract IG edges at this recorded timestep.
                     If None, extract all IG edges.
            companion_type: Filter by companion type:
                - "distance": Only edges where j is i's distance companion
                - "clone": Only edges where j is i's clone companion
                - "both": Only edges where j is both distance and clone companion
                - None: All IG edges (default)

        Returns:
            Directed graph containing only IG edges matching the criteria
        """

        def edge_filter(u, v, d):
            # Must be IG edge
            if d["edge_type"] != "ig":
                return False

            # Timestep filter
            if timestep is not None and d["timestep"] != timestep:
                return False

            # Companion type filter
            if companion_type == "distance":
                return d.get("is_distance_companion", False)
            if companion_type == "clone":
                return d.get("is_clone_companion", False)
            if companion_type == "both":
                return d.get("is_distance_companion", False) and d.get("is_clone_companion", False)

            # No companion filter (all IG edges)
            return True

        edges = [(u, v) for u, v, d in self.graph.edges(data=True) if edge_filter(u, v, d)]
        return self.graph.edge_subgraph(edges).copy()

    def get_ia_subgraph(
        self,
        timestep: int | None = None,
        clone_only: bool | None = None,
    ) -> nx.DiGraph:
        """Extract IA (influence attribution) subgraph.

        Args:
            timestep: If provided, only extract IA edges at this recorded timestep.
                     If None, extract all IA edges.
            clone_only: If True, only include cloning attribution edges. If False,
                        exclude cloning attribution edges. If None, include all.

        Returns:
            Directed graph containing only IA edges matching the criteria
        """

        def edge_filter(u, v, d):
            if d["edge_type"] != "ia":
                return False
            if timestep is not None and d["timestep"] != timestep:
                return False
            if clone_only is True and not d.get("clone_indicator", 0):
                return False
            if clone_only is False and d.get("clone_indicator", 0):
                return False
            return True

        edges = [(u, v) for u, v, d in self.graph.edges(data=True) if edge_filter(u, v, d)]
        return self.graph.edge_subgraph(edges).copy()

    def get_triangles(self, timestep: int | None = None) -> list[dict[str, Any]]:
        """Return interaction triangles, optionally filtered by timestep."""
        triangles = getattr(self, "triangles", [])
        if timestep is None:
            return list(triangles)
        return [triangle for triangle in triangles if triangle.get("timestep") == timestep]

    def get_cloning_events(self) -> list[tuple[int, int, int]]:
        """Get list of all cloning events.

        Returns:
            List of (step, cloner_idx, companion_idx) tuples

        Note: Delegates to RunHistory.get_clone_events() for consistency.
        """
        return self.history.get_clone_events()

    def get_node_data(self, walker_id: int, timestep: int) -> dict[str, Any]:
        """Get all attributes for a specific node.

        Args:
            walker_id: Walker index
            timestep: Recorded timestep index

        Returns:
            Dictionary of node attributes
        """
        node_id = (walker_id, timestep)
        return dict(self.graph.nodes[node_id])

    def get_alive_walkers(self, timestep: int) -> list[int]:
        """Get list of alive walker IDs at a given timestep.

        Args:
            timestep: Recorded timestep index

        Returns:
            List of alive walker indices
        """
        alive = []
        for walker_id in range(self.N):
            node_data = self.get_node_data(walker_id, timestep)
            if node_data.get("alive", False):
                alive.append(walker_id)
        return alive

    # ========================================================================
    # Phase 4: Analysis Query Methods
    # ========================================================================

    def get_energy_statistics(self, timestep: int) -> dict[str, float]:
        """Compute energy statistics for alive walkers at a given timestep.

        Args:
            timestep: Recorded timestep index

        Returns:
            Dictionary with energy statistics:
            - mean_potential: Mean potential energy U
            - std_potential: Std of potential energy
            - mean_kinetic: Mean kinetic energy (1/2 ||v||^2)
            - std_kinetic: Std of kinetic energy
            - mean_total: Mean total energy (U + KE)
            - std_total: Std of total energy
            - min_potential: Minimum potential energy
            - max_potential: Maximum potential energy

        Reference: docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md
        """
        alive_walkers = self.get_alive_walkers(timestep)

        potentials = []
        kinetics = []

        for walker_id in alive_walkers:
            node_data = self.get_node_data(walker_id, timestep)
            U = node_data.get("U", 0.0)
            if U is None:
                U = 0.0
            E_kin = node_data.get("E_kin", 0.0)

            potentials.append(U)
            kinetics.append(E_kin)

        import numpy as np

        potentials = np.array(potentials)
        kinetics = np.array(kinetics) if kinetics else np.array([0.0])
        totals = potentials + kinetics

        return {
            "mean_potential": float(np.mean(potentials)),
            "std_potential": float(np.std(potentials)),
            "mean_kinetic": float(np.mean(kinetics)),
            "std_kinetic": float(np.std(kinetics)),
            "mean_total": float(np.mean(totals)),
            "std_total": float(np.std(totals)),
            "min_potential": float(np.min(potentials)),
            "max_potential": float(np.max(potentials)),
        }

    def get_fitness_statistics(self, timestep: int) -> dict[str, float]:
        """Compute fitness-related statistics for alive walkers at a given timestep.

        Args:
            timestep: Recorded timestep index

        Returns:
            Dictionary with fitness statistics:
            - mean_fitness: Mean fitness potential V_fit
            - std_fitness: Std of fitness potential
            - mean_cloning_score: Mean cloning score S_i
            - std_cloning_score: Std of cloning score
            - mean_cloning_prob: Mean cloning probability π(S_i)
            - fraction_cloned: Fraction of walkers that cloned

        Reference: docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md
        """
        alive_walkers = self.get_alive_walkers(timestep)

        fitnesses = []
        cloning_scores = []
        cloning_probs = []
        will_clone = []

        for walker_id in alive_walkers:
            node_data = self.get_node_data(walker_id, timestep)
            fitnesses.append(node_data.get("fitness", 0.0))
            cloning_scores.append(node_data.get("cloning_score", 0.0))
            cloning_probs.append(node_data.get("cloning_prob", 0.0))
            will_clone.append(1.0 if node_data.get("will_clone", False) else 0.0)

        import numpy as np

        fitnesses = np.array(fitnesses)
        cloning_scores = np.array(cloning_scores)
        cloning_probs = np.array(cloning_probs)
        will_clone = np.array(will_clone)

        return {
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "mean_cloning_score": float(np.mean(cloning_scores)),
            "std_cloning_score": float(np.std(cloning_scores)),
            "mean_cloning_prob": float(np.mean(cloning_probs)),
            "fraction_cloned": float(np.mean(will_clone)),
        }

    def get_distance_statistics(self, timestep: int) -> dict[str, float]:
        """Compute distance-related statistics for alive walkers at a given timestep.

        Args:
            timestep: Recorded timestep index

        Returns:
            Dictionary with distance statistics:
            - mean_algorithmic_distance: Mean d_alg to companion
            - std_algorithmic_distance: Std of d_alg
            - mean_z_distance: Mean Z-score of distances
            - std_z_distance: Std of Z-scores
            - mean_rescaled_distance: Mean rescaled distance d'_i
            - mean_pos_sq_diff: Mean ||Δx||^2
            - mean_vel_sq_diff: Mean ||Δv||^2

        Reference: docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md
        """
        alive_walkers = self.get_alive_walkers(timestep)

        d_algs = []
        z_dists = []
        rescaled_dists = []
        pos_sqs = []
        vel_sqs = []

        for walker_id in alive_walkers:
            node_data = self.get_node_data(walker_id, timestep)
            d_algs.append(node_data.get("algorithmic_distance", 0.0))
            z_dists.append(node_data.get("z_distances", 0.0))
            rescaled_dists.append(node_data.get("rescaled_distances", 0.0))
            pos_sqs.append(node_data.get("pos_squared_diff", 0.0))
            vel_sqs.append(node_data.get("vel_squared_diff", 0.0))

        import numpy as np

        d_algs = np.array(d_algs)
        z_dists = np.array(z_dists)
        rescaled_dists = np.array(rescaled_dists)
        pos_sqs = np.array(pos_sqs)
        vel_sqs = np.array(vel_sqs)

        return {
            "mean_algorithmic_distance": float(np.mean(d_algs)),
            "std_algorithmic_distance": float(np.std(d_algs)),
            "mean_z_distance": float(np.mean(z_dists)),
            "std_z_distance": float(np.std(z_dists)),
            "mean_rescaled_distance": float(np.mean(rescaled_dists)),
            "mean_pos_sq_diff": float(np.mean(pos_sqs)),
            "mean_vel_sq_diff": float(np.mean(vel_sqs)),
        }

    def get_phase_potential_statistics(self, timestep: int) -> dict[str, float]:
        """Compute phase potential statistics for IG edges at a given timestep.

        Args:
            timestep: Recorded timestep index

        Returns:
            Dictionary with phase potential statistics:
            - mean_theta: Mean fitness phase θ_ij
            - std_theta: Std of fitness phase
            - mean_psi_real: Mean real part of ψ_ij
            - mean_psi_imag: Mean imaginary part of ψ_ij
            - mean_psi_magnitude: Mean |ψ_ij|
            - coherence: Mean cos(θ_ij) (phase coherence measure)

        Reference: docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md
        """
        ig_graph = self.get_ig_subgraph(timestep=timestep)

        thetas = []
        psi_reals = []
        psi_imags = []

        for _, _, edge_data in ig_graph.edges(data=True):
            theta = edge_data.get("theta_ij", 0.0)
            psi_real = edge_data.get("psi_ij_real", 1.0)
            psi_imag = edge_data.get("psi_ij_imag", 0.0)

            thetas.append(theta)
            psi_reals.append(psi_real)
            psi_imags.append(psi_imag)

        import numpy as np

        if not thetas:
            # No IG edges at this timestep
            return {
                "mean_theta": 0.0,
                "std_theta": 0.0,
                "mean_psi_real": 1.0,
                "mean_psi_imag": 0.0,
                "mean_psi_magnitude": 1.0,
                "coherence": 1.0,
            }

        thetas = np.array(thetas)
        psi_reals = np.array(psi_reals)
        psi_imags = np.array(psi_imags)
        psi_magnitudes = np.sqrt(psi_reals**2 + psi_imags**2)

        return {
            "mean_theta": float(np.mean(thetas)),
            "std_theta": float(np.std(thetas)),
            "mean_psi_real": float(np.mean(psi_reals)),
            "mean_psi_imag": float(np.mean(psi_imags)),
            "mean_psi_magnitude": float(np.mean(psi_magnitudes)),
            "coherence": float(np.mean(np.cos(thetas))),
        }

    def get_intermediate_fitness_data(self, walker_id: int, timestep: int) -> dict[str, float]:
        """Get all intermediate fitness computation data for a specific walker.

        Args:
            walker_id: Walker index
            timestep: Recorded timestep index

        Returns:
            Dictionary with intermediate fitness values:
            - z_rewards: Z-score of raw reward
            - z_distances: Z-score of algorithmic distance
            - rescaled_rewards: Rescaled reward r'_i
            - rescaled_distances: Rescaled distance d'_i
            - pos_squared_diff: ||Δx||^2
            - vel_squared_diff: ||Δv||^2
            - algorithmic_distance: d_alg to companion
            - fitness: Final fitness potential V_fit
            - cloning_score: Final cloning score S_i

        Reference: docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md
        """
        node_data = self.get_node_data(walker_id, timestep)

        return {
            "z_rewards": node_data.get("z_rewards", 0.0),
            "z_distances": node_data.get("z_distances", 0.0),
            "rescaled_rewards": node_data.get("rescaled_rewards", 0.0),
            "rescaled_distances": node_data.get("rescaled_distances", 0.0),
            "pos_squared_diff": node_data.get("pos_squared_diff", 0.0),
            "vel_squared_diff": node_data.get("vel_squared_diff", 0.0),
            "algorithmic_distance": node_data.get("algorithmic_distance", 0.0),
            "fitness": node_data.get("fitness", 0.0),
            "cloning_score": node_data.get("cloning_score", 0.0),
        }

    def get_gradient_statistics(self, timestep: int) -> dict[str, float] | None:
        """Compute fitness gradient statistics for CST edges at a given timestep.

        Args:
            timestep: Recorded timestep index

        Returns:
            Dictionary with gradient statistics if available, None otherwise:
            - mean_grad_norm: Mean ||∇V_fit||
            - std_grad_norm: Std of ||∇V_fit||
            - max_grad_norm: Maximum ||∇V_fit||
            - min_grad_norm: Minimum ||∇V_fit||

        Note: Only available if adaptive kinetics with fitness force was enabled.

        Reference: docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md
        """
        if self.history.fitness_gradients is None:
            return None

        alive_walkers = self.get_alive_walkers(timestep)

        grad_norms = []
        for walker_id in alive_walkers:
            # Get CST edge from (walker_id, timestep) → (walker_id, timestep+1)
            source = (walker_id, timestep)
            # Find target node (same walker, next timestep)
            targets = [
                (u, v)
                for u, v, d in self.graph.edges(source, data=True)
                if d["edge_type"] == "cst"
            ]

            if targets:
                _, target = targets[0]
                edge_data = self.graph.get_edge_data(source, target)
                grad_norm = edge_data.get("norm_grad_V_fit")
                if grad_norm is not None:
                    grad_norms.append(grad_norm)

        if not grad_norms:
            return None

        import numpy as np

        grad_norms = np.array(grad_norms)

        return {
            "mean_grad_norm": float(np.mean(grad_norms)),
            "std_grad_norm": float(np.std(grad_norms)),
            "max_grad_norm": float(np.max(grad_norms)),
            "min_grad_norm": float(np.min(grad_norms)),
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def num_cst_edges(self) -> int:
        """Total number of CST (temporal) edges."""
        return sum(1 for _, _, d in self.graph.edges(data=True) if d["edge_type"] == "cst")

    @property
    def num_ig_edges(self) -> int:
        """Total number of IG (spatial coupling) edges."""
        return sum(1 for _, _, d in self.graph.edges(data=True) if d["edge_type"] == "ig")

    @property
    def num_ia_edges(self) -> int:
        """Total number of IA (influence attribution) edges."""
        return sum(1 for _, _, d in self.graph.edges(data=True) if d["edge_type"] == "ia")

    @property
    def num_ig_distance_companion_edges(self) -> int:
        """Number of IG edges representing distance companion selection."""
        return sum(
            1
            for _, _, d in self.graph.edges(data=True)
            if d["edge_type"] == "ig" and d.get("is_distance_companion", False)
        )

    @property
    def num_ig_clone_companion_edges(self) -> int:
        """Number of IG edges representing clone companion selection."""
        return sum(
            1
            for _, _, d in self.graph.edges(data=True)
            if d["edge_type"] == "ig" and d.get("is_clone_companion", False)
        )

    @property
    def num_ig_both_companion_edges(self) -> int:
        """Number of IG edges where walker is both distance and clone companion."""
        return sum(
            1
            for _, _, d in self.graph.edges(data=True)
            if d["edge_type"] == "ig"
            and d.get("is_distance_companion", False)
            and d.get("is_clone_companion", False)
        )

    @property
    def total_nodes(self) -> int:
        """Total number of nodes (spacetime points)."""
        return self.graph.number_of_nodes()

    @property
    def num_triangles(self) -> int:
        """Total number of interaction triangles."""
        return len(getattr(self, "triangles", []))

    # ========================================================================
    # Serialization
    # ========================================================================

    def save(self, path: str):
        """Save FractalSet to disk.

        Args:
            path: File path for saving (should end in .pkl or .pickle)

        The graph is saved using pickle format, which preserves
        all node and edge attributes including torch.Tensors.

        Example:
            >>> fractal_set.save("fractal_set.pkl")
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self.graph, f)

    @classmethod
    def load(cls, path: str, history: RunHistory) -> FractalSet:
        """Load FractalSet from disk.

        Args:
            path: File path to load from
            history: RunHistory object (needed for delegation methods)

        Returns:
            FractalSet instance

        Example:
            >>> history = RunHistory.load("history.pt")
            >>> fractal_set = FractalSet.load("fractal_set.pkl", history)
        """
        import pickle

        fs = cls.__new__(cls)
        fs.history = history

        with open(path, "rb") as f:
            fs.graph = pickle.load(f)

        # Restore metadata
        fs.N = history.N
        fs.d = history.d
        fs.n_steps = history.n_steps
        fs.n_recorded = history.n_recorded
        fs.record_every = history.record_every
        fs.delta_t = float(fs.record_every)
        fs.recorded_steps = fs.graph.graph.get("recorded_steps")
        if fs.recorded_steps is None:
            fs.recorded_steps = fs._compute_recorded_steps()
        fs.triangles = fs.graph.graph.get("triangles", [])
        fs.epsilon_c = None
        fs.epsilon_d = None
        fs.lambda_alg = 0.0
        fs.hbar_eff = 1.0

        return fs

    # ========================================================================
    # Summary and Representation
    # ========================================================================

    def summary(self) -> str:
        """Generate human-readable summary of the Fractal Set.

        Returns:
            Multi-line summary string with graph statistics

        Example:
            >>> print(fractal_set.summary())
            FractalSet: 100 steps, 50 walkers, 2D
              Nodes: 550 spacetime points
              CST edges: 500 (temporal evolution)
              IG edges: 24500 (selection coupling)
                Distance companions: 500
                Clone companions: 500
                Both companions: 250
              Graph density: 0.162
        """
        density = nx.density(self.graph)

        lines = [
            f"FractalSet: {self.n_steps} steps, {self.N} walkers, {self.d}D",
            f"  Nodes: {self.total_nodes} spacetime points",
            f"  CST edges: {self.num_cst_edges} (temporal evolution)",
            f"  IG edges: {self.num_ig_edges} (selection coupling)",
            f"    Distance companions: {self.num_ig_distance_companion_edges}",
            f"    Clone companions: {self.num_ig_clone_companion_edges}",
            f"    Both companions: {self.num_ig_both_companion_edges}",
            f"  IA edges: {self.num_ia_edges} (influence attribution)",
            f"  Triangles: {self.num_triangles} (interaction simplices)",
            f"  Graph density: {density:.3f}",
            f"  Recorded: {self.n_recorded} timesteps (every {self.record_every} steps)",
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of FractalSet."""
        return (
            f"FractalSet(N={self.N}, d={self.d}, n_steps={self.n_steps}, "
            f"nodes={self.total_nodes}, cst={self.num_cst_edges}, ig={self.num_ig_edges}, "
            f"ia={self.num_ia_edges}, triangles={self.num_triangles})"
        )
