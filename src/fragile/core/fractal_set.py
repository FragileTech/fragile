"""Fractal Set data structure for representing EuclideanGas execution traces.

This module implements the Fractal Set as defined in the mathematical specification,
providing a complete graph-based representation of algorithm dynamics with two edge types:

- **CST (Causal Spacetime Tree)**: Temporal edges connecting walker states across timesteps
- **IG (Information Graph)**: Directed spatial edges representing selection
  coupling at each timestep

The Fractal Set is constructed from a RunHistory object and stores the complete execution
trace as a networkx directed graph with rich node and edge attributes.

Reference: old_docs/source/13_fractal_set_new/01_fractal_set.md
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import torch
from torch import Tensor

from fragile.core.history import RunHistory


class FractalSet:
    """Complete graph representation of an EuclideanGas run with CST and IG structure.

    The Fractal Set encodes the full execution trace of an EuclideanGas run as a directed
    graph where:

    - **Nodes** represent individual walkers at specific timesteps (spacetime points)
    - **CST edges** connect the same walker across consecutive timesteps (temporal evolution)
    - **IG edges** connect different walkers at the same timestep (selection coupling)

    All nodes store scalar (frame-invariant) quantities like fitness, energy, and status.
    Edges store vectorial quantities representing transitions (CST) or coupling (IG).

    Attributes:
        history: The RunHistory object containing the execution trace data
        graph: NetworkX directed graph storing nodes and edges
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
        >>> trajectory = fractal_set.get_walker_trajectory(walker_id=0)

    Reference: old_docs/source/13_fractal_set_new/01_fractal_set.md § 4.1
    """

    def __init__(
        self,
        history: RunHistory,
        epsilon_c: float | None = None,
        hbar_eff: float = 1.0,
    ):
        """Initialize FractalSet from a RunHistory object.

        Args:
            history: RunHistory object from EuclideanGas.run()
            epsilon_c: Cloning interaction range for phase potential computation.
                      If None, extracts from companion_selection.epsilon or defaults to 1.0.
            hbar_eff: Effective Planck constant for quantum-like coupling (default: 1.0)

        The constructor immediately builds the complete graph structure by:
        1. Creating nodes for all (walker, timestep) pairs
        2. Adding CST edges for temporal evolution
        3. Adding IG edges for selection coupling with phase potentials
        """
        self.history = history
        self.graph = nx.DiGraph()

        # Store metadata for convenience
        self.N = history.N
        self.d = history.d
        self.n_steps = history.n_steps
        self.n_recorded = history.n_recorded
        self.record_every = history.record_every

        # Store quantum parameters for IG edge phase potential computation
        self.epsilon_c = epsilon_c
        self.hbar_eff = hbar_eff

        # Build graph structure
        self._build_nodes()
        self._build_cst_edges()
        self._build_ig_edges()

    # ========================================================================
    # Construction Methods
    # ========================================================================

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

        Reference: old_docs/source/13_fractal_set_new/01_fractal_set.md § 1.1-1.2
        """
        for t_idx in range(self.n_recorded):
            # Absolute step number for this recorded index
            step = t_idx * self.record_every

            for walker_id in range(self.N):
                node_id = (walker_id, t_idx)

                # Extract position and velocity for energy computation
                x = self.history.x_final[t_idx, walker_id, :]
                v = self.history.v_final[t_idx, walker_id, :]

                # Compute kinetic energy
                E_kin = 0.5 * torch.sum(v**2).item()

                # Base node attributes (always available)
                attrs = {
                    "walker_id": walker_id,
                    "timestep": t_idx,
                    "absolute_step": step,
                    "t": step * 1.0,  # Continuous time (assuming Δt=1)
                    "x": x.clone(),
                    "v": v.clone(),
                    "E_kin": E_kin,
                }

                # Add alive status and per-step data (not available at t=0)
                if t_idx > 0:
                    # alive_mask has shape [n_recorded-1, N]
                    alive = self.history.alive_mask[t_idx - 1, walker_id].item()
                    attrs["alive"] = alive

                    # Per-walker per-step data (only for alive walkers with data)
                    attrs.update({
                        "fitness": self.history.fitness[t_idx - 1, walker_id].item(),
                        "reward": self.history.rewards[t_idx - 1, walker_id].item(),
                        "cloning_score": self.history.cloning_scores[t_idx - 1, walker_id].item(),
                        "cloning_prob": self.history.cloning_probs[t_idx - 1, walker_id].item(),
                        "will_clone": self.history.will_clone[t_idx - 1, walker_id].item(),
                        "companion_distance_id": self.history.companions_distance[
                            t_idx - 1, walker_id
                        ].item(),
                        "companion_clone_id": self.history.companions_clone[
                            t_idx - 1, walker_id
                        ].item(),
                        # Intermediate fitness computation scalars (from RunHistory)
                        "z_rewards": self.history.z_rewards[t_idx - 1, walker_id].item(),
                        "z_distances": self.history.z_distances[t_idx - 1, walker_id].item(),
                        "rescaled_rewards": self.history.rescaled_rewards[
                            t_idx - 1, walker_id
                        ].item(),
                        "rescaled_distances": self.history.rescaled_distances[
                            t_idx - 1, walker_id
                        ].item(),
                        "pos_squared_diff": self.history.pos_squared_differences[
                            t_idx - 1, walker_id
                        ].item(),
                        "vel_squared_diff": self.history.vel_squared_differences[
                            t_idx - 1, walker_id
                        ].item(),
                        "algorithmic_distance": self.history.distances[
                            t_idx - 1, walker_id
                        ].item(),
                        # Localized statistics (per-step, global case rho → ∞)
                        "mu_rewards": self.history.mu_rewards[t_idx - 1].item(),
                        "sigma_rewards": self.history.sigma_rewards[t_idx - 1].item(),
                        "mu_distances": self.history.mu_distances[t_idx - 1].item(),
                        "sigma_distances": self.history.sigma_distances[t_idx - 1].item(),
                    })
                else:
                    # At t=0, all walkers are alive by definition
                    attrs["alive"] = True

                self.graph.add_node(node_id, **attrs)

    def _build_cst_edges(self):
        """Construct CST (Causal Spacetime Tree) edges.

        Creates directed temporal edges (i, t) → (i, t+1) for each walker's evolution
        across consecutive timesteps. Only creates edges for alive walkers.

        Each CST edge stores:
        - Velocity at source and target: v_t, v_{t+1}
        - Velocity increment: Δv = v_{t+1} - v_t
        - Position displacement: Δx = x_{t+1} - x_t
        - Derived scalars: ||Δv||, ||Δx||

        Reference: old_docs/source/13_fractal_set_new/01_fractal_set.md § 2.1-2.3
        """
        for t_idx in range(self.n_recorded - 1):
            for walker_id in range(self.N):
                # Check if walker is alive at this timestep
                if not self.history.alive_mask[t_idx, walker_id].item():
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

                attrs = {
                    "edge_type": "cst",
                    "walker_id": walker_id,
                    "timestep": t_idx,
                    # Final states (after all operators)
                    "v_t": v_t.clone(),
                    "v_t1": v_t1.clone(),
                    "Delta_v": Delta_v.clone(),
                    "Delta_x": Delta_x.clone(),
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
        - Companion type flags: is_distance_companion, is_clone_companion

        Note: Creates a complete directed graph (tournament) among alive walkers.
        For k alive walkers, creates k(k-1) directed IG edges.

        Reference: old_docs/source/13_fractal_set_new/01_fractal_set.md § 3.1-3.3
        """
        for t_idx in range(self.n_recorded):
            # Skip t=0 as we don't have fitness data
            if t_idx == 0:
                continue

            # Get alive walkers at this timestep
            alive_mask = self.history.alive_mask[t_idx - 1, :]
            alive_indices = torch.where(alive_mask)[0].tolist()

            # Create directed edges between all pairs of alive walkers
            for i in alive_indices:
                # Get companion indices for this walker
                companion_distance_id = self.history.companions_distance[t_idx - 1, i].item()
                companion_clone_id = self.history.companions_clone[t_idx - 1, i].item()

                for j in alive_indices:
                    if i == j:
                        continue  # No self-edges

                    source = (i, t_idx)
                    target = (j, t_idx)

                    # Extract positions and velocities
                    x_i = self.history.x_final[t_idx, i, :]
                    x_j = self.history.x_final[t_idx, j, :]
                    v_i = self.history.v_final[t_idx, i, :]
                    v_j = self.history.v_final[t_idx, j, :]

                    # Compute relative quantities
                    Delta_x_ij = x_j - x_i
                    Delta_v_ij = v_j - v_i
                    distance = torch.norm(Delta_x_ij).item()

                    # Extract fitness values
                    fitness_i = self.history.fitness[t_idx - 1, i].item()
                    fitness_j = self.history.fitness[t_idx - 1, j].item()

                    # Antisymmetric cloning potential: V_clone(i→j) = Φ_j - Φ_i
                    V_clone = fitness_j - fitness_i

                    # Determine companion type for this edge
                    is_distance_companion = j == companion_distance_id
                    is_clone_companion = j == companion_clone_id

                    # Extract algorithmic distances for both walkers (Phase 3)
                    d_alg_i = self.history.distances[t_idx - 1, i].item()
                    d_alg_j = self.history.distances[t_idx - 1, j].item()

                    # Extract intermediate fitness data for both walkers (Phase 3)
                    z_rewards_i = self.history.z_rewards[t_idx - 1, i].item()
                    z_rewards_j = self.history.z_rewards[t_idx - 1, j].item()
                    z_distances_i = self.history.z_distances[t_idx - 1, i].item()
                    z_distances_j = self.history.z_distances[t_idx - 1, j].item()
                    rescaled_rewards_i = self.history.rescaled_rewards[t_idx - 1, i].item()
                    rescaled_rewards_j = self.history.rescaled_rewards[t_idx - 1, j].item()
                    rescaled_distances_i = self.history.rescaled_distances[t_idx - 1, i].item()
                    rescaled_distances_j = self.history.rescaled_distances[t_idx - 1, j].item()
                    pos_sq_diff_i = self.history.pos_squared_differences[t_idx - 1, i].item()
                    pos_sq_diff_j = self.history.pos_squared_differences[t_idx - 1, j].item()
                    vel_sq_diff_i = self.history.vel_squared_differences[t_idx - 1, i].item()
                    vel_sq_diff_j = self.history.vel_squared_differences[t_idx - 1, j].item()

                    # Determine epsilon_c for phase potential computation (Phase 3)
                    # Extraction from companion_selection is deferred - use provided or default
                    epsilon_c = self.epsilon_c if self.epsilon_c is not None else 1.0

                    # Compute phase potential: theta_ij = -d_alg_i^2 / (2 * epsilon_c^2 * hbar_eff)
                    # Reference: old_docs/source/13_fractal_set_new/01_fractal_set.md (3.22)
                    theta_ij = -(d_alg_i**2) / (2.0 * epsilon_c**2 * self.hbar_eff)

                    # Complex amplitude: psi_ij = exp(i*theta_ij)
                    import numpy as np

                    psi_ij = np.exp(1j * theta_ij)

                    attrs = {
                        "edge_type": "ig",
                        "source_walker": i,
                        "target_walker": j,
                        "timestep": t_idx,
                        "Delta_x_ij": Delta_x_ij.clone(),
                        "Delta_v_ij": Delta_v_ij.clone(),
                        "distance": distance,
                        "V_clone": V_clone,  # KEY: Antisymmetric cloning potential
                        "fitness_i": fitness_i,
                        "fitness_j": fitness_j,
                        "is_distance_companion": is_distance_companion,  # Distance companion flag
                        "is_clone_companion": is_clone_companion,  # Clone companion flag
                        # Phase 3: Algorithmic distance
                        "d_alg_i": d_alg_i,
                        "d_alg_j": d_alg_j,
                        # Phase 3: Phase potential and complex amplitude
                        "theta_ij": theta_ij,
                        "psi_ij_real": float(psi_ij.real),
                        "psi_ij_imag": float(psi_ij.imag),
                        # Phase 3: Intermediate fitness data for both walkers
                        "z_rewards_i": z_rewards_i,
                        "z_rewards_j": z_rewards_j,
                        "z_distances_i": z_distances_i,
                        "z_distances_j": z_distances_j,
                        "rescaled_rewards_i": rescaled_rewards_i,
                        "rescaled_rewards_j": rescaled_rewards_j,
                        "rescaled_distances_i": rescaled_distances_i,
                        "rescaled_distances_j": rescaled_distances_j,
                        "pos_sq_diff_i": pos_sq_diff_i,
                        "pos_sq_diff_j": pos_sq_diff_j,
                        "vel_sq_diff_i": vel_sq_diff_i,
                        "vel_sq_diff_j": vel_sq_diff_j,
                    }

                    self.graph.add_edge(source, target, **attrs)

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

        Reference: old_docs/source/13_fractal_set_new/01_fractal_set.md
        """
        alive_walkers = self.get_alive_walkers(timestep)

        potentials = []
        kinetics = []

        for walker_id in alive_walkers:
            node_data = self.get_node_data(walker_id, timestep)
            U = node_data.get("U", 0.0)
            v = node_data.get("v_final")

            potentials.append(U)
            if v is not None:
                KE = 0.5 * torch.sum(v**2).item()
                kinetics.append(KE)

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

        Reference: old_docs/source/13_fractal_set_new/01_fractal_set.md
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

        Reference: old_docs/source/13_fractal_set_new/01_fractal_set.md
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
            - mean_theta: Mean phase potential θ_ij
            - std_theta: Std of phase potential
            - mean_psi_real: Mean real part of ψ_ij
            - mean_psi_imag: Mean imaginary part of ψ_ij
            - mean_psi_magnitude: Mean |ψ_ij|
            - coherence: Mean cos(θ_ij) (phase coherence measure)

        Reference: old_docs/source/13_fractal_set_new/01_fractal_set.md (3.22)
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

        Reference: old_docs/source/13_fractal_set_new/01_fractal_set.md
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

        Reference: old_docs/source/13_fractal_set_new/01_fractal_set.md
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
            f"  Graph density: {density:.3f}",
            f"  Recorded: {self.n_recorded} timesteps (every {self.record_every} steps)",
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of FractalSet."""
        return (
            f"FractalSet(N={self.N}, d={self.d}, n_steps={self.n_steps}, "
            f"nodes={self.total_nodes}, cst={self.num_cst_edges}, ig={self.num_ig_edges})"
        )
