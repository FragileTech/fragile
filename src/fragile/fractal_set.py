"""
Fractal Set: Discrete Spacetime Representation of Swarm Evolution

This module implements a graph-based representation of swarm dynamics as a discrete
spacetime structure. Each node represents a (walker, timestep) event, and edges
represent causal connections (cloning relationships and temporal evolution).

The FractalSet provides a complete record of the swarm's history, including:
- Walker states at each timestep
- Cloning events and lineages
- Variance and fitness metrics
- Geometric partitions (high-error vs low-error sets)

Mathematical Foundation:
- Discrete spacetime: Nodes are events (i, t) where i is walker index, t is time
- Causal structure: Edges encode parent-child relationships from cloning
- Graph attributes: Store global metrics (variance, energy, etc.)
- Node attributes: Store local walker state (position, velocity, fitness)
- Edge attributes: Store transition information (cloning probability, distance)

References:
- CLAUDE.md § 13_fractal_set/ - Discrete spacetime and lattice QFT formulation
- This implements a causal graph structure for analyzing swarm evolution
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import networkx as nx
import numpy as np
import torch
from torch import Tensor


if TYPE_CHECKING:
    from fragile.euclidean_gas import EuclideanGas, SwarmState


class FractalSet:
    """
    Graph-based representation of swarm evolution as discrete spacetime.

    The FractalSet is a directed graph where:
    - **Nodes**: (walker_id, timestep) tuples representing events
    - **Edges**: Causal connections (cloning or temporal evolution)
    - **Graph attributes**: Global run metadata
    - **Node attributes**: Walker state at that event
    - **Edge attributes**: Transition information

    Node Attributes:
        - x: position [d]
        - v: velocity [d]
        - potential: potential energy U(x)
        - reward: reward R(x, v)
        - fitness: fitness potential V_fit
        - high_error: boolean indicating if walker is in H_k
        - positional_error: distance from swarm centroid
        - alive: boolean indicating if walker is alive (in bounds)

    Edge Attributes:
        - edge_type: 'cloning' or 'kinetic'
        - cloning_prob: probability of cloning (if applicable)
        - companion_id: ID of companion walker (if cloning)
        - distance: algorithmic distance to companion

    Graph Attributes:
        - N: number of walkers
        - d: spatial dimension
        - n_steps: total number of timesteps
        - var_x_trajectory: position variance at each timestep
        - var_v_trajectory: velocity variance at each timestep
        - n_alive_trajectory: number of alive walkers at each timestep
        - centroid_trajectory: swarm centroid at each timestep
        - parameters: algorithm parameters (EuclideanGasParams)
    """

    def __init__(self, N: int, d: int, params: dict | None = None):
        """
        Initialize empty FractalSet.

        Args:
            N: Number of walkers
            d: Spatial dimension
            params: Optional dictionary of algorithm parameters
        """
        self.graph = nx.DiGraph()
        self.N = N
        self.d = d

        # Initialize graph attributes
        self.graph.graph["N"] = N
        self.graph.graph["d"] = d
        self.graph.graph["n_steps"] = 0
        self.graph.graph["parameters"] = params or {}

        # Trajectories (will be updated as we add timesteps)
        self.graph.graph["var_x_trajectory"] = []
        self.graph.graph["var_v_trajectory"] = []
        self.graph.graph["n_alive_trajectory"] = []
        self.graph.graph["centroid_trajectory"] = []

        # Current timestep
        self.current_step = 0

    def add_timestep(
        self,
        state: SwarmState,
        timestep: int,
        high_error_mask: Tensor | None = None,
        alive_mask: Tensor | None = None,
        fitness: Tensor | None = None,
        potential: Tensor | None = None,
        reward: Tensor | None = None,
        companions: Tensor | None = None,
        cloning_probs: Tensor | None = None,
        distances: Tensor | None = None,
        rescaled_reward: Tensor | None = None,
        rescaled_distance: Tensor | None = None,
        clone_uniform_sample: Tensor | None = None,
    ) -> None:
        """
        Add a complete timestep to the FractalSet.

        Args:
            state: SwarmState at this timestep
            timestep: Current timestep index
            high_error_mask: Boolean mask indicating H_k membership [N]
            alive_mask: Boolean mask indicating alive walkers [N]
            fitness: Fitness potential V_fit for each walker [N]
            potential: Potential energy U(x) for each walker [N]
            reward: Reward R(x,v) for each walker [N]
            companions: Companion indices for each walker [N]
            cloning_probs: Cloning probability for each walker [N]
            distances: Algorithmic distance to companion [N]
            rescaled_reward: Rescaled reward (fitness-based) [N]
            rescaled_distance: Rescaled algorithmic distance [N]
            clone_uniform_sample: Random uniform [0,1] sample for cloning decision [N]
        """
        # Convert to numpy (detach first in case tensors require grad)
        x_np = state.x.detach().cpu().numpy()
        v_np = state.v.detach().cpu().numpy()

        # Compute centroids
        centroid = x_np.mean(axis=0)
        v_centroid = v_np.mean(axis=0)

        # Compute positional errors
        positional_errors = np.sqrt(np.sum((x_np - centroid[None, :]) ** 2, axis=1))

        # Default masks if not provided
        if high_error_mask is None:
            # Use median threshold
            threshold = np.median(positional_errors)
            high_error_mask = torch.tensor(positional_errors > threshold)

        if alive_mask is None:
            alive_mask = torch.ones(self.N, dtype=torch.bool)

        # Convert masks to numpy (detach first in case they require grad)
        high_error_np = high_error_mask.detach().cpu().numpy()
        alive_np = alive_mask.detach().cpu().numpy()

        # Add nodes for each walker at this timestep
        for i in range(self.N):
            node_id = (i, timestep)

            node_attrs = {
                "walker_id": i,
                "timestep": timestep,
                "x": x_np[i],
                "v": v_np[i],
                "high_error": bool(high_error_np[i]),
                "positional_error": float(positional_errors[i]),
                "alive": bool(alive_np[i]),
            }

            # Add optional attributes
            if fitness is not None:
                node_attrs["fitness"] = float(fitness[i].item())
            if potential is not None:
                node_attrs["potential"] = float(potential[i].item())
            if reward is not None:
                node_attrs["reward"] = float(reward[i].item())

            # Add new required attributes
            if rescaled_reward is not None:
                node_attrs["rescaled_reward"] = float(rescaled_reward[i].item())
            if distances is not None:
                node_attrs["distance_to_companion"] = float(distances[i].item())
            if rescaled_distance is not None:
                node_attrs["rescaled_distance_to_companion"] = float(rescaled_distance[i].item())
            if companions is not None:
                node_attrs["companion_index"] = int(companions[i].item())
            if cloning_probs is not None:
                node_attrs["clone_score"] = float(cloning_probs[i].item())
            if clone_uniform_sample is not None:
                node_attrs["clone_uniform_sample"] = float(clone_uniform_sample[i].item())

            self.graph.add_node(node_id, **node_attrs)

        # Add edges from previous timestep (temporal evolution)
        if timestep > 0:
            for i in range(self.N):
                prev_node = (i, timestep - 1)
                curr_node = (i, timestep)

                # Default edge: kinetic evolution
                edge_attrs = {
                    "edge_type": "kinetic",
                }

                # Add distance if available
                if distances is not None and i < len(distances):
                    dist_val = float(distances[i].item())
                    edge_attrs["distance"] = dist_val
                    edge_attrs["algorithmic_distance"] = dist_val  # Alias for compatibility

                # If we have companion information, this might be a cloning edge
                if companions is not None and cloning_probs is not None:
                    companion_id = int(companions[i].item())
                    cloning_prob = float(cloning_probs[i].item())

                    # Store companion probability
                    edge_attrs["companion_probability"] = cloning_prob

                    # If cloning probability is high, mark as cloning edge
                    if cloning_prob > 0.5:
                        edge_attrs["edge_type"] = "cloning"
                        edge_attrs["companion_id"] = companion_id
                        edge_attrs["cloning_prob"] = cloning_prob

                        # Add edge from companion (parent) to current walker (child)
                        parent_node = (companion_id, timestep - 1)
                        self.graph.add_edge(parent_node, curr_node, **edge_attrs)
                    else:
                        # Regular kinetic edge (self-evolution)
                        self.graph.add_edge(prev_node, curr_node, **edge_attrs)
                else:
                    # No cloning information, add simple temporal edge
                    self.graph.add_edge(prev_node, curr_node, **edge_attrs)

        # Update graph-level trajectories
        # Compute variance as (1/N) sum_i ||x_i - μ_x||² (matches VectorizedOps.variance_position)
        var_x = np.mean(np.sum((x_np - centroid) ** 2, axis=-1))
        var_v = np.mean(np.sum((v_np - v_centroid) ** 2, axis=-1))
        n_alive = int(alive_np.sum())

        self.graph.graph["var_x_trajectory"].append(float(var_x))
        self.graph.graph["var_v_trajectory"].append(float(var_v))
        self.graph.graph["n_alive_trajectory"].append(n_alive)
        self.graph.graph["centroid_trajectory"].append(centroid.tolist())
        self.graph.graph["n_steps"] = timestep + 1

        self.current_step = timestep

    def get_parent_ids(self, timestep: int) -> np.ndarray:
        """Extract parent IDs for all walkers at a given timestep.

        For each walker at timestep t, find the parent walker ID from timestep t-1
        by following the incoming edge. If the edge is a cloning edge, the parent
        is the companion_id. Otherwise, the parent is the walker itself (kinetic edge).

        Args:
            timestep: Timestep to extract parent IDs for (must be > 0)

        Returns:
            Array of parent IDs [N], where parent_ids[i] is the parent of walker i
        """
        if timestep <= 0:
            raise ValueError(f"Cannot get parent IDs for timestep {timestep} (must be > 0)")

        if timestep >= self.graph.graph.get("n_steps", 0):
            raise ValueError(
                f"Timestep {timestep} out of range (n_steps={self.graph.graph.get('n_steps', 0)})"
            )

        parent_ids = np.zeros(self.N, dtype=int)

        for i in range(self.N):
            curr_node = (i, timestep)

            # Find incoming edges to this node
            predecessors = list(self.graph.predecessors(curr_node))

            if len(predecessors) == 0:
                # No incoming edge - use self as parent (shouldn't happen in normal case)
                parent_ids[i] = i
            elif len(predecessors) == 1:
                # Normal case: one incoming edge
                parent_node = predecessors[0]
                parent_id, parent_time = parent_node

                # Verify parent is from previous timestep
                assert parent_time == timestep - 1, f"Parent time {parent_time} != {timestep - 1}"

                parent_ids[i] = parent_id
            else:
                # Multiple parents (shouldn't happen) - use first
                parent_node = predecessors[0]
                parent_id, _ = parent_node
                parent_ids[i] = parent_id

        return parent_ids

    @classmethod
    def from_run(
        cls,
        gas: EuclideanGas,
        n_steps: int,
        x_init: Tensor | None = None,
        v_init: Tensor | None = None,
        record_fitness: bool = True,
    ) -> FractalSet:
        """
        Create FractalSet by running EuclideanGas and recording full history.

        Args:
            gas: EuclideanGas instance to run
            n_steps: Number of steps to run
            x_init: Initial positions (optional)
            v_init: Initial velocities (optional)
            record_fitness: Whether to compute and record fitness at each step

        Returns:
            FractalSet with complete run history
        """
        from fragile.companion_selection import select_companions_softmax

        # Initialize fractal set
        fractal_set = cls(
            N=gas.params.N,
            d=gas.params.d,
            params={"euclidean_gas_params": gas.params.model_dump()},
        )

        # Initialize state
        state = gas.initialize_state(x_init, v_init)

        # Record initial state
        alive_mask = torch.ones(gas.params.N, dtype=torch.bool)
        if gas.bounds is not None:
            alive_mask = gas.bounds.contains(state.x)

        # Compute initial metrics
        if record_fitness:
            potential = gas.params.potential.evaluate(state.x)
            reward = -potential

            # Compute companions and distances
            companions = select_companions_softmax(
                state.x,
                state.v,
                alive_mask,
                epsilon=gas.params.cloning.get_epsilon_c(),
                lambda_alg=gas.params.cloning.lambda_alg,
                exclude_self=True,
            )

            x_companion = state.x[companions]
            v_companion = state.v[companions]
            pos_diff_sq = torch.sum((state.x - x_companion) ** 2, dim=-1)
            vel_diff_sq = torch.sum((state.v - v_companion) ** 2, dim=-1)
            distances = torch.sqrt(pos_diff_sq + gas.params.cloning.lambda_alg * vel_diff_sq)

            # Simplified fitness (just use distance as proxy)
            fitness = 1.0 / (distances + 1e-6)
        else:
            potential = None
            reward = None
            companions = None
            distances = None
            fitness = None

        # Compute high-error mask
        mu_x = torch.mean(state.x, dim=0, keepdim=True)
        positional_error = torch.sqrt(torch.sum((state.x - mu_x) ** 2, dim=-1))
        threshold = torch.median(positional_error)
        high_error_mask = positional_error > threshold

        fractal_set.add_timestep(
            state=state,
            timestep=0,
            high_error_mask=high_error_mask,
            alive_mask=alive_mask,
            fitness=fitness,
            potential=potential,
            reward=reward,
            companions=companions,
            cloning_probs=None,
            distances=distances,
        )

        # Run for n_steps
        for t in range(1, n_steps + 1):
            # Check if all walkers are dead
            if gas.bounds is not None:
                alive_mask = gas.bounds.contains(state.x)
                if alive_mask.sum() == 0:
                    break

            # Perform one step
            _, state = gas.step(state)

            # Update alive mask
            if gas.bounds is not None:
                alive_mask = gas.bounds.contains(state.x)
            else:
                alive_mask = torch.ones(gas.params.N, dtype=torch.bool)

            # Compute metrics for this timestep
            if record_fitness:
                potential = gas.params.potential.evaluate(state.x)
                reward = -potential

                companions = select_companions_softmax(
                    state.x,
                    state.v,
                    alive_mask,
                    epsilon=gas.params.cloning.get_epsilon_c(),
                    lambda_alg=gas.params.cloning.lambda_alg,
                    exclude_self=True,
                )

                x_companion = state.x[companions]
                v_companion = state.v[companions]
                pos_diff_sq = torch.sum((state.x - x_companion) ** 2, dim=-1)
                vel_diff_sq = torch.sum((state.v - v_companion) ** 2, dim=-1)
                distances = torch.sqrt(pos_diff_sq + gas.params.cloning.lambda_alg * vel_diff_sq)

                fitness = 1.0 / (distances + 1e-6)

                # Estimate cloning probabilities (simplified)
                cloning_probs = torch.clamp(fitness / fitness.mean(), 0.0, 1.0)
            else:
                potential = None
                reward = None
                companions = None
                distances = None
                fitness = None
                cloning_probs = None

            # Compute high-error mask
            mu_x = torch.mean(state.x, dim=0, keepdim=True)
            positional_error = torch.sqrt(torch.sum((state.x - mu_x) ** 2, dim=-1))
            threshold = torch.median(positional_error)
            high_error_mask = positional_error > threshold

            fractal_set.add_timestep(
                state=state,
                timestep=t,
                high_error_mask=high_error_mask,
                alive_mask=alive_mask,
                fitness=fitness,
                potential=potential,
                reward=reward,
                companions=companions,
                cloning_probs=cloning_probs,
                distances=distances,
            )

        return fractal_set

    def get_walker_trajectory(self, walker_id: int) -> dict[str, Any]:
        """
        Extract complete trajectory for a single walker.

        Args:
            walker_id: ID of walker to extract

        Returns:
            Dictionary with trajectory data:
                - timesteps: list of timesteps
                - positions: list of position arrays
                - velocities: list of velocity arrays
                - high_error: list of high-error flags
                - alive: list of alive flags
                - (optional) fitness, potential, reward
        """
        trajectory = {
            "timesteps": [],
            "positions": [],
            "velocities": [],
            "high_error": [],
            "alive": [],
        }

        for t in range(self.graph.graph["n_steps"]):
            node_id = (walker_id, t)
            if not self.graph.has_node(node_id):
                break

            node_data = self.graph.nodes[node_id]
            trajectory["timesteps"].append(t)
            trajectory["positions"].append(node_data["x"])
            trajectory["velocities"].append(node_data["v"])
            trajectory["high_error"].append(node_data["high_error"])
            trajectory["alive"].append(node_data["alive"])

            # Add optional attributes if present
            for attr in ["fitness", "potential", "reward"]:
                if attr in node_data:
                    if attr not in trajectory:
                        trajectory[attr] = []
                    trajectory[attr].append(node_data[attr])

        return trajectory

    def get_lineage(self, walker_id: int, timestep: int) -> list[tuple[int, int]]:
        """
        Trace lineage backwards to find all ancestors (cloning parents).

        Args:
            walker_id: Walker to trace
            timestep: Starting timestep

        Returns:
            List of (walker_id, timestep) tuples representing ancestors
        """
        lineage = []
        current = (walker_id, timestep)

        while self.graph.in_degree(current) > 0:
            # Get parent(s)
            parents = list(self.graph.predecessors(current))

            # Follow cloning edges preferentially
            cloning_parent = None
            for parent in parents:
                edge_data = self.graph.edges[parent, current]
                if edge_data.get("edge_type") == "cloning":
                    cloning_parent = parent
                    break

            if cloning_parent:
                lineage.append(cloning_parent)
                current = cloning_parent
            elif parents:
                # Fall back to temporal evolution
                lineage.append(parents[0])
                current = parents[0]
            else:
                break

        return lineage

    def get_timestep_snapshot(self, timestep: int) -> dict[str, Any]:
        """
        Get complete swarm state at a specific timestep.

        Args:
            timestep: Timestep to query

        Returns:
            Dictionary with swarm state:
                - positions: [N, d] array
                - velocities: [N, d] array
                - high_error_mask: [N] boolean array
                - alive_mask: [N] boolean array
                - centroid: [d] array
                - variance: scalar
                - (optional) fitness, potential, reward arrays
        """
        positions = []
        velocities = []
        high_error = []
        alive = []

        # Collect optional attributes
        optional_attrs = {}

        for i in range(self.N):
            node_id = (i, timestep)
            if not self.graph.has_node(node_id):
                continue

            node_data = self.graph.nodes[node_id]
            positions.append(node_data["x"])
            velocities.append(node_data["v"])
            high_error.append(node_data["high_error"])
            alive.append(node_data["alive"])

            # Collect optional attributes
            for attr in ["fitness", "potential", "reward"]:
                if attr in node_data:
                    if attr not in optional_attrs:
                        optional_attrs[attr] = []
                    optional_attrs[attr].append(node_data[attr])

        snapshot = {
            "positions": np.array(positions),
            "velocities": np.array(velocities),
            "high_error_mask": np.array(high_error),
            "alive_mask": np.array(alive),
            "centroid": self.graph.graph["centroid_trajectory"][timestep],
            "variance": self.graph.graph["var_x_trajectory"][timestep],
        }

        # Add optional attributes
        for attr, values in optional_attrs.items():
            snapshot[attr] = np.array(values)

        return snapshot

    def get_cloning_events(self) -> list[dict[str, Any]]:
        """
        Extract all cloning events from the graph.

        Returns:
            List of dictionaries, each representing a cloning event:
                - timestep: when cloning occurred
                - parent_id: ID of parent walker (companion)
                - child_id: ID of child walker (cloned)
                - cloning_prob: cloning probability
                - parent_pos: parent position
                - child_pos: child position (after cloning)
        """
        events = []

        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == "cloning":
                parent_id, _parent_t = u
                child_id, child_t = v

                event = {
                    "timestep": child_t,
                    "parent_id": parent_id,
                    "child_id": child_id,
                    "cloning_prob": data.get("cloning_prob"),
                    "parent_pos": self.graph.nodes[u]["x"],
                    "child_pos": self.graph.nodes[v]["x"],
                }

                events.append(event)

        return events

    def summary_statistics(self) -> dict[str, Any]:
        """
        Compute summary statistics for the entire run.

        Returns:
            Dictionary with statistics:
                - total_nodes: number of nodes (walker-timestep events)
                - total_edges: number of edges (transitions)
                - n_cloning_events: number of cloning events
                - initial_variance: initial position variance (if available)
                - final_variance: final position variance (if available)
                - variance_reduction: fractional reduction (if available)
                - mean_high_error_fraction: average fraction in H_k over time
        """
        n_cloning = sum(
            1 for _, _, d in self.graph.edges(data=True) if d.get("edge_type") == "cloning"
        )

        var_traj = np.array(self.graph.graph["var_x_trajectory"])

        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "n_cloning_events": n_cloning,
            "n_steps": self.graph.graph["n_steps"],
        }

        # Only add variance stats if we have data
        if len(var_traj) > 0:
            initial_var = var_traj[0]
            final_var = var_traj[-1]
            stats["initial_variance"] = float(initial_var)
            stats["final_variance"] = float(final_var)
            stats["variance_reduction"] = float(1.0 - final_var / (initial_var + 1e-10))

        # Compute mean high-error fraction
        if self.graph.graph["n_steps"] > 0:
            high_error_fractions = []
            for t in range(self.graph.graph["n_steps"]):
                n_high_error = sum(
                    1
                    for i in range(self.N)
                    if self.graph.has_node((i, t)) and self.graph.nodes[i, t]["high_error"]
                )
                high_error_fractions.append(n_high_error / self.N)

            stats["mean_high_error_fraction"] = float(np.mean(high_error_fractions))
        else:
            stats["mean_high_error_fraction"] = 0.0

        return stats

    def save(self, filepath: str) -> None:
        """
        Save FractalSet to disk using GraphML format.

        Args:
            filepath: Path to save file (will add .graphml extension if missing)
        """
        import json

        if not filepath.endswith(".graphml"):
            filepath += ".graphml"

        # Create a copy of the graph for serialization
        graph_copy = self.graph.copy()

        # Convert dict parameters to JSON string (GraphML doesn't support dicts)
        if "parameters" in graph_copy.graph and isinstance(graph_copy.graph["parameters"], dict):
            graph_copy.graph["parameters"] = json.dumps(graph_copy.graph["parameters"])

        # Convert numpy arrays to lists for serialization
        for node_id in graph_copy.nodes():
            node_data = graph_copy.nodes[node_id]
            if "x" in node_data and isinstance(node_data["x"], np.ndarray):
                graph_copy.nodes[node_id]["x"] = json.dumps(node_data["x"].tolist())
            if "v" in node_data and isinstance(node_data["v"], np.ndarray):
                graph_copy.nodes[node_id]["v"] = json.dumps(node_data["v"].tolist())

        # Convert list attributes to JSON strings
        for attr in [
            "var_x_trajectory",
            "var_v_trajectory",
            "n_alive_trajectory",
            "centroid_trajectory",
        ]:
            if attr in graph_copy.graph and isinstance(graph_copy.graph[attr], list):
                graph_copy.graph[attr] = json.dumps(graph_copy.graph[attr])

        nx.write_graphml(graph_copy, filepath)

    @classmethod
    def load(cls, filepath: str) -> FractalSet:
        """
        Load FractalSet from disk.

        Args:
            filepath: Path to GraphML file

        Returns:
            Loaded FractalSet instance
        """
        import json

        graph = nx.read_graphml(filepath, node_type=str)

        # Create instance
        fractal_set = cls(N=int(graph.graph["N"]), d=int(graph.graph["d"]))

        # Convert JSON strings back to Python objects
        if "parameters" in graph.graph and isinstance(graph.graph["parameters"], str):
            graph.graph["parameters"] = json.loads(graph.graph["parameters"])

        for attr in [
            "var_x_trajectory",
            "var_v_trajectory",
            "n_alive_trajectory",
            "centroid_trajectory",
        ]:
            if attr in graph.graph and isinstance(graph.graph[attr], str):
                graph.graph[attr] = json.loads(graph.graph[attr])

        # Convert node attribute JSON strings back to lists/arrays
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            if "x" in node_data and isinstance(node_data["x"], str):
                graph.nodes[node_id]["x"] = np.array(json.loads(node_data["x"]))
            if "v" in node_data and isinstance(node_data["v"], str):
                graph.nodes[node_id]["v"] = np.array(json.loads(node_data["v"]))

        # Convert string tuples back to int tuples for node IDs
        # (GraphML serializes tuples as strings)
        mapping = {}
        for node_id in list(graph.nodes()):
            if isinstance(node_id, str) and "," in node_id:
                # Parse tuple string
                parts = node_id.strip("()").split(",")
                new_id = (int(parts[0]), int(parts[1]))
                mapping[node_id] = new_id

        fractal_set.graph = nx.relabel_nodes(graph, mapping)

        return fractal_set
