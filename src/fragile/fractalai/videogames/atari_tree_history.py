"""AtariTreeHistory: graph-backed history recorder for Atari games.

Uses ``DataTree`` to store per-walker Atari-specific data with cloning-aware
parent edges.  When walker *w* clones from companion *c* at step *t*, the
parent edge points to ``(t-1, c)`` instead of ``(t-1, w)``, turning the flat
list of iterations into a proper tree reflecting cloning lineage.

Node IDs are deterministic:

    node_id = step * (N + 1) + walker + 1
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from fragile.fractalai.core.tree import DataTree, DEFAULT_ROOT_ID
from fragile.fractalai.videogames.atari_history import AtariHistory
from fragile.fractalai.robots.robotic_history import RoboticHistory

if TYPE_CHECKING:
    from fragile.fractalai.videogames.atari_gas import WalkerState


def _node_id(step: int, walker: int, N: int) -> int:
    """Deterministic node ID for a (step, walker) pair."""
    return step * (N + 1) + walker + 1


class AtariTreeHistory:
    """Graph-backed history recorder for Atari games.

    Parameters
    ----------
    N : int
        Number of walkers.
    game_name : str
        Name of the Atari game.
    max_iterations : int
        Maximum number of iterations (informational).
    """

    def __init__(self, N: int, game_name: str = "", max_iterations: int = 1000):
        self.N = N
        self.game_name = game_name
        self.max_iterations = max_iterations

        self._tree = DataTree(
            names=[],
            node_names=[],
            edge_names=[],
        )

        self._n_recorded: int = 0
        self._step_metadata: list[dict] = []

    # ------------------------------------------------------------------
    # Deterministic node helpers
    # ------------------------------------------------------------------

    def _nid(self, step: int, walker: int) -> int:
        return _node_id(step, walker, self.N)

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record_initial_atari_state(self, state: WalkerState) -> None:
        """Record initial state at t=0.

        Creates N child nodes of root, one per walker.
        """
        step = 0
        for w in range(self.N):
            nid = self._nid(step, w)
            action_w = state.actions[w]
            action_val = int(action_w) if np.ndim(action_w) == 0 else action_w.tolist()
            node_data = {
                "observations": state.observations[w].detach().cpu().numpy(),
                "reward": 0.0,
                "step_reward": 0.0,
                "virtual_reward": 0.0,
                "done": False,
                "truncated": False,
                "alive": True,
                "action": action_val,
                "will_clone": False,
                "clone_companion": -1,
            }
            self._tree.append_leaf(
                leaf_id=nid,
                parent_id=int(DEFAULT_ROOT_ID),
                node_data=node_data,
                edge_data={},
                epoch=step,
            )

        self._step_metadata.append({
            "n_alive": int(state.alive.sum().item()),
            "num_cloned": 0,
            "mean_reward": 0.0,
            "max_reward": 0.0,
            "min_reward": 0.0,
            "mean_virtual_reward": 0.0,
            "max_virtual_reward": 0.0,
            "best_frame": None,
            "best_walker_idx": 0,
        })
        self._n_recorded = 1

    def record_atari_step(
        self,
        state_before: WalkerState,
        state_after_clone: WalkerState,
        state_final: WalkerState,
        info: dict,
        clone_companions: torch.Tensor,
        will_clone: torch.Tensor,
        best_frame: np.ndarray | None = None,
    ) -> None:
        """Record a single Atari step with cloning-aware parent edges.

        When walker *w* cloned from companion *c*, the parent of ``(step, w)``
        is ``(step-1, c)`` instead of ``(step-1, w)``.
        """
        step = self._n_recorded

        for w in range(self.N):
            nid = self._nid(step, w)

            # Cloning-aware parent edge
            if will_clone[w].item():
                companion = int(clone_companions[w].item())
                parent_id = _node_id(step - 1, companion, self.N)
            else:
                parent_id = _node_id(step - 1, w, self.N)

            vr = 0.0
            if state_final.virtual_rewards is not None:
                vr = float(state_final.virtual_rewards[w].item())

            action_w = state_final.actions[w]
            action_val = int(action_w) if np.ndim(action_w) == 0 else action_w.tolist()
            node_data = {
                "observations": state_final.observations[w].detach().cpu().numpy(),
                "reward": float(state_final.rewards[w].item()),
                "step_reward": float(state_final.step_rewards[w].item()),
                "virtual_reward": vr,
                "done": bool(state_final.dones[w].item()),
                "truncated": bool(state_final.truncated[w].item()),
                "alive": bool(state_final.alive[w].item()),
                "action": action_val,
                "will_clone": bool(will_clone[w].item()),
                "clone_companion": int(clone_companions[w].item()) if will_clone[w].item() else -1,
            }

            self._tree.append_leaf(
                leaf_id=nid,
                parent_id=parent_id,
                node_data=node_data,
                edge_data={},
                epoch=step,
            )

        # Step metadata
        rewards = state_final.rewards
        best_idx = int(rewards.argmax().item())

        vr_mean = 0.0
        vr_max = 0.0
        if state_final.virtual_rewards is not None:
            vr_mean = float(state_final.virtual_rewards.mean().item())
            vr_max = float(state_final.virtual_rewards.max().item())

        self._step_metadata.append({
            "n_alive": int(state_final.alive.sum().item()),
            "num_cloned": int(will_clone.sum().item()),
            "mean_reward": float(rewards.mean().item()),
            "max_reward": float(rewards.max().item()),
            "min_reward": float(rewards.min().item()),
            "mean_virtual_reward": vr_mean,
            "max_virtual_reward": vr_max,
            "best_frame": best_frame,
            "best_walker_idx": best_idx,
        })

        self._n_recorded += 1

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_dead_branches(self) -> int:
        """Remove orphaned branches left behind by cloned-over dead walkers.

        After cloning, a dead walker *w*'s previous leaf ``(t, w)`` loses its
        only potential child (the new node ``(t+1, w)`` points to the
        companion's parent instead).  That makes ``(t, w)`` an orphaned leaf.
        This method finds all such orphaned leaves in the tree (any leaf that
        is *not* one of the latest-step walker nodes) and recursively prunes
        them and their exclusively-dead ancestors.

        Returns
        -------
        int
            Number of graph nodes removed.
        """
        if self._n_recorded <= 1:
            return 0

        step = self._n_recorded - 1
        # Current walker leaves — the only leaves we want to keep
        alive_leafs: set[int] = set()
        for w in range(self.N):
            nid = self._nid(step, w)
            if nid in self._tree.data.nodes:
                alive_leafs.add(nid)

        # Everything else the tree considers a leaf is orphaned
        dead_leafs = self._tree.leafs - alive_leafs - {int(DEFAULT_ROOT_ID)}

        if not dead_leafs:
            return 0

        nodes_before = self._tree.data.number_of_nodes()
        self._tree.prune_dead_branches(
            dead_leafs=dead_leafs, alive_leafs=alive_leafs,
        )
        nodes_after = self._tree.data.number_of_nodes()
        return nodes_before - nodes_after

    # ------------------------------------------------------------------
    # Dashboard-compatible read properties
    # ------------------------------------------------------------------

    @property
    def iterations(self) -> list[int]:
        """Iteration indices (excludes the initial-state recording)."""
        return list(range(self._n_recorded - 1))

    @property
    def rewards_mean(self) -> list[float]:
        return [m["mean_reward"] for m in self._step_metadata[1:]]

    @property
    def rewards_max(self) -> list[float]:
        return [m["max_reward"] for m in self._step_metadata[1:]]

    @property
    def rewards_min(self) -> list[float]:
        return [m["min_reward"] for m in self._step_metadata[1:]]

    @property
    def alive_counts(self) -> list[int]:
        return [m["n_alive"] for m in self._step_metadata[1:]]

    @property
    def num_cloned(self) -> list[int]:
        return [m["num_cloned"] for m in self._step_metadata[1:]]

    @property
    def virtual_rewards_mean(self) -> list[float]:
        return [m["mean_virtual_reward"] for m in self._step_metadata[1:]]

    @property
    def virtual_rewards_max(self) -> list[float]:
        return [m["max_virtual_reward"] for m in self._step_metadata[1:]]

    @property
    def best_frames(self) -> list[np.ndarray | None]:
        return [m.get("best_frame") for m in self._step_metadata[1:]]

    @property
    def best_rewards(self) -> list[float]:
        return [m["max_reward"] for m in self._step_metadata[1:]]

    @property
    def best_indices(self) -> list[int]:
        return [m["best_walker_idx"] for m in self._step_metadata[1:]]

    @property
    def has_frames(self) -> bool:
        frames = self.best_frames
        return len(frames) > 0 and frames[0] is not None

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_atari_history(self) -> AtariHistory:
        """Convert to :class:`AtariHistory` for backward compatibility."""
        return AtariHistory(
            iterations=self.iterations,
            rewards_mean=self.rewards_mean,
            rewards_max=self.rewards_max,
            rewards_min=self.rewards_min,
            alive_counts=self.alive_counts,
            num_cloned=self.num_cloned,
            virtual_rewards_mean=self.virtual_rewards_mean,
            virtual_rewards_max=self.virtual_rewards_max,
            best_frames=self.best_frames,
            best_rewards=self.best_rewards,
            best_indices=self.best_indices,
            N=self.N,
            max_iterations=len(self.iterations),
            game_name=self.game_name,
        )

    def to_robotic_history(self) -> RoboticHistory:
        """Convert to :class:`RoboticHistory` for the robots dashboard."""
        return RoboticHistory(
            iterations=self.iterations,
            rewards_mean=self.rewards_mean,
            rewards_max=self.rewards_max,
            rewards_min=self.rewards_min,
            alive_counts=self.alive_counts,
            num_cloned=self.num_cloned,
            virtual_rewards_mean=self.virtual_rewards_mean,
            virtual_rewards_max=self.virtual_rewards_max,
            best_frames=self.best_frames,
            best_rewards=self.best_rewards,
            best_indices=self.best_indices,
            N=self.N,
            max_iterations=len(self.iterations),
            task_name=self.game_name,
        )

    def get_walker_branch(self, walker_idx: int) -> list[int]:
        """Trace a walker's lineage through clone events.

        Returns the path of node IDs from the root to the walker's current
        leaf node, following cloning edges back through the graph.
        """
        if self._n_recorded == 0:
            return []
        leaf_nid = self._nid(self._n_recorded - 1, walker_idx)
        return self._tree.get_branch(leaf_nid)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save to disk using torch.save."""
        # Strip numpy frames from metadata for pickling safety — keep as-is
        data = {
            "N": self.N,
            "game_name": self.game_name,
            "max_iterations": self.max_iterations,
            "n_recorded": self._n_recorded,
            "step_metadata": self._step_metadata,
            "graph_nodes": dict(self._tree.data.nodes(data=True)),
            "graph_edges": list(self._tree.data.edges()),
        }
        torch.save(data, path)

    @classmethod
    def load(cls, path: str) -> AtariTreeHistory:
        """Load from disk."""
        data = torch.load(path, weights_only=False)
        hist = cls(
            N=data["N"],
            game_name=data["game_name"],
            max_iterations=data["max_iterations"],
        )
        hist._n_recorded = data["n_recorded"]
        hist._step_metadata = data["step_metadata"]
        # Restore graph
        hist._tree.data.clear()
        for nid, attrs in data["graph_nodes"].items():
            hist._tree.data.add_node(nid, **attrs)
        for u, v in data["graph_edges"]:
            hist._tree.data.add_edge(u, v)
        return hist
