"""AtariTreeHistory: graph-backed history recorder for Atari games.

Uses ``DataTree`` to store per-walker Atari-specific data with cloning-aware
parent edges.  When walker *w* clones from companion *c* at step *t*, the
parent edge points to ``(t-1, c)`` instead of ``(t-1, w)``, turning the flat
list of iterations into a proper tree reflecting cloning lineage.

Node IDs are deterministic:

    node_id = step * (N + 1) + walker + 1
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import numpy as np
import torch

from fragile.fractalai.core.tree import DataTree, DEFAULT_ROOT_ID
from fragile.fractalai.robots.robotic_history import RoboticHistory
from fragile.fractalai.videogames.atari_history import AtariHistory


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
    n_elite : int
        Number of top-reward walkers whose tree branches are protected
        from pruning.  Set to 0 to disable elite protection.
    """

    def __init__(self, N: int, game_name: str = "", max_iterations: int = 1000, n_elite: int = 0):
        self.N = N
        self.game_name = game_name
        self.max_iterations = max_iterations
        self.n_elite = n_elite
        self._elite_node_ids: set[int] = set()

        self._tree = DataTree(
            names=[],
            node_names=[],
            edge_names=[],
        )

        self._n_recorded: int = 0
        self._step_metadata: list[dict] = []

        # Track the all-time best walker for replay
        self._global_best_reward: float = -float("inf")
        self._global_best_step: int = 0
        self._global_best_walker_idx: int = 0

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
        Each node stores the walker's ``env_state`` for later frame rendering.
        """
        step = 0
        for w in range(self.N):
            nid = self._nid(step, w)
            action_w = state.actions[w]
            action_val = int(action_w) if np.ndim(action_w) == 0 else action_w.tolist()
            env_state = state.states[w]
            if hasattr(env_state, "copy"):
                env_state = env_state.copy()

            # Extract rgb_frame from env state if available
            frame = None
            if hasattr(env_state, "rgb_frame") and env_state.rgb_frame is not None:
                frame = env_state.rgb_frame.copy()

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
                "env_state": env_state,
                "frame": frame,
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
            "min_virtual_reward": 0.0,
            "mean_dt": 1.0,
            "min_dt": 1,
            "max_dt": 1,
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

        rewards = state_final.rewards
        best_idx = int(rewards.argmax().item())

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

            # Store env_state for potential re-rendering
            env_state = state_final.states[w]
            if hasattr(env_state, "copy"):
                env_state = env_state.copy()

            # Extract rgb_frame directly from the walker's AtariState
            frame = None
            if hasattr(env_state, "rgb_frame") and env_state.rgb_frame is not None:
                frame = env_state.rgb_frame.copy()
            elif w == best_idx and best_frame is not None:
                frame = best_frame

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
                "env_state": env_state,
                "frame": frame,
            }

            self._tree.append_leaf(
                leaf_id=nid,
                parent_id=parent_id,
                node_data=node_data,
                edge_data={},
                epoch=step,
            )

        # Track elite walker node IDs (protect from pruning).
        # Replace each step so only the current top-k are protected.
        if self.n_elite > 0:
            k = min(self.n_elite, self.N)
            _, elite_indices = rewards.topk(k)
            self._elite_node_ids = {self._nid(step, int(ei.item())) for ei in elite_indices}

        # Update global best walker tracking
        best_reward = float(rewards[best_idx].item())
        if best_reward > self._global_best_reward:
            self._global_best_reward = best_reward
            self._global_best_step = step
            self._global_best_walker_idx = best_idx

        # Step metadata

        vr_mean = 0.0
        vr_max = 0.0
        if state_final.virtual_rewards is not None:
            vr_mean = float(state_final.virtual_rewards.mean().item())
            vr_max = float(state_final.virtual_rewards.max().item())

        vr_min = 0.0
        if state_final.virtual_rewards is not None:
            vr_min = float(state_final.virtual_rewards.min().item())

        self._step_metadata.append({
            "n_alive": int(state_final.alive.sum().item()),
            "num_cloned": int(will_clone.sum().item()),
            "mean_reward": float(rewards.mean().item()),
            "max_reward": float(rewards.max().item()),
            "min_reward": float(rewards.min().item()),
            "std_reward": float(rewards.std().item()),
            "mean_virtual_reward": vr_mean,
            "max_virtual_reward": vr_max,
            "min_virtual_reward": vr_min,
            "mean_dt": float(info.get("mean_dt", 1.0)),
            "min_dt": int(info.get("min_dt", 1)),
            "max_dt": int(info.get("max_dt", 1)),
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

        # Protect elite walker branches from pruning
        alive_leafs |= {nid for nid in self._elite_node_ids if nid in self._tree.data.nodes}

        # Everything else the tree considers a leaf is orphaned
        dead_leafs = self._tree.leafs - alive_leafs - {int(DEFAULT_ROOT_ID)}

        if not dead_leafs:
            return 0

        nodes_before = self._tree.data.number_of_nodes()
        self._tree.prune_dead_branches(
            dead_leafs=dead_leafs,
            alive_leafs=alive_leafs,
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
    def rewards_std(self) -> list[float]:
        return [m.get("std_reward", 0.0) for m in self._step_metadata[1:]]

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
    def virtual_rewards_min(self) -> list[float]:
        return [m.get("min_virtual_reward", 0.0) for m in self._step_metadata[1:]]

    @property
    def dt_mean(self) -> list[float]:
        return [m.get("mean_dt", 1.0) for m in self._step_metadata[1:]]

    @property
    def dt_min(self) -> list[int]:
        return [m.get("min_dt", 1) for m in self._step_metadata[1:]]

    @property
    def dt_max(self) -> list[int]:
        return [m.get("max_dt", 1) for m in self._step_metadata[1:]]

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
        """Convert to :class:`AtariHistory` for backward compatibility.

        Uses path-based frames (root→best-leaf) when available, falling
        back to per-iteration metadata frames.
        """
        path_frames = self.get_best_path_frames()
        # Use path frames if they exist and have at least one non-None entry
        if path_frames and any(f is not None for f in path_frames):
            frames = path_frames
        else:
            frames = self.best_frames
        return AtariHistory(
            iterations=self.iterations,
            rewards_mean=self.rewards_mean,
            rewards_max=self.rewards_max,
            rewards_min=self.rewards_min,
            rewards_std=self.rewards_std,
            alive_counts=self.alive_counts,
            num_cloned=self.num_cloned,
            virtual_rewards_mean=self.virtual_rewards_mean,
            virtual_rewards_max=self.virtual_rewards_max,
            virtual_rewards_min=self.virtual_rewards_min,
            dt_mean=self.dt_mean,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            best_frames=frames,
            best_rewards=self.best_rewards,
            best_indices=self.best_indices,
            N=self.N,
            max_iterations=len(self.iterations),
            game_name=self.game_name,
        )

    def to_robotic_history(self) -> RoboticHistory:
        """Convert to :class:`RoboticHistory` for the robots dashboard.

        Uses path-based frames (root→best-leaf) when available, falling
        back to per-iteration metadata frames.
        """
        path_frames = self.get_best_path_frames()
        if path_frames and any(f is not None for f in path_frames):
            frames = path_frames
        else:
            frames = self.best_frames
        return RoboticHistory(
            iterations=self.iterations,
            rewards_mean=self.rewards_mean,
            rewards_max=self.rewards_max,
            rewards_min=self.rewards_min,
            alive_counts=self.alive_counts,
            num_cloned=self.num_cloned,
            virtual_rewards_mean=self.virtual_rewards_mean,
            virtual_rewards_max=self.virtual_rewards_max,
            virtual_rewards_min=self.virtual_rewards_min,
            dt_mean=self.dt_mean,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            best_frames=frames,
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

    def get_best_walker_branch(self) -> list[int]:
        """Return the branch (root → leaf) of the all-time best-reward walker.

        Uses the walker that achieved the highest reward at any point during
        the simulation, not just the final step.  The replay may be shorter
        than the full simulation when the peak occurs before the last step.

        Falls back to the final-step best walker if the global best node was
        pruned (e.g. when ``n_elite=0``).

        Returns an empty list if nothing has been recorded.
        """
        if self._n_recorded <= 1:
            return []
        leaf_nid = self._nid(self._global_best_step, self._global_best_walker_idx)
        if leaf_nid in self._tree.data.nodes:
            return self._tree.get_branch(leaf_nid)
        # Global best was pruned — fall back to final-step best
        last_meta = self._step_metadata[-1]
        best_idx = last_meta["best_walker_idx"]
        return self.get_walker_branch(best_idx)

    def get_best_path_frames(self) -> list[np.ndarray | None]:
        """Collect frames along the best walker's branch.

        Returns a list of frames (or ``None`` where no frame was stored)
        ordered from root to leaf.  The root sentinel node is excluded.
        """
        branch = self.get_best_walker_branch()
        if not branch:
            return []
        nodes = self._tree.data.nodes
        # Skip root node (branch[0] is DEFAULT_ROOT_ID)
        return [nodes[nid].get("frame") for nid in branch[1:]]

    def render_missing_path_frames(
        self,
        render_fn: Callable[[object], np.ndarray],
    ) -> list[np.ndarray | None]:
        """Fill in missing frames along the best walker's branch.

        For each node on the best path that has ``frame=None`` but has an
        ``env_state``, call ``render_fn(env_state)`` and store the result
        back in the tree node.

        Parameters
        ----------
        render_fn : callable
            ``(env_state) -> np.ndarray`` — renders an RGB frame from a
            serialised environment state.  Typically
            ``gas._render_walker_frame``.

        Returns
        -------
        list[np.ndarray | None]
            Ordered frames along the best branch (root excluded).
        """
        branch = self.get_best_walker_branch()
        if not branch:
            return []

        nodes = self._tree.data.nodes
        frames: list[np.ndarray | None] = []
        for nid in branch[1:]:  # skip root
            node = nodes[nid]
            frame = node.get("frame")
            if frame is None and node.get("env_state") is not None:
                env_st = node["env_state"]
                if hasattr(env_st, "rgb_frame") and env_st.rgb_frame is not None:
                    frame = env_st.rgb_frame
                elif render_fn is not None:
                    frame = render_fn(env_st)
                node["frame"] = frame
            frames.append(frame)
        return frames

    def get_path_frames_for_node(self, leaf_node_id: int) -> list[np.ndarray | None]:
        """Collect frames along a specific branch from root to *leaf_node_id*.

        Returns an empty list when the node does not exist in the tree.
        """
        if leaf_node_id not in self._tree.data.nodes:
            return []
        branch = self._tree.get_branch(leaf_node_id)
        nodes = self._tree.data.nodes
        return [nodes[nid].get("frame") for nid in branch[1:]]  # skip root

    def get_elite_branches_info(self) -> list[dict]:
        """Return metadata for distinct elite walker trajectories.

        Walks through all elite node IDs (sorted by reward descending),
        deduplicates overlapping trajectories (keeps the highest-reward
        endpoint for each unique branch), and returns info dicts.
        """
        if not self._elite_node_ids:
            return []

        nodes = self._tree.data.nodes

        # Gather (node_id, reward) for every elite node still in the tree
        elite_entries: list[tuple[int, float]] = []
        for nid in self._elite_node_ids:
            if nid in nodes:
                reward = nodes[nid].get("reward", 0.0)
                elite_entries.append((nid, reward))

        if not elite_entries:
            return []

        # Sort by reward descending so the best trajectory is first
        elite_entries.sort(key=lambda x: x[1], reverse=True)

        claimed: set[int] = set()
        results: list[dict] = []

        for nid, reward in elite_entries:
            # Skip if this leaf is already an intermediate node of a
            # higher-reward branch (same replay but shorter).
            if nid in claimed:
                continue
            branch = self._tree.get_branch(nid)
            claimed |= set(branch)

            # Recover step and walker index from the node id
            step = (nid - 1) // (self.N + 1)
            walker_idx = (nid - 1) % (self.N + 1)

            results.append({
                "node_id": nid,
                "step": step,
                "walker_idx": walker_idx,
                "reward": reward,
                "label": f"#{len(results) + 1} (R={reward:.0f}, step {step})",
            })

        return results

    def get_elite_reward_curves(self) -> list[dict]:
        """Return per-step cumulative reward for each distinct elite branch.

        Each entry is a dict with ``'label'``, ``'steps'`` (list[int]),
        and ``'rewards'`` (list[float]).
        """
        infos = self.get_elite_branches_info()
        nodes = self._tree.data.nodes
        curves: list[dict] = []
        for info in infos:
            branch = self._tree.get_branch(info["node_id"])
            steps: list[int] = []
            rewards: list[float] = []
            for nid in branch[1:]:  # skip root
                step = (nid - 1) // (self.N + 1)
                reward = nodes[nid].get("reward", 0.0)
                steps.append(step)
                rewards.append(reward)
            curves.append({
                "label": info["label"],
                "steps": steps,
                "rewards": rewards,
            })
        return curves

    def render_elite_path_frames(
        self,
        render_fn: Callable[[object], np.ndarray],
    ) -> None:
        """Fill missing frames along all elite branches.

        Same logic as :meth:`render_missing_path_frames` but applied to
        every distinct elite branch instead of just the global best.
        """
        for info in self.get_elite_branches_info():
            branch = self._tree.get_branch(info["node_id"])
            nodes = self._tree.data.nodes
            for nid in branch[1:]:  # skip root
                node = nodes[nid]
                frame = node.get("frame")
                if frame is None and node.get("env_state") is not None:
                    env_st = node["env_state"]
                    if hasattr(env_st, "rgb_frame") and env_st.rgb_frame is not None:
                        frame = env_st.rgb_frame
                    elif render_fn is not None:
                        frame = render_fn(env_st)
                    node["frame"] = frame

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
            "n_elite": self.n_elite,
            "n_recorded": self._n_recorded,
            "step_metadata": self._step_metadata,
            "graph_nodes": dict(self._tree.data.nodes(data=True)),
            "graph_edges": list(self._tree.data.edges()),
            "elite_node_ids": self._elite_node_ids,
            "global_best_reward": self._global_best_reward,
            "global_best_step": self._global_best_step,
            "global_best_walker_idx": self._global_best_walker_idx,
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
            n_elite=data.get("n_elite", 0),
        )
        hist._n_recorded = data["n_recorded"]
        hist._step_metadata = data["step_metadata"]
        hist._elite_node_ids = data.get("elite_node_ids", set())
        hist._global_best_reward = data.get("global_best_reward", -float("inf"))
        hist._global_best_step = data.get("global_best_step", 0)
        hist._global_best_walker_idx = data.get("global_best_walker_idx", 0)
        # Restore graph
        hist._tree.data.clear()
        for nid, attrs in data["graph_nodes"].items():
            hist._tree.data.add_node(nid, **attrs)
        for u, v in data["graph_edges"]:
            hist._tree.data.add_edge(u, v)
        return hist
