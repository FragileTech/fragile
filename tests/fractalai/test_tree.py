"""Tests for DataTree / NetworkxTree with mock data and recursive pruning."""

import numpy as np
import pytest

from fragile.fractalai.core.tree import (
    DataTree,
    DEFAULT_FIRST_NODE_ID,
    DEFAULT_ROOT_ID,
    NetworkxTree,
)


skip_ids = {DEFAULT_ROOT_ID, DEFAULT_FIRST_NODE_ID}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UID = np.uint64


def _make_node_data(obs_val: float = 0.0, reward: float = 0.0) -> dict:
    """Create minimal node data matching DEFAULT_NODE_DATA keys."""
    return {
        "observs": np.array([obs_val]),
        "rewards": np.array([reward]),
        "oobs": np.array([False]),
        "scores": np.array([0.0]),
    }


def _make_edge_data(action_val: float = 0.0) -> dict:
    return {"actions": np.array([action_val])}


def _build_chain(tree: NetworkxTree, parent: int, ids: list[int], epoch: int = 0):
    """Append a linear chain of nodes: parent -> ids[0] -> ids[1] -> ..."""
    prev = _UID(parent)
    for nid in ids:
        nid = _UID(nid)
        tree.append_leaf(
            leaf_id=nid,
            parent_id=prev,
            node_data=_make_node_data(obs_val=float(nid)),
            edge_data=_make_edge_data(action_val=float(nid)),
            epoch=epoch,
        )
        prev = nid


def _new_tree(prune: bool = False, names=None) -> DataTree:
    """Create a fresh DataTree with standard node/edge names."""
    if names is None:
        names = ["observs", "rewards", "oobs", "scores", "actions"]
    tree = DataTree(
        names=names,
        prune=prune,
        node_names=NetworkxTree.DEFAULT_NODE_DATA,
        edge_names=NetworkxTree.DEFAULT_EDGE_DATA,
    )
    # Manually reset with root node data so append_leaf can find the root
    tree.reset_graph(root_id=DEFAULT_ROOT_ID, node_data=_make_node_data(), epoch=-1)
    return tree


# ===================================================================
# Basic tree construction
# ===================================================================


class TestTreeConstruction:
    """Test append_leaf, node counting, and leaf tracking."""

    def test_empty_tree_has_root(self):
        tree = _new_tree()
        assert len(tree) == 1
        assert tree.root_id in tree.leafs

    def test_append_single_leaf(self):
        tree = _new_tree()
        tree.append_leaf(
            leaf_id=_UID(10),
            parent_id=DEFAULT_ROOT_ID,
            node_data=_make_node_data(1.0),
            edge_data=_make_edge_data(1.0),
        )
        assert len(tree) == 2
        assert _UID(10) in tree.leafs
        # Root is no longer a leaf
        assert DEFAULT_ROOT_ID not in tree.leafs

    def test_append_chain(self):
        tree = _new_tree()
        _build_chain(tree, int(DEFAULT_ROOT_ID), [10, 20, 30])
        assert len(tree) == 4  # root + 3 nodes
        # Only the tip is a leaf
        assert tree.leafs == {_UID(30)}

    def test_append_branching(self):
        """Build a tree with two branches from the root."""
        tree = _new_tree()
        _build_chain(tree, int(DEFAULT_ROOT_ID), [10, 20])
        _build_chain(tree, int(DEFAULT_ROOT_ID), [11, 21])
        assert len(tree) == 5
        assert tree.leafs == {_UID(20), _UID(21)}

    def test_duplicate_leaf_not_added(self):
        """Appending a node that already exists should be a no-op."""
        tree = _new_tree()
        _build_chain(tree, int(DEFAULT_ROOT_ID), [10])
        count_before = len(tree)
        tree.append_leaf(
            leaf_id=_UID(10),
            parent_id=DEFAULT_ROOT_ID,
            node_data=_make_node_data(),
            edge_data=_make_edge_data(),
        )
        assert len(tree) == count_before

    def test_append_missing_parent_raises(self):
        tree = _new_tree()
        with pytest.raises(ValueError, match="Parent"):
            tree.append_leaf(
                leaf_id=_UID(99),
                parent_id=_UID(999),
                node_data=_make_node_data(),
                edge_data=_make_edge_data(),
            )


# ===================================================================
# Path / branch retrieval
# ===================================================================


class TestPathRetrieval:
    def test_get_parent(self):
        tree = _new_tree()
        _build_chain(tree, int(DEFAULT_ROOT_ID), [10, 20])
        assert tree.get_parent(_UID(20)) == _UID(10)
        assert tree.get_parent(_UID(10)) == DEFAULT_ROOT_ID

    def test_get_branch(self):
        tree = _new_tree()
        _build_chain(tree, int(DEFAULT_ROOT_ID), [10, 20, 30])
        branch = tree.get_branch(_UID(30))
        assert branch == [DEFAULT_ROOT_ID, _UID(10), _UID(20), _UID(30)]

    def test_get_leaf_nodes(self):
        tree = _new_tree()
        _build_chain(tree, int(DEFAULT_ROOT_ID), [10, 20])
        _build_chain(tree, int(DEFAULT_ROOT_ID), [11])
        leaves = set(tree.get_leaf_nodes())
        assert leaves == {_UID(20), _UID(11)}


# ===================================================================
# Pruning
# ===================================================================


class TestPruning:
    """Test recursive branch pruning."""

    def _make_forked_tree(self, prune: bool = True):
        r"""Build a tree shaped like:

              root
             /    \
            10     11
           / \      \
          20  21     22
          |
          30
        """
        tree = _new_tree(prune=prune)
        _build_chain(tree, int(DEFAULT_ROOT_ID), [10, 20, 30])
        tree.append_leaf(
            leaf_id=_UID(21),
            parent_id=_UID(10),
            node_data=_make_node_data(),
            edge_data=_make_edge_data(),
        )
        _build_chain(tree, int(DEFAULT_ROOT_ID), [11, 22])
        assert len(tree) == 7
        assert tree.leafs == {_UID(30), _UID(21), _UID(22)}
        return tree

    def test_prune_single_dead_leaf(self):
        """Pruning a single dead leaf removes it and recurses up if parent becomes orphan."""
        tree = self._make_forked_tree()
        # Kill leaf 21 — its parent (10) still has child 20, so only 21 is removed
        tree.frozen_node_ids = set()
        tree.prune_branch(_UID(21), alive_nodes={_UID(30), _UID(22)})
        assert _UID(21) not in tree.data.nodes
        assert _UID(10) in tree.data.nodes  # Still has child 20->30
        assert len(tree) == 6

    def test_prune_dead_branch_recursively(self):
        """Pruning a dead leaf whose ancestors have no other children removes the whole branch."""
        tree = self._make_forked_tree()
        tree.frozen_node_ids = set()
        # Kill branch 11->22: leaf 22 is dead, and after removal 11 has no children
        tree.prune_branch(_UID(22), alive_nodes={_UID(30), _UID(21)})
        assert _UID(22) not in tree.data.nodes
        assert _UID(11) not in tree.data.nodes  # Recursively pruned
        assert len(tree) == 5  # root, 10, 20, 21, 30

    def test_prune_stops_at_alive_parent(self):
        """Pruning won't remove a leaf whose parent is in alive_nodes."""
        tree = self._make_forked_tree()
        tree.frozen_node_ids = set()
        # Leaf 30's parent is 20. When 20 is alive the pruner refuses to
        # remove 30 (the branch is still "alive" from the parent's perspective).
        tree.prune_branch(_UID(30), alive_nodes={_UID(20), _UID(21), _UID(22)})
        assert _UID(30) in tree.data.nodes  # NOT removed — parent alive
        assert _UID(20) in tree.data.nodes

    def test_prune_leaf_with_dead_parent_removes_both(self):
        """When a leaf and its parent are both dead, both get pruned."""
        tree = self._make_forked_tree()
        tree.frozen_node_ids = set()
        # Kill leaf 22. Parent 11 has no other children and is not alive.
        # After removing 22, 11 becomes a childless non-alive node → also pruned.
        tree.prune_branch(_UID(22), alive_nodes={_UID(30), _UID(21)})
        assert _UID(22) not in tree.data.nodes
        assert _UID(11) not in tree.data.nodes
        # Root gained 11's slot back as a potential leaf? No — root still has child 10.
        assert DEFAULT_ROOT_ID not in tree.leafs

    def test_prune_stops_at_sentinel(self):
        """Pruning should stop at the prune sentinel (root by default)."""
        tree = _new_tree(prune=True)
        _build_chain(tree, int(DEFAULT_ROOT_ID), [10, 20])
        tree.frozen_node_ids = set()
        # Kill both leaves recursively — should stop at root
        tree.prune_branch(_UID(20), alive_nodes=set())
        # 20 and 10 removed, root survives (it's the sentinel)
        assert DEFAULT_ROOT_ID in tree.data.nodes
        assert _UID(10) not in tree.data.nodes
        assert _UID(20) not in tree.data.nodes

    def test_prune_node_with_children_no_op(self):
        """Cannot prune a node that still has children."""
        tree = self._make_forked_tree()
        tree.frozen_node_ids = set()
        count_before = len(tree)
        tree.prune_branch(_UID(10), alive_nodes=set())  # 10 has children
        assert len(tree) == count_before

    def test_prune_tree_method(self):
        """The high-level prune_tree method removes dead branches."""
        tree = self._make_forked_tree()
        tree.frozen_node_ids = set()
        # Only leaf 30 is alive — branches ending in 21 and 22 should be pruned
        tree.prune_tree(alive_leafs={_UID(30)})
        assert _UID(21) not in tree.data.nodes
        assert _UID(22) not in tree.data.nodes
        assert _UID(11) not in tree.data.nodes  # Recursively removed
        assert _UID(30) in tree.data.nodes
        assert _UID(20) in tree.data.nodes  # Ancestor of alive leaf

    def test_prune_tree_disabled(self):
        """When prune=False, prune_tree should be a no-op."""
        tree = self._make_forked_tree(prune=False)
        tree.frozen_node_ids = set()
        count_before = len(tree)
        tree.prune_tree(alive_leafs={_UID(30)})
        assert len(tree) == count_before

    def test_frozen_nodes_not_pruned(self):
        """Frozen nodes should survive pruning even if they're dead leaves."""
        tree = self._make_forked_tree()
        tree.frozen_node_ids = {_UID(22)}
        tree.prune_tree(alive_leafs={_UID(30)})
        # 22 is frozen → kept, 11 is ancestor of frozen → kept
        assert _UID(22) in tree.data.nodes
        assert _UID(11) in tree.data.nodes
        # 21 is not frozen and not alive → pruned
        assert _UID(21) not in tree.data.nodes


# ===================================================================
# Reset
# ===================================================================


class TestReset:
    def test_reset_graph(self):
        tree = _new_tree()
        _build_chain(tree, int(DEFAULT_ROOT_ID), [10, 20, 30])
        assert len(tree) == 4

        tree.reset_graph(root_id=DEFAULT_ROOT_ID, node_data=_make_node_data(), epoch=0)
        assert len(tree) == 1
        assert tree.leafs == {DEFAULT_ROOT_ID}
        assert len(tree.frozen_node_ids) == 0


# ===================================================================
# DataTree data generators
# ===================================================================


class TestDataTreeGenerators:
    """Test iterate_branch, iterate_path_data, and iterate_nodes_at_random."""

    def _build_data_tree(self) -> DataTree:
        """Build: root -> 10 -> 20 -> 30 with distinct obs/action values."""
        tree = _new_tree()
        for nid, parent in [(10, int(DEFAULT_ROOT_ID)), (20, 10), (30, 20)]:
            tree.append_leaf(
                leaf_id=_UID(nid),
                parent_id=_UID(parent),
                node_data=_make_node_data(obs_val=float(nid), reward=float(nid) / 10),
                edge_data=_make_edge_data(action_val=float(nid) * 2),
            )
        return tree

    def test_iterate_branch_returns_all_steps(self):
        tree = self._build_data_tree()
        steps = list(tree.iterate_branch(_UID(30), names=["observs", "actions"]))
        # Path: root -> 10 -> 20 -> 30
        # Generator yields (node, edge) for root->10, 10->20, 20->30
        # But root and FIRST_NODE_ID are skipped, so we get transitions 10->20, 20->30
        assert len(steps) == 2
        # Each step is (observs_value, actions_value)
        obs_vals = [s[0].item() for s in steps]
        assert obs_vals == [10.0, 20.0]

    def test_iterate_branch_with_next_prefix(self):
        tree = self._build_data_tree()
        steps = list(tree.iterate_branch(_UID(30), names=["observs", "actions", "next_observs"]))
        assert len(steps) == 2
        # step 0: node=10, next_node=20
        assert steps[0][0].item() == 10.0  # observs of node 10
        assert steps[0][2].item() == 20.0  # next_observs = observs of node 20

    def test_iterate_nodes_at_random_visits_all(self):
        tree = self._build_data_tree()
        steps = list(tree.iterate_nodes_at_random(names=["observs"]))
        # 3 non-skip nodes (10, 20, 30 as children); the generator yields
        # the *parent* node's data for each child, so we see root(0), 10, 20.
        assert len(steps) == 3
        obs_set = {s[0].item() for s in steps}
        assert obs_set == {0.0, 10.0, 20.0}

    def test_validate_names_rejects_unknown(self):
        tree = self._build_data_tree()
        with pytest.raises(KeyError):
            list(tree.iterate_branch(_UID(30), names=["nonexistent"]))

    def test_iterate_branch_batched(self):
        tree = self._build_data_tree()
        batches = list(tree.iterate_branch(_UID(30), names=["observs"], batch_size=2))
        # 2 steps in one batch
        assert len(batches) == 1


# ===================================================================
# Compose
# ===================================================================


class TestCompose:
    def test_compose_merges_graphs(self):
        tree1 = _new_tree()
        _build_chain(tree1, int(DEFAULT_ROOT_ID), [10, 20])

        tree2 = _new_tree()
        _build_chain(tree2, int(DEFAULT_ROOT_ID), [11, 21])

        tree1.compose(tree2)
        assert _UID(10) in tree1.data.nodes
        assert _UID(11) in tree1.data.nodes
        assert _UID(20) in tree1.data.nodes
        assert _UID(21) in tree1.data.nodes


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_prune_empty_dead_set(self):
        """Pruning with no dead leaves is a no-op."""
        tree = _new_tree(prune=True)
        _build_chain(tree, int(DEFAULT_ROOT_ID), [10])
        tree.frozen_node_ids = set()
        count_before = len(tree)
        tree.prune_dead_branches(dead_leafs=set(), alive_leafs={_UID(10)})
        assert len(tree) == count_before

    def test_prune_all_leaves_dead(self):
        """If all leaves die, the tree is reduced to the root."""
        tree = _new_tree(prune=True)
        _build_chain(tree, int(DEFAULT_ROOT_ID), [10, 20])
        _build_chain(tree, int(DEFAULT_ROOT_ID), [11])
        tree.frozen_node_ids = set()
        tree.prune_dead_branches(
            dead_leafs={_UID(20), _UID(11)},
            alive_leafs=set(),
        )
        # Everything should be pruned down to root
        assert len(tree) == 1
        assert DEFAULT_ROOT_ID in tree.data.nodes

    def test_deep_recursive_prune(self):
        """A long chain with a single dead tip is fully removed."""
        tree = _new_tree(prune=True)
        chain = list(range(10, 110))  # 100 nodes deep
        _build_chain(tree, int(DEFAULT_ROOT_ID), chain)
        assert len(tree) == 101  # root + 100

        tree.frozen_node_ids = set()
        tree.prune_branch(_UID(chain[-1]), alive_nodes=set())
        # Entire chain pruned, only root remains
        assert len(tree) == 1

    def test_callable_returns_self(self):
        tree = _new_tree()
        assert tree() is tree
