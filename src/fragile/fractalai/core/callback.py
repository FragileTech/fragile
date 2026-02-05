"""Data structure to keep track of the graph that stores all the states visited by a Swarm."""

from collections.abc import Iterable
import copy
from pathlib import Path
import pickle
from typing import Any

import networkx as nx
import numpy
import torch

from fragile.core.config import NumpyArray
from fragile.core.loop import CallbackLoop
from fragile.core.module import FragileModule
from fragile.core.tree import DataTree, DEFAULT_FIRST_NODE_ID, DEFAULT_ROOT_ID, NetworkxTree
from fragile.typing import (
    BooleanValue,
    DictValues,
    DictWalker,
    HASH_DTYPE,
    InputsConfig,
    InputValues,
    NamesData,
    NodeDataGenerator,
    NodeId,
    OutputValues,
    Value,
)


class Callback(FragileModule, CallbackLoop):
    name = "callback"

    def __init__(self, minimize: bool = False, **kwargs):
        self._minimize = minimize
        super().__init__(**kwargs)

    @property
    def minimize(self) -> bool:
        return self.parent.minimize if self.is_linked else self._minimize

    def forward(self, **inputs: InputValues) -> OutputValues:
        pass


class TrackWalkersId(Callback):
    name = "track_walkers_id"
    inputs_ = {"id_walkers": {"clone": True}, "parent_ids": {"clone": True}}
    config_ = {
        "id_walkers": NumpyArray(dtype=HASH_DTYPE, default=DEFAULT_FIRST_NODE_ID),
        "parent_ids": NumpyArray(dtype=HASH_DTYPE, default=DEFAULT_ROOT_ID),
    }
    outputs_ = ("id_walkers", "parent_ids")

    def forward(
        self,
        id_walkers: numpy.ndarray = None,
        parent_ids: numpy.ndarray = None,
        **_,
    ) -> OutputValues:
        name = "states" if "states" in self.parent.names else "observs"
        new_ids = self.parent.hash_batch(name)
        id_walkers = self.get("id_walkers") if id_walkers is None else id_walkers
        return {
            "parent_ids": self.copy(id_walkers=id_walkers),
            "id_walkers": new_ids,
        }

    def on_env_end(self):
        self.step()


class HistoryTree(TrackWalkersId):
    """Tree data structure that keeps track of the visited states.

    It allows to save the :class:`Swarm` data after every iteration, and provides methods to \
    recover the sampled data after the algorithm run.

    The data that will be stored in the graph it's defined in the ``names`` parameter.
    For example:

     - If names is ``["observs", "actions"]``, the observations of every visited \
       state will be stored as node attributes, and actions will be stored as edge attributes.

     - If names is ``["observs", "actions", "next_observs"]`` the same data will be stored,
       but when the data generator methods are called the observation corresponding \
       to the next state will also be returned.

     The attribute ``names`` also defines the order of the data returned by the generator.

     As long as the data is stored in the graph (passing a valid ``names`` list at \
     initialization), the order in which we sample the data can be redefined passing \
     the ``names`` parameter to the generator method.

     For example, if the ``names`` passed at initialization is ``["states", "rewards"]``, \
     you can call the generator methods with ``names=["rewards", "states", "next_states"]`` \
     and the returned data will be a tuple containing (rewards, states, next_states).

    """

    name = "tree"

    def __init__(
        self,
        names: NamesData = None,
        prune: bool = False,
        root_id: NodeId = DEFAULT_ROOT_ID,
        node_names: NamesData = None,
        edge_names: NamesData = None,
        next_prefix: str = NetworkxTree.DEFAULT_NEXT_PREFIX,
        file_name: str | Path = "tree.pkl",
        save_on_end: bool = False,
        **kwargs,
    ):
        self.track_names = names
        self.prune = prune
        self.root_id = root_id
        self.node_names = node_names
        self.edge_names = edge_names or ()
        self.next_prefix = next_prefix
        self.save_on_end = save_on_end
        self.file_name = Path(file_name)
        self._tree = None
        super().__init__(**kwargs)

    @property
    def graph(self) -> nx.Graph:
        return self._tree.data

    @property
    def data_tree(self) -> DataTree | None:
        return self._tree

    def __getattr__(self, item):
        try:
            getattr(self._tree, item)
        except AttributeError:
            return super().__getattribute__(item)

    def __repr__(self):
        return (
            f"<strong>{self.__class__.__name__}</strong>: "
            f"Nodes {len(self._tree)} Leafs {len(self._tree.leafs)}"
        )

    def to_html(self):
        tree = self._tree
        return (
            f"<strong>{self.__class__.__name__}</strong>: "
            f"Nodes {len(tree)} Leafs {len(tree.leafs)}"
        )

    def build_inputs(self, inputs: InputsConfig | None = None) -> InputsConfig:
        inputs = super().build_inputs(inputs)
        if self.track_names is None:
            return inputs
        return {n: {**inputs.get(n, {}), "clone": True} for n in self.track_names}

    def setup_parent(self, parent):
        super().setup_parent(parent)
        if self.node_names is None:
            node_names = tuple(set(self.parent.config_names) - set(self.edge_names))
            if hasattr(self.parent, "walkers") and self.parent.walkers is not None:
                node_names = tuple(set(node_names) - set(self.parent.walkers.config_names))
            self.node_names = node_names

        if self.track_names is None:
            self.track_names = tuple(set(self.node_names + self.edge_names))
        track_names = tuple(n for n in self.track_names if n in parent.config_names)
        node_names = tuple(n for n in self.node_names if n in parent.config_names)
        edge_names = tuple(n for n in self.edge_names if n in parent.config_names)
        self._tree = DataTree(
            names=track_names,
            prune=self.prune,
            root_id=self.root_id,
            node_names=node_names,
            edge_names=edge_names,
            next_prefix=self.next_prefix,
        )

    def warmup(self, n_walkers: int, state: DictWalker | DictValues) -> OutputValues:
        if self._tree is not None and self.parent is not None and self.parent.state:
            self._tree.reset(module=self.parent)
        return super().warmup(n_walkers=n_walkers, state=state)
        # self.update_tree(**data)

    def update_tree(self, id_walkers=None, parent_ids=None, **_):
        id_walkers = self.get("id_walkers") if id_walkers is None else id_walkers
        parent_ids = self.get("parent_ids") if parent_ids is None else parent_ids
        if id_walkers is None or parent_ids is None:
            msg = "id_walkers and parent_ids must be defined to update the tree."
            raise ValueError(msg)
        self._tree.update(
            parent_ids=parent_ids,
            node_ids=id_walkers,
            n_iter=(self.parent.epoch if hasattr(self.parent, "epoch") else 0),
            module=self.parent,
        )

    def on_reset_end(self) -> None:
        self.update_tree()

    def on_env_update_end(self):
        """Update the tree with the data from the last iteration of the Swarm."""
        self.update_tree()

    def on_walkers_update_end(self):
        """Prune the tree to remove the branches that are no longer being expanded."""
        self._tree.prune_tree(alive_leafs=self.get("id_walkers"))

    def on_run_end(self) -> None:
        """Save the tree to a file."""
        if self.save_on_end:
            self.save_tree(self.file_name)

    def iterate_root_path(
        self,
        batch_size: int | None = None,
        names: NamesData = None,
    ) -> NodeDataGenerator:
        """Return a generator that yields the data of the nodes contained in the best path.

        Args:
            batch_size: If it is not None, the generator will return batches of \
                        data with the target batch size. If Batch size is less than 1 \
                        it will return a single batch containing all the data.
            names: Names of the data attributes that will be yielded by the generator.

        Returns:
            Generator providing the data corresponding to a branch of the internal tree.

        """
        return self._tree.iterate_branch(
            node_id=self.parent.root.id_walkers,
            batch_size=batch_size,
            names=names,
        )

    def get_root_graph(self) -> nx.DiGraph:
        """Return a copy of the graph containing only the nodes and data of the root path."""
        root_path_nodes = self.data_tree.get_path_node_ids(self.parent.root.id_walkers)
        return copy.deepcopy(self.graph.subgraph(root_path_nodes))

    def save_tree(self, path: str | Path | None = None) -> None:
        """Save the graph to a file."""
        path = Path(path) if path is not None else self.file_name
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        if not (str(path).endswith(".pkl") or str(path).endswith(".pck")):
            path = Path(f"{path}.pkl")
        with path.open("wb") as f:
            pickle.dump(self.data_tree, f)


class RootWalker(Callback):
    name = "root"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._root_state = {}

    @property
    def n_walkers(self) -> int:
        return 1

    @property
    def data(self) -> DictWalker:
        return self._root_state

    def __getattr__(self, item):
        if item + "s" in self.data.keys():
            item += "s"
        if item in self.data.keys():
            value = self.data[item][0]
            if isinstance(value, torch.Tensor) and len(value.shape) == 0:
                return value.item()
            return value
        return self.__getattribute__(item)

    def __repr__(self) -> str:
        # score = self.data.get('scores', [numpy.nan])[0]
        return f"{self.__class__.__name__}: score: {self.data.get('scores', [numpy.nan])[0]}"

    def get(
        self,
        names: str | Iterable[str] | None = None,
        default: Any = None,
        index: Any = None,
        as_dict: bool = False,
    ) -> DictValues | Value:
        if self.is_linked:
            return self.parent.get(names=names, default=default, index=index, as_dict=as_dict)
        return super().get(names=names, default=default, index=index, as_dict=as_dict)

    def warmup(self, **kwargs):
        self._root_state = {}
        return super().warmup(**kwargs)

    def export_root(self, index: int) -> None:
        self.import_walker(self.data, index=index)

    def import_root(self, index: int = 0):
        self._root_state = self.copy(self.get(index=index))

    def on_env_update_end(self) -> None:
        self.import_root()


class BestWalker(RootWalker):
    track_name_ = "scores"
    inputs_ = {
        track_name_: {},
    }

    def __init__(
        self,
        fix_best: bool = True,
        always_update: bool = False,
        track_name: str = track_name_,
        **kwargs,
    ):
        self.always_update = always_update
        self.fix_best = fix_best
        self.track_name = track_name
        super().__init__(**kwargs)

    def to_html(self):
        scores = numpy.nan
        try:
            scores = self.data[self.track_name]
        except KeyError:
            pass
        return (
            f"<strong>{self.__class__.__name__}</strong>: Score: {scores}\n"
            # f"Score: {self.data.get('scores', [numpy.nan])[0]}\n"
        )

    def build_inputs(self, inputs: InputsConfig | None = None) -> InputsConfig:
        inputs_ = copy.deepcopy(super().build_inputs(inputs))
        if self.track_name != self.track_name_ and self.track_name_ in inputs_:
            inputs_[self.track_name] = inputs_.pop(self.track_name_)
        return inputs_

    def get_valid_walkers(self):
        oobs = self.get("oobs")
        terminals = self.get("terminals")
        oobs = (
            torch.zeros(self.n_walkers, dtype=torch.bool, device=self.device)
            if oobs is None
            else oobs
        )
        terminals = (
            torch.zeros(self.n_walkers, dtype=torch.bool, device=self.device)
            if terminals is None
            else terminals
        )
        return torch.logical_not(torch.logical_or(oobs, terminals))

    def get_best_index(self, valid_walkers: BooleanValue | None = None) -> torch.int64:
        scores: torch.Tensor = self.get(self.track_name)
        if scores is None:
            msg = "Scores must be defined to get the best index."
            raise ValueError(msg)
        valid_walkers = self.get_valid_walkers() if valid_walkers is None else valid_walkers
        if valid_walkers.any():
            index = torch.arange(len(scores)).to(device=valid_walkers.device)
            alives_ix = index[valid_walkers]
            target = scores[valid_walkers]
            sub_ix = torch.argmin(target) if self.minimize else torch.argmax(target)
            return index[alives_ix][sub_ix]
        return None

    def import_root(self, index: int = 0):
        valid_walkers = self.get_valid_walkers()
        if not valid_walkers.any():
            return None
        best_ix = self.get_best_index(valid_walkers)
        if best_ix is None:  # If there is no valid walker, we do not update the root
            return None
        if not self.data or self.always_update or self.parent.epoch == 0:
            return super().import_root(index=best_ix)

        current_best = self.get(self.track_name, index=best_ix)[0]
        root_score = (
            self.data[self.track_name][0]
            if self.data
            else (numpy.inf if self.minimize else -numpy.inf)
        )
        score_improves = current_best < root_score if self.minimize else current_best > root_score
        if score_improves:
            super().import_root(index=best_ix)
            return None
        return None

    def on_walkers_update_end(self) -> None:
        if self.fix_best:
            self.export_root(index=0)
