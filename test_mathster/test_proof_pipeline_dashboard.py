"""Tests for the preprocess tag dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import types

import holoviews as hv
import networkx as nx
import numpy as np
import pandas as pd
import panel as pn
import pytest


def _install_fragile_stub() -> None:
    """Provide a minimal fragile.shaolin.graph stub for environments without torch."""

    if "fragile.shaolin.graph" in sys.modules:
        return

    graph_mod = types.ModuleType("fragile.shaolin.graph")

    def create_graphviz_layout(graph: nx.Graph, **_kwargs):
        return nx.spring_layout(graph, seed=0)

    def nodes_as_df(graph: nx.DiGraph, positions: dict[str, tuple[float, float]]):
        rows = []
        for node, attrs in graph.nodes(data=True):
            row = dict(attrs)
            pos = positions.get(node, (0.0, 0.0))
            row.update({"x": pos[0], "y": pos[1], "_node": node})
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=["x", "y"])
        df = pd.DataFrame(rows)
        return df.set_index("_node")

    def edges_as_df(graph: nx.DiGraph, data: bool = True):
        cols = []
        for src, dst, attrs in graph.edges(data=True):
            row = {"from": src, "to": dst}
            if data:
                row.update(attrs)
            cols.append(row)
        return pd.DataFrame(cols)

    def find_closest_point(points: np.ndarray, x: float, y: float) -> int:
        distances = np.linalg.norm(points - np.array([x, y]), axis=1)
        return int(np.argmin(distances))

    class InteractiveGraph:
        def __init__(self, df_nodes, df_edges, ignore_node_cols=(), n_cols=3):
            self.df_nodes = df_nodes
            self.df_edges = df_edges
            self.ignore_node_cols = ignore_node_cols
            self.n_cols = n_cols
            self.dmap = hv.Points([])

        def bind_to_stream(self, _callback):
            return pn.pane.Markdown("stub")

        def layout(self):
            return pn.pane.Markdown("stub layout")

    graph_mod.create_graphviz_layout = create_graphviz_layout
    graph_mod.edges_as_df = edges_as_df
    graph_mod.find_closest_point = find_closest_point
    graph_mod.InteractiveGraph = InteractiveGraph
    graph_mod.nodes_as_df = nodes_as_df

    fragile_mod = types.ModuleType("fragile")
    shaolin_mod = types.ModuleType("fragile.shaolin")
    shaolin_mod.graph = graph_mod

    sys.modules["fragile"] = fragile_mod
    sys.modules["fragile.shaolin"] = shaolin_mod
    sys.modules["fragile.shaolin.graph"] = graph_mod


_install_fragile_stub()
os.environ.setdefault("MATHSTER_SKIP_DASHBOARD_INIT", "1")

from mathster.proof_pipeline_dashboard import ProofPipelineDashboard


@pytest.fixture
def preprocess_dir(tmp_path: Path) -> Path:
    """Create a minimal preprocess directory with two linked definitions."""

    definitions = [
        {
            "label": "def-a",
            "type": "definition",
            "tags": ["tag-A"],
            "document_id": "doc-01",
            "references": ["def-b"],
            "registry_context": {"chapter_file": "chapter_1.json"},
        },
        {
            "label": "def-b",
            "type": "definition",
            "tags": ["tag-B"],
            "document_id": "doc-02",
            "references": [],
            "registry_context": {"chapter_file": "chapter_1.json"},
        },
    ]

    (tmp_path / "definition.json").write_text(json.dumps(definitions))
    # Other preprocess files can be empty lists
    for filename in (
        "algorithm.json",
        "axiom.json",
        "corollary.json",
        "lemma.json",
        "proposition.json",
        "remark.json",
        "theorem.json",
    ):
        (tmp_path / filename).write_text("[]")

    return tmp_path


def test_dashboard_builds_tag_graph(monkeypatch, preprocess_dir: Path):
    """Dashboard should build tag graph and populate filters without errors."""

    def fake_discover(self):
        return {"Test": str(preprocess_dir)}

    monkeypatch.setattr(ProofPipelineDashboard, "_discover_preprocess_dirs", fake_discover)

    dashboard = ProofPipelineDashboard(default_registry_path=str(preprocess_dir))

    assert dashboard.tag_graph.number_of_nodes() == 2
    assert "tag-A" in dashboard.tag_graph.nodes

    # Filters should default to include every tag/entity/doc
    assert dashboard.tag_filter.value == dashboard.tag_filter.options
    assert dashboard.entity_type_filter.value == dashboard.entity_type_filter.options
    assert dashboard.document_filter.value == dashboard.document_filter.options

    filtered = dashboard._filter_graph(  # pylint: disable=protected-access
        selected_tags=list(dashboard.tag_filter.options),
        entity_types=list(dashboard.entity_type_filter.options),
        documents=list(dashboard.document_filter.options),
        min_frequency=1,
        min_edge_weight=1,
    )

    # Expect one edge from tag-A to tag-B
    assert filtered.number_of_edges() == 1
    assert ("tag-A", "tag-B") in filtered.edges

    # Rendering the graph should not leave list-valued columns in df_nodes
    dashboard._render_graph(  # pylint: disable=protected-access
        selected_tags=list(dashboard.tag_filter.options),
        entity_types=list(dashboard.entity_type_filter.options),
        documents=list(dashboard.document_filter.options),
        min_frequency=1,
        min_edge_weight=1,
        layout="neato",
        hover_columns=["label"],
    )

    assert dashboard._current_df_nodes is not None  # pylint: disable=protected-access
    for column in dashboard._current_df_nodes.columns:
        assert (
            not dashboard._current_df_nodes[column]
            .map(lambda v: isinstance(v, list | tuple | set | dict))
            .any()
        )


def test_dashboard_ignore_columns_exist(monkeypatch, preprocess_dir: Path):
    """Ensure ignore_node_cols never references missing columns."""

    def fake_discover(self):
        return {"Test": str(preprocess_dir)}

    monkeypatch.setattr(ProofPipelineDashboard, "_discover_preprocess_dirs", fake_discover)

    dashboard = ProofPipelineDashboard(default_registry_path=str(preprocess_dir))

    dashboard._render_graph(  # pylint: disable=protected-access
        selected_tags=list(dashboard.tag_filter.options),
        entity_types=list(dashboard.entity_type_filter.options),
        documents=list(dashboard.document_filter.options),
        min_frequency=1,
        min_edge_weight=1,
        layout="neato",
        hover_columns=["label"],
    )

    ig = dashboard._current_ig  # pylint: disable=protected-access
    assert ig is not None
    for column in ig.ignore_node_cols:
        assert column in dashboard._current_df_nodes.columns  # pylint: disable=protected-access
