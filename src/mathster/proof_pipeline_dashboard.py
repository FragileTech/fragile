#!/usr/bin/env python3
"""Interactive dashboard for preprocess tag-reference graphs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import holoviews as hv
import networkx as nx
import numpy as np
import pandas as pd
import panel as pn

from fragile.shaolin.graph import (
    create_graphviz_layout,
    edges_as_df,
    find_closest_point,
    InteractiveGraph,
    nodes_as_df,
)
from mathster.relationships.preprocess_graph import (
    build_tag_reference_graph,
    load_preprocess_registry,
)


hv.extension("bokeh")
pn.extension("mathjax")


logger = logging.getLogger("fragile.mathster.tag_dashboard")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _has_overlap(values: list[str], allowed: set[str]) -> bool:
    if not allowed:
        return True
    return any(val in allowed for val in values)


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_non_scalar(value: Any) -> bool:
    if isinstance(value, list | tuple | set | dict):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return False


def _sequence_length(value: Any) -> int:
    if isinstance(value, list | tuple | set):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, np.ndarray):
        return int(np.prod(value.shape))
    return 0


class ProofPipelineDashboard:
    """Dashboard that visualizes the preprocess tag reference graph."""

    def __init__(self, default_registry_path: str | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.default_registry_path = (
            default_registry_path or project_root / "unified_registry" / "preprocess"
        )

        self.preprocess_dir: Path | None = None
        self.registry_data: dict[str, dict[str, object]] = {}
        self.tag_graph: nx.DiGraph = nx.DiGraph()
        self.filtered_graph: nx.DiGraph = nx.DiGraph()
        self.layout_cache: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}

        self._current_ig: InteractiveGraph | None = None
        self._current_df_nodes: pd.DataFrame | None = None

        self._create_data_source_widgets()
        self._create_filter_widgets()
        self._create_reactive_components()

        self.console_lines: list[str] = []
        self.console_pane = pn.pane.Markdown("```\nConsole initialized.\n```", width=480)
        self.node_details_container = pn.Column(
            pn.pane.Markdown(
                "**Click a tag node to see details**\n\n"
                "Filters and edge weights update the graph automatically.",
                width=480,
            )
        )

        self._load_tag_graph()

    # ------------------------------------------------------------------
    # Widget creation
    # ------------------------------------------------------------------

    def _discover_preprocess_dirs(self) -> dict[str, str]:
        project_root = Path(__file__).resolve().parents[2]
        options: dict[str, str] = {}

        unified = project_root / "unified_registry" / "preprocess"
        if unified.exists():
            options["Unified Registry (preprocess)"] = str(unified)

        per_doc_root = project_root / "registries" / "per_document"
        if per_doc_root.exists():
            for doc_dir in sorted(per_doc_root.iterdir()):
                preprocess_dir = doc_dir / "preprocess"
                if preprocess_dir.exists() and any(preprocess_dir.glob("*.json")):
                    options[f"Per-document: {doc_dir.name}"] = str(preprocess_dir)

        options["Custom Path"] = "__custom__"
        return options

    def _create_data_source_widgets(self) -> None:
        options = self._discover_preprocess_dirs()
        default_value = next(iter(options.values())) if options else "__custom__"

        self.data_source_selector = pn.widgets.Select(
            name="Preprocess Source",
            options=options,
            value=default_value,
            width=280,
        )

        self.custom_path_input = pn.widgets.TextInput(
            name="Preprocess Directory",
            value=str(self.default_registry_path),
            width=280,
            visible=default_value == "__custom__",
        )

        self.reload_button = pn.widgets.Button(
            name="Reload Graph",
            button_type="primary",
            width=280,
        )
        self.reload_button.on_click(self._on_reload_requested)

        def _toggle_custom(event) -> None:
            self.custom_path_input.visible = event.new == "__custom__"

        self.data_source_selector.param.watch(_toggle_custom, "value")

    def _create_filter_widgets(self) -> None:
        self.tag_filter = pn.widgets.MultiChoice(
            name="Tags",
            options=[],
            value=[],
            width=280,
            placeholder="Select tags (defaults to all)",
        )

        self.entity_type_filter = pn.widgets.MultiChoice(
            name="Entity Types",
            options=[],
            value=[],
            width=280,
        )

        self.document_filter = pn.widgets.MultiChoice(
            name="Documents",
            options=[],
            value=[],
            width=280,
        )

        self.min_frequency_slider = pn.widgets.IntSlider(
            name="Min Tag Frequency",
            start=1,
            end=10,
            value=1,
            step=1,
            width=280,
        )

        self.min_edge_weight_slider = pn.widgets.IntSlider(
            name="Min Edge Weight",
            start=1,
            end=5,
            value=1,
            step=1,
            width=280,
        )

        self.layout_selector = pn.widgets.Select(
            name="Layout Algorithm",
            options={
                "Spring (neato)": "neato",
                "Hierarchical (dot)": "dot",
                "Force-directed (sfdp)": "sfdp",
                "Spectral": "spectral",
                "Circular": "circular",
            },
            value="neato",
            width=280,
        )

        self.hover_columns_selector = pn.widgets.MultiChoice(
            name="Hover Fields",
            options=["label", "frequency", "entity_types", "documents", "in_degree", "out_degree"],
            value=["label", "frequency", "entity_types"],
            width=280,
        )

        self.reset_filters_button = pn.widgets.Button(
            name="Reset Filters",
            button_type="warning",
            width=280,
        )
        self.reset_filters_button.on_click(self._reset_filters)

    def _create_reactive_components(self) -> None:
        self.graph_view = pn.bind(
            self._render_graph,
            selected_tags=self.tag_filter,
            entity_types=self.entity_type_filter,
            documents=self.document_filter,
            min_frequency=self.min_frequency_slider,
            min_edge_weight=self.min_edge_weight_slider,
            layout=self.layout_selector,
            hover_columns=self.hover_columns_selector,
        )

        self.stats_view = pn.bind(self._render_statistics)

    # ------------------------------------------------------------------
    # Data loading & filtering
    # ------------------------------------------------------------------

    def _resolve_preprocess_dir(self) -> Path:
        selection = self.data_source_selector.value
        if selection == "__custom__":
            return Path(self.custom_path_input.value).expanduser()
        return Path(selection)

    def _load_tag_graph(self) -> None:
        preprocess_dir = self._resolve_preprocess_dir()
        self.preprocess_dir = preprocess_dir

        if not preprocess_dir.exists():
            logger.warning("Preprocess directory %s does not exist", preprocess_dir)
            self.registry_data = {}
            self.tag_graph = nx.DiGraph()
        else:
            self.registry_data = load_preprocess_registry(preprocess_dir)
            self.tag_graph = build_tag_reference_graph(preprocess_dir)
            self._augment_node_metadata()

        self._update_filters_from_graph()
        self._append_console(f"[Load] Graph built from {preprocess_dir}")

    def _update_filters_from_graph(self) -> None:
        tags = sorted(self.tag_graph.nodes())
        self.tag_filter.options = tags
        self.tag_filter.value = tags

        entity_types = sorted({
            et
            for _, data in self.tag_graph.nodes(data=True)
            for et in data.get("entity_types", [])
        })
        self.entity_type_filter.options = entity_types
        self.entity_type_filter.value = entity_types

        documents = sorted({
            doc for _, data in self.tag_graph.nodes(data=True) for doc in data.get("documents", [])
        })
        self.document_filter.options = documents
        self.document_filter.value = documents

        max_frequency = max(
            (data.get("frequency", 1) for _, data in self.tag_graph.nodes(data=True)),
            default=1,
        )
        self.min_frequency_slider.start = 1
        self.min_frequency_slider.end = max_frequency
        self.min_frequency_slider.value = 1

        max_weight = max(
            (data.get("weight", 1) for _, _, data in self.tag_graph.edges(data=True)),
            default=1,
        )
        self.min_edge_weight_slider.start = 1
        self.min_edge_weight_slider.end = max_weight
        self.min_edge_weight_slider.value = 1

        self.layout_cache.clear()

    def _reset_filters(self, _event) -> None:
        self.tag_filter.value = list(self.tag_filter.options)
        self.entity_type_filter.value = list(self.entity_type_filter.options)
        self.document_filter.value = list(self.document_filter.options)
        self.min_frequency_slider.value = self.min_frequency_slider.start
        self.min_edge_weight_slider.value = self.min_edge_weight_slider.start

    def _on_reload_requested(self, _event) -> None:
        self._append_console("[Reload] User requested graph rebuild")
        self._load_tag_graph()

    # ------------------------------------------------------------------
    # Graph rendering
    # ------------------------------------------------------------------

    def _augment_node_metadata(self) -> None:
        for tag, data in self.tag_graph.nodes(data=True):
            labels = data.get("labels", [])
            entity_types: set[str] = set()
            documents: set[str] = set()
            chapters: set[str] = set()
            for label in labels:
                entry = self.registry_data.get(label, {}) or {}
                entity_type = _safe_str(entry.get("type"))
                if entity_type:
                    entity_types.add(entity_type)
                document = entry.get("document_id") or (entry.get("registry_context") or {}).get(
                    "document_id"
                )
                doc_str = _safe_str(document)
                if doc_str:
                    documents.add(doc_str)
                chapter = (entry.get("registry_context") or {}).get("chapter_file") or (
                    entry.get("registry_context") or {}
                ).get("chapter_index")
                chap_str = _safe_str(chapter)
                if chap_str:
                    chapters.add(chap_str)

            data["entity_types"] = sorted(entity_types)
            data["documents"] = sorted(documents)
            data["chapters"] = sorted(chapters)

    def _filter_graph(
        self,
        selected_tags: list[str],
        entity_types: list[str],
        documents: list[str],
        min_frequency: int,
        min_edge_weight: int,
    ) -> nx.DiGraph:
        if self.tag_graph.number_of_nodes() == 0:
            return nx.DiGraph()

        allowed_tags = set(selected_tags or self.tag_graph.nodes())
        allowed_entity_types = set(entity_types or self.entity_type_filter.options)
        allowed_documents = set(documents or self.document_filter.options)
        nodes_to_keep = {
            node
            for node, data in self.tag_graph.nodes(data=True)
            if data.get("frequency", 0) >= min_frequency
            and node in allowed_tags
            and _has_overlap(data.get("entity_types", []), allowed_entity_types)
            and _has_overlap(data.get("documents", []), allowed_documents)
        }

        subgraph = self.tag_graph.subgraph(nodes_to_keep).copy()

        edges_to_remove = [
            (u, v)
            for u, v, data in subgraph.edges(data=True)
            if data.get("weight", 0) < min_edge_weight
        ]
        subgraph.remove_edges_from(edges_to_remove)

        return subgraph

    def _render_graph(
        self,
        selected_tags: list[str],
        entity_types: list[str],
        documents: list[str],
        min_frequency: int,
        min_edge_weight: int,
        layout: str,
        hover_columns: list[str],
    ):
        graph = self._filter_graph(
            selected_tags,
            entity_types,
            documents,
            min_frequency,
            min_edge_weight,
        )
        self.filtered_graph = graph

        if graph.number_of_nodes() == 0:
            return pn.pane.Markdown(
                "**No tags match the current filters.**\n\nAdjust the sliders or tag selection.",
                sizing_mode="stretch_width",
            )

        layout_positions = self._get_or_compute_layout(graph, layout)
        df_nodes = nodes_as_df(graph, layout_positions)
        df_nodes["label"] = df_nodes.index
        df_nodes["frequency"] = [graph.nodes[n].get("frequency", 0) for n in df_nodes.index]
        df_nodes["entity_types"] = [
            ", ".join(graph.nodes[n].get("entity_types", [])) for n in df_nodes.index
        ]
        df_nodes["documents"] = [
            ", ".join(graph.nodes[n].get("documents", [])) for n in df_nodes.index
        ]
        df_nodes["in_degree"] = [graph.in_degree(n) for n in df_nodes.index]
        df_nodes["out_degree"] = [graph.out_degree(n) for n in df_nodes.index]

        df_nodes = self._drop_non_scalar_columns(df_nodes, preserve_counts={"labels"})

        df_edges = (
            edges_as_df(graph, data=True)
            if graph.number_of_edges()
            else pd.DataFrame(columns=["from", "to"])
        )
        df_edges = self._drop_non_scalar_columns(df_edges)

        for col in ["frequency", "in_degree", "out_degree"]:
            if col in df_nodes.columns:
                df_nodes[col] = pd.to_numeric(df_nodes[col])

        ignore_node_cols = tuple(col for col in ("labels",) if col in df_nodes.columns)

        ig = InteractiveGraph(
            df_nodes=df_nodes,
            df_edges=df_edges,
            ignore_node_cols=ignore_node_cols,
            n_cols=3,
        )
        self._current_ig = ig
        self._current_df_nodes = df_nodes

        fields = list(hover_columns) if hover_columns else ["label"]
        fields = [f for f in fields if f in df_nodes.columns]
        if "label" not in fields:
            fields.insert(0, "label")
        tooltips = [(field, f"@{field}") for field in fields]
        ig.dmap = ig.dmap.opts(tools=["tap", "hover"], hover_tooltips=tooltips)

        bound = ig.bind_to_stream(self._on_node_select_xy)
        self.node_details_container.objects = [bound]

        return ig.dmap

    @staticmethod
    def _drop_non_scalar_columns(
        df: pd.DataFrame,
        preserve_counts: set[str] | None = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        keep = df.copy()
        preserve_counts = preserve_counts or set()

        for column in list(keep.columns):
            series = keep[column]
            if series.map(_is_non_scalar).any():
                if column in preserve_counts:
                    keep[f"{column}_count"] = series.map(_sequence_length)
                keep = keep.drop(columns=[column])
        return keep

    def _get_or_compute_layout(self, graph: nx.DiGraph, layout_name: str):
        cache_key = (layout_name, f"{graph.number_of_nodes()}_{graph.number_of_edges()}")
        if cache_key in self.layout_cache:
            return self.layout_cache[cache_key]

        if layout_name in {"dot", "neato", "fdp", "sfdp", "circo", "twopi"}:
            layout = create_graphviz_layout(graph, prog=layout_name, top_to_bottom=False)
        elif layout_name == "spectral":
            layout = nx.spectral_layout(graph)
        elif layout_name == "circular":
            layout = nx.circular_layout(graph)
        else:
            layout = nx.spring_layout(graph, k=0.6, iterations=50)

        self.layout_cache[cache_key] = layout
        return layout

    # ------------------------------------------------------------------
    # Node selection & statistics
    # ------------------------------------------------------------------

    def _on_node_select_xy(self, x: float, y: float):
        df = self._current_df_nodes
        if df is None or df.empty:
            return pn.pane.Markdown("**No node data available.**", width=480)

        try:
            points = df[["x", "y"]].values
            idx = int(find_closest_point(points, x, y))
            label = df.index[idx]
            self._append_console(f"[NodeTap] ({x:.3f}, {y:.3f}) -> {label}")
            return pn.Column(self._render_node_details(label))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Node selection error: %s", exc)
            return pn.pane.Markdown(f"**Error:** {exc}", width=480)

    def _render_node_details(self, label: str):
        graph = self.filtered_graph
        if not graph or label not in graph.nodes:
            return pn.pane.Markdown(f"**Tag:** `{label}`\n\n*Not present in graph.*", width=480)

        data = graph.nodes[label]
        frequency = data.get("frequency", 0)
        labels = data.get("labels", [])
        entity_types = data.get("entity_types", [])
        documents = data.get("documents", [])
        outgoing = sorted(graph.successors(label))
        incoming = sorted(graph.predecessors(label))

        lines = ["## Tag", f"**Label:** `{label}`", f"**Entities:** {frequency}"]

        if entity_types:
            lines.append(f"**Entity Types:** {', '.join(entity_types)}")

        if documents:
            lines.append(f"**Documents:** {', '.join(documents)}")

        if labels:
            preview = "\n".join(f"- `{lbl}`" for lbl in labels[:15])
            more = f"\n- … ({len(labels) - 15} more)" if len(labels) > 15 else ""
            lines.append("\n**Entity Labels:**\n" + preview + more)

        if outgoing:
            lines.append(
                "\n**Outgoing References:**\n" + "\n".join(f"- `{dst}`" for dst in outgoing[:15])
            )
            if len(outgoing) > 15:
                lines.append(f"- … ({len(outgoing) - 15} more)")

        if incoming:
            lines.append(
                "\n**Incoming References:**\n" + "\n".join(f"- `{src}`" for src in incoming[:15])
            )
            if len(incoming) > 15:
                lines.append(f"- … ({len(incoming) - 15} more)")

        return pn.pane.Markdown("\n".join(lines), width=480)

    def _render_statistics(self):
        graph = self.filtered_graph or self.tag_graph
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()

        top_tags = sorted(
            graph.nodes(data=True),
            key=lambda item: item[1].get("frequency", 0),
            reverse=True,
        )[:10]

        documents = {}
        for _, data in graph.nodes(data=True):
            for doc in data.get("documents", []):
                documents[doc] = documents.get(doc, 0) + 1

        lines = [
            "## Tag Graph Statistics",
            f"**Nodes (tags):** {node_count}",
            f"**Edges (references):** {edge_count}",
        ]

        if top_tags:
            lines.append("\n**Top Tags by Frequency:**")
            for tag, data in top_tags:
                lines.append(f"- `{tag}`: {data.get('frequency', 0)} entities")

        if documents:
            lines.append("\n**Tags by Document:**")
            for doc, count in sorted(documents.items()):
                lines.append(f"- `{doc}`: {count}")

        return pn.pane.Markdown("\n".join(lines), width=480)

    # ------------------------------------------------------------------
    # Console helper
    # ------------------------------------------------------------------

    def _append_console(self, message: str) -> None:
        self.console_lines.append(message)
        self.console_lines = self.console_lines[-200:]
        self.console_pane.object = "```\n" + "\n".join(self.console_lines) + "\n```"

    # ------------------------------------------------------------------
    # Template assembly
    # ------------------------------------------------------------------

    def create_dashboard(self) -> pn.Template:
        sidebar = [
            pn.pane.Markdown("## Data Source", margin=(0, 0, 10, 0)),
            self.data_source_selector,
            self.custom_path_input,
            self.reload_button,
            pn.layout.Divider(),
            pn.pane.Markdown("## Filters", margin=(10, 0, 10, 0)),
            self.tag_filter,
            self.entity_type_filter,
            self.document_filter,
            self.min_frequency_slider,
            self.min_edge_weight_slider,
            self.layout_selector,
            self.hover_columns_selector,
            self.reset_filters_button,
        ]

        main_content = [
            pn.Card(
                pn.panel(self.graph_view),
                title="Tag Reference Graph",
                sizing_mode="stretch_width",
            ),
            pn.Row(
                pn.Card(
                    self.node_details_container,
                    title="Tag Details",
                    width=900,
                    height=700,
                    scroll=True,
                ),
                pn.Card(
                    pn.panel(self.stats_view),
                    title="Statistics",
                    width=400,
                    height=700,
                    scroll=True,
                ),
            ),
            pn.Card(self.console_pane, title="Console", width=600, height=250, scroll=True),
        ]

        return pn.template.FastListTemplate(
            title="Proof Pipeline Tag Dashboard",
            sidebar=sidebar,
            main=main_content,
            accent_base_color="#3498db",
            header_background="#2c3e50",
        )


def main() -> pn.Template:
    return ProofPipelineDashboard().create_dashboard()


if __name__ == "__main__":
    template = main()
    pn.serve(template.servable(), port=5006, show=False)
elif not os.environ.get("MATHSTER_SKIP_DASHBOARD_INIT"):
    template = main()
    template.servable()
else:
    template = None
