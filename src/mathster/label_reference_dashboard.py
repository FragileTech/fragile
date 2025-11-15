#!/usr/bin/env python3
"""Interactive dashboard for label-level reference graphs across registry stages."""

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
from mathster.relationships.directives_graph import (
    build_label_reference_graph as build_directives_label_reference_graph,
    load_directives_registry,
)
from mathster.relationships.preprocess_graph import (
    build_label_reference_graph as build_preprocess_label_reference_graph,
    load_preprocess_registry,
)
from mathster.reports.show_report import render_label_report


hv.extension("bokeh")
pn.extension("mathjax")


logger = logging.getLogger("fragile.mathster.label_dashboard")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

ENTITY_TYPE_COLORS: dict[str, str] = {
    "definition": "#0B84A5",
    "theorem": "#F6C85F",
    "lemma": "#6F4E7C",
    "proposition": "#9DD866",
    "corollary": "#CA472F",
    "assumption": "#FF8C42",
    "axiom": "#2176FF",
    "remark": "#BC5090",
    "proof": "#58508D",
    "algorithm": "#FF6361",
    "example": "#2A9D8F",
    "unknown": "#95A5A6",
}
DEFAULT_ENTITY_COLOR = "#7F8C8D"


def _has_overlap(values: list[str], allowed: set[str]) -> bool:
    if not allowed:
        return True
    return any(val in allowed for val in values if val)


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


def _matches_search(label: str, data: dict[str, Any], search_text: str) -> bool:
    if not search_text:
        return True
    needles = [label]
    for key in ("title", "section", "document_id"):
        value = data.get(key)
        if isinstance(value, str):
            needles.append(value)
    for tag in data.get("tags", []):
        needles.append(tag)
    lowercase = search_text.lower()
    return any(lowercase in (needle or "").lower() for needle in needles)


class LabelReferenceDashboard:
    """Dashboard that visualizes label references across preprocess and directives stages."""

    def __init__(self, default_registry_path: str | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        preprocess_default = (
            Path(default_registry_path).expanduser()
            if default_registry_path
            else project_root / "unified_registry" / "preprocess"
        )
        self.default_registry_paths: dict[str, Path] = {
            "preprocess": preprocess_default,
            "directives": project_root / "unified_registry" / "directives",
        }

        self.registry_dir: Path | None = None
        self.registry_data: dict[str, dict[str, Any]] = {}
        self.label_graph: nx.DiGraph = nx.DiGraph()
        self.filtered_graph: nx.DiGraph = nx.DiGraph()
        self.layout_cache: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
        self._stage_selection: dict[str, str] = {}
        self._custom_path_history: dict[str, str] = {
            stage: str(path) for stage, path in self.default_registry_paths.items()
        }

        self._current_ig: InteractiveGraph | None = None
        self._current_df_nodes: pd.DataFrame | None = None

        self._create_data_source_widgets()
        self._create_filter_widgets()
        self._create_reactive_components()

        self.console_lines: list[str] = []
        self.console_pane = pn.pane.Markdown("```\nConsole initialized.\n```", width=480)
        self.node_details_container = pn.Column(
            pn.pane.Markdown(
                "**Click a label node to see details**\n\n"
                "Filters and edge weights update the graph automatically.",
                width=480,
            )
        )
        self.dimension_controls = pn.Column(
            pn.pane.Markdown(
                "**Graph Appearance Controls**\n\n"
                "Render the graph to expose color/size mapping widgets.",
                width=600,
            )
        )

        self._load_label_graph()

    # ------------------------------------------------------------------
    # Widget creation
    # ------------------------------------------------------------------

    def _discover_registry_dirs(self, stage: str) -> dict[str, str]:
        project_root = Path(__file__).resolve().parents[2]
        options: dict[str, str] = {}
        stage = stage or "preprocess"
        stage_label = stage.capitalize()

        unified = project_root / "unified_registry" / stage
        if unified.exists():
            options[f"Unified Registry ({stage_label})"] = str(unified)

        per_doc_root = project_root / "registries" / "per_document"
        if per_doc_root.exists():
            for doc_dir in sorted(per_doc_root.iterdir()):
                stage_dir = doc_dir / stage
                if stage_dir.exists() and any(stage_dir.glob("*.json")):
                    options[f"Per-document ({stage_label}): {doc_dir.name}"] = str(stage_dir)

        options[f"Custom Path ({stage_label})"] = "__custom__"
        return options

    def _create_data_source_widgets(self) -> None:
        self.stage_toggle = pn.widgets.RadioButtonGroup(
            name="Registry Stage",
            options={"Preprocess": "preprocess", "Directives": "directives"},
            value="preprocess",
            button_type="primary",
            width=280,
        )

        self.data_source_selector = pn.widgets.Select(
            name="Registry Source",
            options={},
            value=None,
            width=280,
        )

        self.custom_path_input = pn.widgets.TextInput(
            name="Registry Directory",
            value=str(self.default_registry_paths["preprocess"]),
            width=280,
            visible=False,
        )

        self.reload_button = pn.widgets.Button(
            name="Reload Graph",
            button_type="primary",
            width=280,
        )
        self.reload_button.on_click(self._on_reload_requested)

        def _toggle_custom(event) -> None:
            is_custom = event.new == "__custom__"
            self.custom_path_input.visible = is_custom
            stage = self.stage_toggle.value
            if is_custom:
                default_path = self._custom_path_history.get(
                    stage, str(self.default_registry_paths.get(stage, ""))
                )
                if default_path:
                    self.custom_path_input.value = default_path
            self._stage_selection[stage] = event.new

        def _remember_custom_path(event) -> None:
            stage = self.stage_toggle.value
            self._custom_path_history[stage] = event.new

        self.stage_toggle.param.watch(self._on_stage_change, "value")
        self.data_source_selector.param.watch(_toggle_custom, "value")
        self.custom_path_input.param.watch(_remember_custom_path, "value")
        self._refresh_data_source_options()

    def _refresh_data_source_options(self) -> None:
        stage = getattr(self, "stage_toggle", None)
        stage_value = stage.value if stage else "preprocess"
        stage_label = stage_value.capitalize()
        options = self._discover_registry_dirs(stage_value)
        if not options:
            options = {f"Custom Path ({stage_label})": "__custom__"}

        available_values = list(options.values())
        preferred_value = self._stage_selection.get(stage_value)
        value = preferred_value if preferred_value in available_values else available_values[0]

        self.data_source_selector.options = options
        self.data_source_selector.value = value
        self.data_source_selector.name = f"{stage_label} Source"

        is_custom = value == "__custom__"
        self.custom_path_input.visible = is_custom
        self.custom_path_input.name = f"{stage_label} Directory"
        if is_custom:
            default_path = self._custom_path_history.get(
                stage_value, str(self.default_registry_paths.get(stage_value, ""))
            )
            if default_path:
                self.custom_path_input.value = default_path
        else:
            self._stage_selection[stage_value] = value

    def _create_filter_widgets(self) -> None:
        self.label_filter = pn.widgets.MultiChoice(
            name="Labels",
            options=[],
            value=[],
            width=280,
            placeholder="Select labels (defaults to all)",
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

        self.section_filter = pn.widgets.MultiChoice(
            name="Sections",
            options=[],
            value=[],
            width=280,
        )

        self.tag_filter = pn.widgets.MultiChoice(
            name="Tags",
            options=[],
            value=[],
            width=280,
        )

        self.filter_accordion = pn.Accordion(
            ("Label Filters", pn.Column(self.label_filter, sizing_mode="stretch_width")),
            ("Section Filters", pn.Column(self.section_filter, sizing_mode="stretch_width")),
            ("Tag Filters", pn.Column(self.tag_filter, sizing_mode="stretch_width")),
            sizing_mode="stretch_width",
        )
        self.filter_accordion.active = []

        self.search_input = pn.widgets.TextInput(
            name="Search",
            placeholder="Substring in label/title/tag",
            width=280,
        )

        self.min_degree_slider = pn.widgets.IntSlider(
            name="Min Degree",
            start=0,
            end=5,
            value=0,
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
            options=[
                "label",
                "entity_type",
                "document_id",
                "section",
                "title",
                "tags",
                "in_degree",
                "out_degree",
            ],
            value=["label", "entity_type", "document_id"],
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
            selected_labels=self.label_filter,
            entity_types=self.entity_type_filter,
            documents=self.document_filter,
            sections=self.section_filter,
            tags=self.tag_filter,
            search_text=self.search_input,
            min_degree=self.min_degree_slider,
            min_edge_weight=self.min_edge_weight_slider,
            layout=self.layout_selector,
            hover_columns=self.hover_columns_selector,
        )

        self.stats_view = pn.bind(self._render_statistics)
        self.connectivity_view = pn.bind(self._render_connectivity_report)
        self.legend_view = pn.bind(self._create_entity_type_legend)

    # ------------------------------------------------------------------
    # Data loading & filtering
    # ------------------------------------------------------------------

    def _resolve_registry_dir(self) -> Path:
        selection = self.data_source_selector.value
        if selection == "__custom__":
            custom_value = self.custom_path_input.value.strip()
            if custom_value:
                return Path(custom_value).expanduser()
        if selection:
            return Path(selection)
        default_path = self.default_registry_paths.get(self.stage_toggle.value)
        if default_path:
            return default_path
        return Path(".")

    def _resolve_report_preprocess_dir(self) -> Path | None:
        stage = self.stage_toggle.value
        current_dir = self.registry_dir or self._resolve_registry_dir()
        if stage == "preprocess":
            return current_dir
        if stage == "directives" and current_dir:
            sibling = current_dir.parent / "preprocess"
            if sibling.exists():
                return sibling
        fallback = self.default_registry_paths.get("preprocess")
        if fallback and fallback.exists():
            return fallback
        return None

    def _get_stage_handlers(self, stage: str):
        if stage == "directives":
            return load_directives_registry, build_directives_label_reference_graph
        return load_preprocess_registry, build_preprocess_label_reference_graph

    def _load_label_graph(self) -> None:
        stage = self.stage_toggle.value
        registry_dir = self._resolve_registry_dir()
        self.registry_dir = registry_dir

        loader, builder = self._get_stage_handlers(stage)

        if not registry_dir.exists():
            logger.warning("%s directory %s does not exist", stage.title(), registry_dir)
            self.registry_data = {}
            self.label_graph = nx.DiGraph()
        else:
            self.registry_data = loader(registry_dir)
            self.label_graph = builder(registry_dir)
            self._augment_node_metadata(stage)

        self._update_filters_from_graph()
        self._append_console(f"[Load] ({stage}) label graph built from {registry_dir}")

    def _on_stage_change(self, event) -> None:
        previous_stage = event.old
        if previous_stage:
            self._stage_selection[previous_stage] = self.data_source_selector.value or ""
        self.layout_cache.clear()
        self._append_console(f"[Stage] Switched to {event.new}")
        self._refresh_data_source_options()
        self._load_label_graph()

    def _update_filters_from_graph(self) -> None:
        labels = sorted(self.label_graph.nodes())
        self.label_filter.options = labels
        self.label_filter.value = labels

        entity_types = sorted({
            data.get("entity_type")
            for _, data in self.label_graph.nodes(data=True)
            if data.get("entity_type")
        })
        self.entity_type_filter.options = entity_types
        self.entity_type_filter.value = entity_types

        documents = sorted({
            data.get("document_id")
            for _, data in self.label_graph.nodes(data=True)
            if data.get("document_id")
        })
        self.document_filter.options = documents
        self.document_filter.value = documents

        sections = sorted({
            data.get("section")
            for _, data in self.label_graph.nodes(data=True)
            if data.get("section")
        })
        self.section_filter.options = sections
        self.section_filter.value = sections

        tags = sorted({
            tag
            for _, data in self.label_graph.nodes(data=True)
            for tag in data.get("tags", [])
            if tag
        })
        self.tag_filter.options = tags
        self.tag_filter.value = tags

        max_degree = max(
            (self.label_graph.degree(n) for n in self.label_graph.nodes()),
            default=0,
        )
        self.min_degree_slider.start = 0
        self.min_degree_slider.end = max(1, max_degree)
        self.min_degree_slider.value = 0

        max_weight = max(
            (data.get("weight", 1) for _, _, data in self.label_graph.edges(data=True)),
            default=1,
        )
        self.min_edge_weight_slider.start = 1
        self.min_edge_weight_slider.end = max_weight
        self.min_edge_weight_slider.value = 1

        self.layout_cache.clear()

    def _reset_filters(self, _event) -> None:
        self.label_filter.value = list(self.label_filter.options)
        self.entity_type_filter.value = list(self.entity_type_filter.options)
        self.document_filter.value = list(self.document_filter.options)
        self.section_filter.value = list(self.section_filter.options)
        self.tag_filter.value = list(self.tag_filter.options)
        self.search_input.value = ""
        self.min_degree_slider.value = self.min_degree_slider.start
        self.min_edge_weight_slider.value = self.min_edge_weight_slider.start

    def _on_reload_requested(self, _event) -> None:
        self._append_console("[Reload] User requested graph rebuild")
        self._load_label_graph()

    # ------------------------------------------------------------------
    # Graph rendering
    # ------------------------------------------------------------------

    def _augment_node_metadata(self, stage: str) -> None:
        for label, data in self.label_graph.nodes(data=True):
            entry = self.registry_data.get(label, {}) or {}
            if stage == "directives":
                entity_type = (
                    data.get("entity_type")
                    or data.get("directive_type")
                    or _safe_str(entry.get("directive_type"))
                    or "unknown"
                )
                registry_ctx = entry.get("_registry_context") or {}
            else:
                entity_type = data.get("entity_type") or _safe_str(entry.get("type")) or "unknown"
                registry_ctx = entry.get("registry_context") or {}

            title = data.get("title") or _safe_str(entry.get("title") or entry.get("term"))
            document_id = (
                data.get("document_id")
                or _safe_str(entry.get("document_id"))
                or _safe_str(registry_ctx.get("document_id"))
            )
            section = data.get("section") or _safe_str(entry.get("section"))
            if not section:
                section = _safe_str(
                    registry_ctx.get("section_id")
                    or registry_ctx.get("chapter_file")
                    or registry_ctx.get("section")
                )

            data["entity_type"] = entity_type
            if stage == "directives":
                data.setdefault("directive_type", entity_type)

            data["title"] = title
            data["document_id"] = document_id
            data["section"] = section
            raw_tags: list[str] = []
            raw_tags.extend(filter(None, data.get("tags") or []))
            entry_tags = entry.get("tags") or []
            for tag in entry_tags:
                clean_tag = _safe_str(tag)
                if clean_tag:
                    raw_tags.append(clean_tag)
            data["tags"] = sorted({tag for tag in raw_tags if tag})

    def _filter_graph(
        self,
        selected_labels: list[str],
        entity_types: list[str],
        documents: list[str],
        sections: list[str],
        tags: list[str],
        search_text: str,
        min_degree: int,
        min_edge_weight: int,
    ) -> nx.DiGraph:
        if self.label_graph.number_of_nodes() == 0:
            return nx.DiGraph()

        allowed_labels = set(selected_labels or self.label_graph.nodes())
        allowed_entity_types = set(entity_types or self.entity_type_filter.options)
        allowed_documents = set(documents or self.document_filter.options)
        allowed_sections = set(sections or self.section_filter.options)
        allowed_tags = set(tags or self.tag_filter.options)
        search_text = (search_text or "").strip()

        nodes_to_keep = {
            node
            for node, data in self.label_graph.nodes(data=True)
            if node in allowed_labels
            and _has_overlap([data.get("entity_type")], allowed_entity_types)
            and _has_overlap([data.get("document_id")], allowed_documents)
            and _has_overlap([data.get("section")], allowed_sections)
            and _has_overlap(data.get("tags", []), allowed_tags)
            and _matches_search(node, data, search_text)
        }

        subgraph = self.label_graph.subgraph(nodes_to_keep).copy()

        edges_to_remove = [
            (u, v)
            for u, v, data in subgraph.edges(data=True)
            if data.get("weight", 0) < min_edge_weight
        ]
        subgraph.remove_edges_from(edges_to_remove)

        if min_degree > 0:
            low_degree_nodes = [
                node for node in subgraph.nodes() if subgraph.degree(node) < min_degree
            ]
            subgraph.remove_nodes_from(low_degree_nodes)

        return subgraph

    def _render_graph(
        self,
        selected_labels: list[str],
        entity_types: list[str],
        documents: list[str],
        sections: list[str],
        tags: list[str],
        search_text: str,
        min_degree: int,
        min_edge_weight: int,
        layout: str,
        hover_columns: list[str],
    ):
        graph = self._filter_graph(
            selected_labels,
            entity_types,
            documents,
            sections,
            tags,
            search_text,
            min_degree,
            min_edge_weight,
        )
        self.filtered_graph = graph

        if graph.number_of_nodes() == 0:
            return pn.pane.Markdown(
                "**No labels match the current filters.**\n\nAdjust the sliders or selection.",
                sizing_mode="stretch_width",
            )

        layout_positions = self._get_or_compute_layout(graph, layout)
        df_nodes = nodes_as_df(graph, layout_positions)
        df_nodes["label"] = df_nodes.index
        df_nodes["entity_type"] = [graph.nodes[n].get("entity_type", "") for n in df_nodes.index]
        df_nodes["document_id"] = [graph.nodes[n].get("document_id", "") for n in df_nodes.index]
        df_nodes["section"] = [graph.nodes[n].get("section", "") for n in df_nodes.index]
        df_nodes["title"] = [graph.nodes[n].get("title", "") for n in df_nodes.index]
        df_nodes["tags"] = [", ".join(graph.nodes[n].get("tags", [])) for n in df_nodes.index]
        df_nodes["in_degree"] = [graph.in_degree(n) for n in df_nodes.index]
        df_nodes["out_degree"] = [graph.out_degree(n) for n in df_nodes.index]

        df_nodes["entity_type"] = (
            df_nodes["entity_type"]
            .fillna("unknown")
            .replace("", "unknown")
            .str.strip()
            .str.lower()
        )
        available_types = list(dict.fromkeys(df_nodes["entity_type"]))  # preserve order
        custom_order = [etype for etype in ENTITY_TYPE_COLORS if etype in available_types]
        remaining = [etype for etype in available_types if etype not in custom_order]
        category_order = custom_order + remaining
        df_nodes["entity_type"] = pd.Categorical(
            df_nodes["entity_type"],
            categories=category_order,
            ordered=True,
        )
        palette = [
            ENTITY_TYPE_COLORS.get(cat, DEFAULT_ENTITY_COLOR)
            for cat in df_nodes["entity_type"].cat.categories
        ]

        df_nodes = self._drop_non_scalar_columns(df_nodes)

        if graph.number_of_edges():
            df_edges = edges_as_df(graph, data=True)
        else:
            df_edges = pd.DataFrame(columns=["from", "to"])
        df_edges = self._drop_non_scalar_columns(
            df_edges, preserve_counts={"contexts", "context_counts"}
        )

        for col in ["in_degree", "out_degree"]:
            if col in df_nodes.columns:
                df_nodes[col] = pd.to_numeric(df_nodes[col])

        ig = InteractiveGraph(
            df_nodes=df_nodes,
            df_edges=df_edges,
            ignore_node_cols=(),
            n_cols=3,
        )
        color_dim = ig.node_dims.dimensions.get("node_color")
        if color_dim and "entity_type" in getattr(color_dim, "valid_cols", []):
            color_dim.cmap = palette or list(ENTITY_TYPE_COLORS.values())
            color_dim.column.value = "entity_type"
        self._current_ig = ig
        self._current_df_nodes = df_nodes
        try:
            self.dimension_controls.objects = [ig.layout()]
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to build dimension controls: %s", exc)
            self.dimension_controls.objects = [
                pn.pane.Markdown(
                    "**Graph Appearance Controls**\n\n"
                    "Unable to render dimension widgets. Check console for details.",
                    width=600,
                )
            ]

        fields = list(hover_columns) if hover_columns else ["label"]
        fields = [field for field in fields if field in df_nodes.columns]
        if "label" not in fields:
            fields.insert(0, "label")
        tooltips = [(field, f"@{field}") for field in fields]
        ig.dmap = ig.dmap.opts(
            tools=["tap", "hover"],
            hover_tooltips=tooltips,
            xaxis=None,
            yaxis=None,
        )

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
        #  X AND Y ARE SWAPPED BY DEFAULT, LET GET THEM RIGHT
        x, y = y, x
        df = self._current_df_nodes
        if df is None or df.empty:
            return pn.pane.Markdown("**No node data available.**", width=480)

        try:
            points = df[["x", "y"]].values
            idx = int(find_closest_point(points, x, y))
            label = df.index[idx]
            self._append_console(f"[NodeTap] ({x:.3f}, {y:.3f}) -> {label}")
            return self._render_node_details(label)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Node selection error: %s", exc)
            return pn.pane.Markdown(f"**Error:** {exc}", width=480)

    def _render_node_details(self, label: str):
        graph = self.filtered_graph
        if not graph or label not in graph.nodes:
            return pn.Column(
                pn.pane.Markdown(
                    f"**Label:** `{label}`\n\n*Not present in graph.*",
                    width=480,
                ),
                sizing_mode="stretch_width",
            )

        data = graph.nodes[label]
        stage = self.stage_toggle.value
        stage_label = "Directives" if stage == "directives" else "Preprocess"
        entity_type = data.get("entity_type", "unknown")
        title = data.get("title") or "(no title)"
        document_id = data.get("document_id") or "?"
        section = data.get("section") or "?"
        tags = ", ".join(data.get("tags", [])) or "(none)"

        outgoing = sorted(graph.successors(label))
        incoming = sorted(graph.predecessors(label))

        lines = [
            "## Label",
            f"**ID:** `{label}`",
            f"**Title:** {title}",
            f"**Entity Type:** {entity_type}",
            f"**Registry Stage:** {stage_label}",
            f"**Document:** {document_id}",
            f"**Section:** {section}",
            f"**Tags:** {tags}",
        ]

        if stage == "directives":
            start_line = data.get("start_line")
            end_line = data.get("end_line")
            chapter_file = data.get("chapter_file")
            chapter_index = data.get("chapter_index")
            lines.append(f"**Chapter File:** {chapter_file or '—'}")
            if chapter_index is not None:
                lines.append(f"**Chapter Index:** {chapter_index}")
            if start_line or end_line:
                lines.append(f"**Line Range:** {start_line or '—'} – {end_line or '—'}")

        def _edge_context_string(edge_data: dict[str, Any]) -> str:
            contexts = edge_data.get("context_counts") or edge_data.get("contexts") or []
            if not contexts:
                return ""
            if isinstance(contexts, list) and contexts and isinstance(contexts[0], tuple):
                return ", ".join(f"{ctx} ({count})" for ctx, count in contexts)
            if isinstance(contexts, list):
                return ", ".join(contexts)
            return str(contexts)

        if outgoing:
            lines.append("\n**Outgoing References:**")
            for dst in outgoing[:20]:
                edge_data = graph.get_edge_data(label, dst) or {}
                context_str = _edge_context_string(edge_data)
                context_suffix = f" — {context_str}" if context_str else ""
                lines.append(f"- `{dst}`{context_suffix}")
            if len(outgoing) > 20:
                lines.append(f"- … ({len(outgoing) - 20} more)")

        if incoming:
            lines.append("\n**Incoming References:**")
            for src in incoming[:20]:
                edge_data = graph.get_edge_data(src, label) or {}
                context_str = _edge_context_string(edge_data)
                context_suffix = f" — {context_str}" if context_str else ""
                lines.append(f"- `{src}`{context_suffix}")
            if len(incoming) > 20:
                lines.append(f"- … ({len(incoming) - 20} more)")

        summary = pn.pane.Markdown("\n".join(lines), sizing_mode="stretch_width")

        preprocess_dir = self._resolve_report_preprocess_dir()
        if preprocess_dir:
            try:
                report_md = render_label_report(label, preprocess_dir=preprocess_dir)
                report_pane = pn.pane.Markdown(report_md, sizing_mode="stretch_width")
            except Exception as exc:  # pragma: no cover - defensive
                log_message = (
                    "Failed to render preprocess report for %s (stage=%s, dir=%s): %s"
                )
                if stage == "directives":
                    logger.info(log_message, label, stage, preprocess_dir, exc)
                else:
                    logger.exception(log_message, label, stage, preprocess_dir, exc)
                report_pane = pn.pane.Markdown(
                    f"**Report unavailable:** {exc}",
                    sizing_mode="stretch_width",
                )
        else:
            report_pane = pn.pane.Markdown(
                "**Report unavailable:** No preprocess registry configured for reports.",
                sizing_mode="stretch_width",
            )

        directive_pane = None
        entry = self.registry_data.get(label)
        if entry:
            directive_markdown = entry.get("content_markdown") or entry.get("raw_directive")
            if isinstance(directive_markdown, str) and directive_markdown.strip():
                directive_pane = pn.pane.Markdown(
                    f"## Directive\n\n{directive_markdown.strip()}",
                    sizing_mode="stretch_width",
                    styles={"maxHeight": "600px", "overflow-y": "auto"},
                )

        panes = [summary, pn.layout.Divider(), report_pane]
        if directive_pane is not None:
            panes.extend([pn.layout.Divider(), directive_pane])

        return pn.Column(*panes, sizing_mode="stretch_width")

    def _create_entity_type_legend(self) -> pn.pane.HTML:
        """Create a legend showing entity type to color mapping.

        Returns:
            pn.pane.HTML: A Panel HTML pane with colored boxes and entity type labels.

        """
        graph = self.filtered_graph or self.label_graph

        if graph.number_of_nodes() == 0:
            return pn.pane.HTML(
                '<div style="font-family: sans-serif; color: #666;">No data</div>',
                width=220,
            )

        # Get unique entity types present in the current filtered graph
        entity_types_in_graph = sorted({
            data.get("entity_type", "unknown")
            for _, data in graph.nodes(data=True)
            if data.get("entity_type")
        })

        # Add unknown if there are nodes without entity_type
        has_unknown = any(not data.get("entity_type") for _, data in graph.nodes(data=True))
        if has_unknown and "unknown" not in entity_types_in_graph:
            entity_types_in_graph.append("unknown")

        # Build HTML legend
        legend_html = '<div style="font-family: sans-serif; font-size: 13px;">'
        legend_html += '<div style="font-weight: bold; margin-bottom: 12px; font-size: 14px;">Entity Types</div>'

        for entity_type in entity_types_in_graph:
            color = ENTITY_TYPE_COLORS.get(entity_type, DEFAULT_ENTITY_COLOR)
            display_name = entity_type.replace("_", " ").title()

            legend_html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="width: 18px; height: 18px; background-color: {color};
                            border: 1px solid #555; margin-right: 10px;
                            border-radius: 3px; flex-shrink: 0;">
                </div>
                <span style="line-height: 1.2;">{display_name}</span>
            </div>
            """

        legend_html += "</div>"
        return pn.pane.HTML(legend_html, width=220, sizing_mode="fixed")

    def _render_statistics(self):
        graph = self.filtered_graph or self.label_graph
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()

        top_labels = sorted(
            graph.nodes(data=True),
            key=lambda item: graph.degree(item[0]),
            reverse=True,
        )[:10]

        documents = {}
        for _, data in graph.nodes(data=True):
            doc = data.get("document_id")
            if doc:
                documents[doc] = documents.get(doc, 0) + 1

        lines = [
            "## Label Graph Statistics",
            f"**Nodes (labels):** {node_count}",
            f"**Edges (references):** {edge_count}",
        ]

        if top_labels:
            lines.append("\n**Top Labels by Degree:**")
            for label, _ in top_labels:
                lines.append(f"- `{label}`: degree {graph.degree(label)}")

        if documents:
            lines.append("\n**Labels by Document:**")
            for doc, count in sorted(documents.items()):
                lines.append(f"- `{doc}`: {count}")

        return pn.pane.Markdown("\n".join(lines), width=480)

    def _render_connectivity_report(self):
        graph = self.filtered_graph or self.label_graph
        if graph.number_of_nodes() == 0:
            return pn.pane.Markdown("**Connectivity report unavailable.**", width=480)

        isolated = sorted(node for node in graph.nodes() if graph.degree(node) == 0)
        sources = sorted(
            node
            for node in graph.nodes()
            if graph.in_degree(node) == 0 and graph.out_degree(node) > 0
        )
        leaves = sorted(
            node
            for node in graph.nodes()
            if graph.in_degree(node) > 0 and graph.out_degree(node) == 0
        )

        def _format_list(values: list[str], limit: int = 25) -> list[str]:
            if not values:
                return ["- *(none)*"]
            lines = [f"- `{value}`" for value in values[:limit]]
            if len(values) > limit:
                lines.append(f"- … ({len(values) - limit} more)")
            return lines

        lines = [
            "## Connectivity Report",
            f"**Isolated labels (no edges):** {len(isolated)}",
        ]
        lines.append("\n**Isolated Nodes:**")
        lines.extend(_format_list(isolated))

        lines.append(f"\n**Source labels (only outgoing):** {len(sources)}")
        lines.extend(_format_list(sources))

        lines.append(f"\n**Leaf labels (only incoming):** {len(leaves)}")
        lines.extend(_format_list(leaves))

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
            self.stage_toggle,
            self.filter_accordion,
            self.entity_type_filter,
            self.document_filter,
            self.search_input,
            self.min_degree_slider,
            self.min_edge_weight_slider,
            self.layout_selector,
            self.hover_columns_selector,
            self.reset_filters_button,
        ]

        main_content = [
            pn.Row(
                pn.Card(
                    pn.panel(self.graph_view),
                    title="Label Reference Graph",
                    sizing_mode="stretch_both",
                    min_height=600,
                ),
                pn.Card(
                    self.node_details_container,
                    title="Label Details",
                    sizing_mode="stretch_height",
                    width=1040,
                    min_height=600,
                    scroll=True,
                ),
            ),
            pn.Row(
                pn.Card(
                    pn.panel(self.stats_view),
                    title="Statistics",
                    width=375,
                    height=700,
                    scroll=True,
                ),
                pn.Card(
                    pn.panel(self.connectivity_view),
                    title="Connectivity Report",
                    width=375,
                    height=700,
                    scroll=True,
                ),
                pn.Card(
                    pn.panel(self.legend_view),
                    title="Legend",
                    width=250,
                    height=700,
                    scroll=True,
                ),
            ),
            pn.Card(
                self.dimension_controls,
                title="Graph Appearance Controls",
                sizing_mode="stretch_width",
                scroll=True,
            ),
            pn.Card(self.console_pane, title="Console", width=600, height=250, scroll=True),
        ]

        return pn.template.FastListTemplate(
            title="Label Reference Dashboard",
            sidebar=sidebar,
            main=main_content,
            accent_base_color="#9b59b6",
            header_background="#2c3e50",
        )


def main() -> pn.Template:
    return LabelReferenceDashboard().create_dashboard()


if __name__ == "__main__":
    template = main()
    pn.serve(template.servable(), port=5010, show=False)
elif not os.environ.get("MATHSTER_SKIP_DASHBOARD_INIT"):
    template = main()
    template.servable()
else:
    template = None
