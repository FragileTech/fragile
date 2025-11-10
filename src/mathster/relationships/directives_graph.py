"""Build directed label graphs from directives registry references.

This module mirrors :mod:`mathster.relationships.preprocess_graph` but
operates on the directive-stage registry (``registry/directives``). Each
directive entry is expected to expose a ``label`` along with an optional
``references`` field listing other labels cited inside the directive
content. The resulting graph is useful for visualizing document-local
dependency structure before preprocess normalization occurs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import networkx as nx


logger = logging.getLogger(__name__)

# Directive payloads currently expose a single reference-bearing field, but
# we keep the tuple form for forward compatibility.
REFERENCE_FIELDS: tuple[str, ...] = ("references",)


def build_label_reference_graph(directives_dir: Path | str) -> nx.DiGraph:
    """Return a directed graph that connects directive labels via references.

    Each node corresponds to a directive label (``def-*``, ``lem-*``, …)
    discovered inside ``directives_dir``. A directed edge ``label_a ->
    label_b`` is created whenever ``label_a`` cites ``label_b`` through one
    of the supported reference fields.

    Args:
        directives_dir: Path to the ``registry/directives`` directory that
            belongs to a specific document or aggregated registry.

    Returns:
        ``networkx.DiGraph`` populated with node/edge metadata. Node
        attributes include directive-specific information (type, section,
        chapter metadata). Edge attributes capture the contexts that produced
        each reference alongside multiplicity information.
    """

    registry = load_directives_registry(directives_dir)
    if not registry:
        logger.warning(
            "No directive entities available in %s; returning empty graph.",
            directives_dir,
        )
        return nx.DiGraph()

    label_graph = nx.DiGraph()

    def _add_node(label: str, entry: dict[str, Any] | None) -> None:
        if label_graph.has_node(label):
            return
        node_data = _build_label_node_attributes(entry)
        label_graph.add_node(label, **node_data)

    edge_instances: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    edge_context_counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(
        Counter
    )

    for label, entry in registry.items():
        _add_node(label, entry)

        reference_map = _collect_reference_contexts(entry)
        if not reference_map:
            continue

        for target_label, contexts in reference_map.items():
            if target_label == label:
                continue

            _add_node(target_label, registry.get(target_label))
            edge_key = (label, target_label)
            edge_instances[edge_key].extend(contexts)
            edge_context_counts[edge_key].update(contexts)

    for (source_label, target_label), contexts in edge_instances.items():
        context_counter = edge_context_counts[source_label, target_label]
        unique_contexts = sorted(context_counter.keys())
        context_counts = sorted(context_counter.items())

        label_graph.add_edge(
            source_label,
            target_label,
            weight=len(contexts),
            contexts=unique_contexts,
            context_counts=context_counts,
        )

    return label_graph


def load_directives_registry(directives_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Load directive entities from ``directives_dir``.

    Directive registry files contain metadata plus an ``items`` list with the
    actual directive entries. This helper normalizes the layout into a label
    keyed mapping that downstream consumers can traverse efficiently.

    Args:
        directives_dir: Directory holding directive JSON files.

    Returns:
        Mapping from directive label to the associated payload.
    """

    root = Path(directives_dir).expanduser()
    if not root.exists():
        logger.warning("Directive directory %s is missing.", directives_dir)
        return {}

    registry: dict[str, dict[str, Any]] = {}

    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Failed to load %s: %s", path, exc)
            continue

        items = payload.get("items")
        if not isinstance(items, list):
            logger.warning("Directive file %s missing 'items' list; skipping", path)
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if not isinstance(label, str) or not label.strip():
                continue
            if label in registry:
                logger.debug(
                    "Duplicate directive label %s encountered in %s; first instance preserved.",
                    label,
                    path,
                )
                continue
            registry[label] = item

    return registry


def _collect_reference_contexts(entry: dict[str, Any] | None) -> dict[str, list[str]]:
    """Gather referenced labels mapped to contexts responsible for them."""

    label_to_contexts: defaultdict[str, list[str]] = defaultdict(list)
    if not isinstance(entry, dict):
        return label_to_contexts

    for field in REFERENCE_FIELDS:
        for label in _normalize_reference_list(entry.get(field)):
            label_to_contexts[label].append(field)

    return label_to_contexts


def _normalize_reference_list(raw_refs: Any) -> set[str]:
    """Convert a raw reference payload into a normalized label set."""

    labels: set[str] = set()
    if raw_refs is None:
        return labels

    iterable: Iterable[Any]
    if isinstance(raw_refs, list | tuple | set):
        iterable = raw_refs
    else:
        iterable = [raw_refs]

    for item in iterable:
        label = _extract_label(item)
        if label:
            labels.add(label)
    return labels


def _extract_label(ref: Any) -> str | None:
    """Extract a label string from heterogeneous reference payloads."""

    if isinstance(ref, str):
        ref = ref.strip()
        return ref or None

    if isinstance(ref, dict):
        for key in ("label", "target", "id", "reference"):
            value = ref.get(key)
            if isinstance(value, str):
                value = value.strip()
                if value:
                    return value
    return None


def _build_label_node_attributes(entry: dict[str, Any] | None) -> dict[str, Any]:
    """Produce node metadata for a directive label."""

    base = {
        "directive_type": "unknown",
        "title": None,
        "section": None,
        "document_id": None,
        "chapter_index": None,
        "chapter_file": None,
        "start_line": None,
        "end_line": None,
    }

    if not isinstance(entry, dict):
        return base

    def _normalize_str(value: Any) -> str | None:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return None

    registry_ctx = entry.get("_registry_context") or {}
    if isinstance(registry_ctx, dict):
        document_id = _normalize_str(registry_ctx.get("document_id"))
        if document_id:
            base["document_id"] = document_id
        chapter_index = registry_ctx.get("chapter_index")
        if isinstance(chapter_index, int):
            base["chapter_index"] = chapter_index
        base["chapter_file"] = _normalize_str(registry_ctx.get("chapter_file"))

    base["directive_type"] = _normalize_str(entry.get("directive_type")) or "unknown"
    base["title"] = _normalize_str(entry.get("title"))
    base["section"] = _normalize_str(entry.get("section"))
    base["start_line"] = entry.get("start_line")
    base["end_line"] = entry.get("end_line")

    return base


__all__ = [
    "build_label_reference_graph",
    "load_directives_registry",
]
