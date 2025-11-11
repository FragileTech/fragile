#!/usr/bin/env python3
"""CLI utility that reports document-level connectivity in markdown."""

from __future__ import annotations

import argparse
from collections import defaultdict
import logging
from pathlib import Path
from typing import Any, Iterable, Literal, NamedTuple

from mathster.iterators import discover_registry_folders
from mathster.paths import UNIFIED_DIRECTIVES_PATH, UNIFIED_PREPROCESS_PATH
from mathster.relationships.directives_graph import (
    build_label_reference_graph as build_directives_label_reference_graph,
    load_directives_registry,
)
from mathster.relationships.preprocess_graph import (
    build_label_reference_graph as build_preprocess_label_reference_graph,
    load_preprocess_registry,
)


logger = logging.getLogger(__name__)


class _RegistryLoadResult(NamedTuple):
    source: Literal["preprocess", "directives"]
    path: Path
    registry: dict[str, dict[str, Any]]
    doc_entries: dict[str, dict[str, Any]]


def _normalize_str(value: object) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def _entry_document_id(entry: dict) -> str | None:
    if not isinstance(entry, dict):
        return None
    registry_ctx_candidates = []
    for key in ("registry_context", "_registry_context"):
        ctx = entry.get(key)
        if isinstance(ctx, dict):
            registry_ctx_candidates.append(ctx)
    registry_ctx = registry_ctx_candidates[0] if registry_ctx_candidates else {}
    return (
        _normalize_str(entry.get("document_id"))
        or _normalize_str(entry.get("document"))
        or _normalize_str(entry.get("source_document"))
        or _normalize_str(registry_ctx.get("document_id"))
    )


def _resolve_registry_path(
    document_id: str,
    preferred_dir: str | Path | None,
    subfolder: str,
    fallback_path: Path,
) -> Path | None:
    """Resolve a registry directory for ``document_id`` and ``subfolder``."""

    if preferred_dir:
        path = Path(preferred_dir).expanduser()
        return path if path.exists() else None

    discovered = discover_registry_folders(subfolder=subfolder, document=document_id)
    candidate: Path | None = None
    if discovered:
        exact = [path for path in discovered if path.parent.parent.name == document_id]
        candidate = exact[0] if exact else discovered[0]
        candidate = Path(candidate).expanduser()
        if candidate.exists():
            return candidate

    fallback_path = Path(fallback_path).expanduser()
    if fallback_path.exists():
        return fallback_path
    return None


def _filter_entries_by_document(
    registry: dict[str, dict[str, Any]],
    document_id: str,
) -> dict[str, dict[str, Any]]:
    return {
        label: entry
        for label, entry in registry.items()
        if _entry_document_id(entry) == document_id
    }


def _determine_entity_type(entry: dict[str, Any]) -> str:
    return (
        _normalize_str(entry.get("entity_type"))
        or _normalize_str(entry.get("type"))
        or _normalize_str(entry.get("directive_type"))
        or "unknown"
    )


def _load_document_registry(
    document_id: str,
    preprocess_dir: str | Path | None,
) -> _RegistryLoadResult:
    """Load registry data for ``document_id``, falling back to directives."""

    preprocess_path = _resolve_registry_path(
        document_id=document_id,
        preferred_dir=preprocess_dir,
        subfolder="preprocess",
        fallback_path=UNIFIED_PREPROCESS_PATH,
    )

    if preprocess_path:
        registry = load_preprocess_registry(preprocess_path)
        doc_entries = _filter_entries_by_document(registry, document_id)
        if doc_entries:
            return _RegistryLoadResult(
                source="preprocess",
                path=preprocess_path,
                registry=registry,
                doc_entries=doc_entries,
            )
        logger.info(
            "Preprocess registry %s does not contain entries for document '%s'; "
            "attempting directives fallback.",
            preprocess_path,
            document_id,
        )

    directives_path = _resolve_registry_path(
        document_id=document_id,
        preferred_dir=None,
        subfolder="directives",
        fallback_path=UNIFIED_DIRECTIVES_PATH,
    )

    if not directives_path:
        if preprocess_path:
            raise FileNotFoundError(
                f"Document '{document_id}' not found in preprocess registry "
                f"('{preprocess_path}') and no directives registry is available."
            )
        raise FileNotFoundError(
            f"No preprocess registry found for '{document_id}', "
            "and directives registry is unavailable."
        )

    registry = load_directives_registry(directives_path)
    doc_entries = _filter_entries_by_document(registry, document_id)
    if not doc_entries:
        raise ValueError(
            f"No directive entries found for document '{document_id}' in '{directives_path}'."
        )

    logger.info(
        "Using directives registry at %s for document '%s'.",
        directives_path,
        document_id,
    )
    return _RegistryLoadResult(
        source="directives",
        path=directives_path,
        registry=registry,
        doc_entries=doc_entries,
    )


def _normalize_tags(raw_tags: object) -> list[str]:
    """Return a sanitized list of tag strings."""

    if raw_tags is None:
        return []
    if isinstance(raw_tags, list | tuple | set):
        iterable = raw_tags
    else:
        iterable = [raw_tags]

    tags: list[str] = []
    for tag in iterable:
        if isinstance(tag, str):
            clean = tag.strip()
            if clean:
                tags.append(clean)
    return tags


def _populate_node_metadata(graph, doc_entries: dict[str, dict[str, Any]]) -> None:
    """Ensure each node carries normalized metadata derived from doc entries."""

    for label, entry in doc_entries.items():
        if label not in graph:
            continue
        node_data = graph.nodes[label]
        node_data["entity_type"] = _normalize_str(
            node_data.get("entity_type")
        ) or _determine_entity_type(entry)
        node_data["title"] = (
            node_data.get("title")
            or _normalize_str(entry.get("title"))
            or _normalize_str(entry.get("term"))
        )
        node_data["document_id"] = _entry_document_id(entry)
        node_data["section"] = node_data.get("section") or _normalize_str(entry.get("section"))
        node_data["tags"] = _normalize_tags(entry.get("tags"))


def build_document_connectivity_graph(
    document_id: str,
    preprocess_dir: str | Path | None = None,
):
    """Return the label connectivity graph for a single document.

    The loader prefers preprocess registry data (from the document workspace
    or the unified registry). When preprocess data is unavailable, it falls
    back to directive registries to provide the same connectivity summary.
    """

    document_id = (document_id or "").strip()
    if not document_id:
        msg = "Document identifier must be a non-empty string."
        raise ValueError(msg)

    registry_view = _load_document_registry(document_id, preprocess_dir)

    if registry_view.source == "preprocess":
        graph = build_preprocess_label_reference_graph(registry_view.path)
    else:
        graph = build_directives_label_reference_graph(registry_view.path)

    doc_entries = registry_view.doc_entries
    doc_graph = graph.subgraph(doc_entries.keys()).copy()

    missing_nodes = set(doc_entries) - set(doc_graph.nodes)
    for label in missing_nodes:
        entry = doc_entries[label]
        doc_graph.add_node(
            label,
            entity_type=_normalize_str(entry.get("type")) or "unknown",
            title=_normalize_str(entry.get("title")) or _normalize_str(entry.get("term")),
            document_id=_entry_document_id(entry),
            section=_normalize_str(entry.get("section")),
            tags=_normalize_tags(entry.get("tags")),
        )

    _populate_node_metadata(doc_graph, doc_entries)

    return doc_graph


def build_document_connectivity_report_from_graph(
    document_id: str,
    graph,
) -> str:
    """Return a markdown report for the provided document graph."""
    document_id = (document_id or "").strip()
    if not document_id:
        msg = "Document identifier must be a non-empty string."
        raise ValueError(msg)

    doc_nodes = sorted(graph.nodes())
    if not doc_nodes:
        raise ValueError(f"Connectivity graph for document '{document_id}' contains no nodes.")

    type_groups: defaultdict[str, list[str]] = defaultdict(list)
    definition_tags: dict[str, list[str]] = {}

    for label in doc_nodes:
        node_data = graph.nodes[label]
        entity_type = (_normalize_str(node_data.get("entity_type")) or "unknown").lower()
        type_groups[entity_type].append(label)
        if entity_type == "definition":
            definition_tags[label] = _normalize_tags(node_data.get("tags"))

    def _filter_labels(predicate) -> list[str]:
        return sorted(label for label in doc_nodes if predicate(graph, label))

    isolated = _filter_labels(lambda g, n: g.degree(n) == 0)
    sources = _filter_labels(lambda g, n: g.in_degree(n) == 0 and g.out_degree(n) > 0)
    sinks = _filter_labels(lambda g, n: g.in_degree(n) > 0 and g.out_degree(n) == 0)
    connected = _filter_labels(lambda g, n: g.in_degree(n) > 0 and g.out_degree(n) > 0)

    lines: list[str] = []
    lines.extend((
        f"# Document Connectivity Report: `{document_id}`\n",
        f"- **Entities in document:** {len(doc_nodes)}",
        f"- **Definitions:** {len(type_groups.get('definition', []))}",
        f"- **Theorems:** {len(type_groups.get('theorem', []))}",
        f"- **Lemmas:** {len(type_groups.get('lemma', []))}",
        f"- **Propositions:** {len(type_groups.get('proposition', []))}",
        f"- **Corollaries:** {len(type_groups.get('corollary', []))}",
        "",
        "## Connectivity Summary",
    ))
    lines.extend(_format_category("Isolated labels", isolated))
    lines.extend(_format_category("Sources (outgoing only)", sources))
    lines.extend(_format_category("Leaves (incoming only)", sinks))
    lines.extend(_format_category("Bidirectional (incoming & outgoing)", connected))
    lines.extend(("", "## Definitions & Tags"))
    if definition_tags:
        for label in sorted(definition_tags):
            tag_list = definition_tags[label]
            tag_str = ", ".join(tag_list) if tag_list else "(no tags)"
            lines.append(f"- **definition** `{label}`: {tag_str}")
    else:
        lines.append("- No definitions found.")
    lines.extend(("", "## Entities by Type"))
    for entity_type in sorted(type_groups):
        labels = ", ".join(f"`{label}`" for label in sorted(type_groups[entity_type]))
        label_str = labels or "(none)"
        lines.append(f"- **{entity_type}**: {label_str}")

    return "\n".join(lines).strip() + "\n"


def generate_document_connectivity_report(
    document_id: str,
    preprocess_dir: str | Path | None = None,
) -> str:
    """Return a markdown report describing connectivity for a document."""
    graph = build_document_connectivity_graph(
        document_id=document_id,
        preprocess_dir=preprocess_dir,
    )
    return build_document_connectivity_report_from_graph(document_id, graph)


def _format_category(title: str, labels: Iterable[str]) -> list[str]:
    label_list = sorted(labels)
    count = len(label_list)
    lines = ["", f"### {title} ({count})"]
    if count == 0:
        lines.append("- (none)")
        return lines
    lines.extend(f"- `{label}`" for label in label_list)
    return lines


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a markdown connectivity report for a single document "
            "using preprocess registry data (falling back to directives when necessary)."
        )
    )
    parser.add_argument(
        "document",
        help="Document identifier (document_id) to analyze.",
    )
    parser.add_argument(
        "--preprocess-dir",
        help=(
            "Directory that contains preprocess JSON files. When omitted (or missing), "
            "the CLI attempts to locate the document-specific registry, falling back to "
            "<project_root>/unified_registry/preprocess> or the directives registry."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = generate_document_connectivity_report(
        document_id=args.document,
        preprocess_dir=args.preprocess_dir,
    )
    print(report)


if __name__ == "__main__":
    main()
