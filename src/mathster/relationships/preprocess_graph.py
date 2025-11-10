"""Build directed tag graphs from preprocess registry references.

This module inspects the unified preprocess registry (definition.json,
lemma.json, etc.) and constructs a directed graph where each node
represents a tag and each edge indicates that a tag from one entity cites
another tag through an explicit reference.

Usage example::

    from pathlib import Path
    from mathster.relationships.preprocess_graph import build_tag_reference_graph

    graph = build_tag_reference_graph(Path("unified_registry/preprocess"))
    print(graph.number_of_nodes(), graph.number_of_edges())
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import networkx as nx


logger = logging.getLogger(__name__)


# Files that make up the preprocess registry. proof.json intentionally
# omitted because the list of proofs can be large and the statements they
# prove already appear in the other files.
PREPROCESS_ENTITY_FILES: tuple[str, ...] = (
    "algorithm.json",
    "axiom.json",
    "corollary.json",
    "definition.json",
    "lemma.json",
    "proposition.json",
    "remark.json",
    "theorem.json",
)

# Reference-like fields that may contain labels pointing to other entries.
REFERENCE_FIELDS: tuple[str, ...] = (
    "references",
    "local_refs",
    "related_refs",
)


def build_tag_reference_graph(preprocess_dir: Path | str) -> nx.DiGraph:
    """Return a directed tag graph derived from preprocess references.

    Nodes correspond to tags attached to preprocess entities. A directed
    edge ``tag_a -> tag_b`` exists when any entity tagged with ``tag_a``
    cites (via references/local_refs/proof references) an entity tagged
    with ``tag_b``. The edge weight equals the number of unique
    (source-label, target-label) reference pairs that produced the tag
    connection.

    Args:
        preprocess_dir: Directory that contains preprocess JSON files
            (definition.json, lemma.json, etc.).

    Returns:
        ``networkx.DiGraph`` with node/edge metadata:
            - Node attributes: ``frequency`` (#entities carrying the tag),
              ``labels`` (sorted list of entity labels).
            - Edge attributes: ``weight`` (#label pairs contributing) and
              ``label_pairs`` (sorted list of ``(source, target)`` tuples).
    """

    registry = load_preprocess_registry(preprocess_dir)
    if not registry:
        logger.warning("No preprocess entities found in directory %s", preprocess_dir)
        return nx.DiGraph()

    label_to_tags: dict[str, tuple[str, ...]] = {}
    tag_to_labels: defaultdict[str, set[str]] = defaultdict(set)

    for label, entry in registry.items():
        tags = _collect_tags(entry)
        if not tags:
            continue
        label_to_tags[label] = tuple(tags)
        for tag in tags:
            tag_to_labels[tag].add(label)

    if not label_to_tags:
        logger.info(
            "Preprocess registry %s contains no tag data; returning empty graph.",
            preprocess_dir,
        )
        return nx.DiGraph()

    tag_graph = nx.DiGraph()
    for tag, labels in tag_to_labels.items():
        tag_graph.add_node(
            tag,
            frequency=len(labels),
            labels=sorted(labels),
        )

    edge_weights: Counter[tuple[str, str]] = Counter()
    edge_examples: defaultdict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)

    for source_label, entry in registry.items():
        source_tags = label_to_tags.get(source_label)
        if not source_tags:
            continue

        referenced_labels = _collect_reference_labels(entry)
        if not referenced_labels:
            continue

        for target_label in referenced_labels:
            if target_label == source_label:
                continue  # Ignore self references
            target_tags = label_to_tags.get(target_label)
            if not target_tags:
                continue

            for src_tag in source_tags:
                for tgt_tag in target_tags:
                    edge_key = (src_tag, tgt_tag)
                    edge_weights[edge_key] += 1
                    edge_examples[edge_key].add((source_label, target_label))

    for (src_tag, tgt_tag), weight in edge_weights.items():
        examples = sorted(edge_examples[src_tag, tgt_tag])
        tag_graph.add_edge(
            src_tag,
            tgt_tag,
            weight=weight,
            label_pairs=examples,
        )

    return tag_graph


def build_label_reference_graph(preprocess_dir: Path | str) -> nx.DiGraph:
    """Return a directed graph that connects labels via explicit references.

    Each node in the graph corresponds to a unique entity label (e.g.,
    ``def-*``, ``thm-*``) found in the preprocess registry. A directed edge
    ``label_a -> label_b`` is added whenever the entity with label
    ``label_a`` cites ``label_b`` through any of the reference-bearing
    fields (``references``, ``local_refs``, ``related_refs``) or within the
    proof structure.

    Node attributes expose lightweight metadata about the entity (type,
    title, document identifier, tags). Edge attributes capture how many
    reference instances produced the connection, along with the specific
    contexts (e.g., ``references``, ``proof.steps[2].references``) in which
    the reference appeared.
    """

    registry = load_preprocess_registry(preprocess_dir)
    if not registry:
        logger.warning(
            "No preprocess entities available in %s; returning empty graph.",
            preprocess_dir,
        )
        return nx.DiGraph()

    label_graph = nx.DiGraph()

    def _add_node_from_entry(label: str, entry: dict[str, Any] | None) -> None:
        if label_graph.has_node(label):
            return
        node_data = _build_label_node_attributes(entry)
        label_graph.add_node(label, **node_data)

    edge_instances: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    edge_context_counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for label, entry in registry.items():
        _add_node_from_entry(label, entry)

        reference_map = _collect_reference_contexts(entry)
        if not reference_map:
            continue

        for target_label, contexts in reference_map.items():
            if target_label == label:
                continue

            _add_node_from_entry(target_label, registry.get(target_label))
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


def load_preprocess_registry(preprocess_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Load all preprocess entities under ``preprocess_dir``.

    Args:
        preprocess_dir: Directory containing preprocess JSON files.

    Returns:
        Mapping from entity label to the underlying dictionary payload.
        Non-dict payloads are ignored. Later duplicates keep the first
        occurrence, but a warning gets emitted for visibility.
    """

    root = Path(preprocess_dir).expanduser()
    registry: dict[str, dict[str, Any]] = {}

    for filename in PREPROCESS_ENTITY_FILES:
        path = root / filename
        if not path.exists():
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load %s: %s", path, exc)
            continue

        if not isinstance(payload, list):
            logger.warning("Expected list payload in %s; skipping", path)
            continue

        for item in payload:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if not isinstance(label, str) or not label.strip():
                continue
            if label in registry:
                logger.debug(
                    "Duplicate preprocess label %s encountered in %s; first instance preserved.",
                    label,
                    path,
                )
                continue
            registry[label] = item

    return registry


def _collect_tags(entry: dict[str, Any]) -> list[str]:
    """Return a normalized list of tags for an entry."""

    tags = entry.get("tags") or []
    normalized: list[str] = []
    for tag in tags:
        if isinstance(tag, str):
            clean = tag.strip()
            if clean:
                normalized.append(clean)
    return normalized


def _collect_reference_labels(entry: dict[str, Any]) -> set[str]:
    """Gather referenced labels from an entry and its proof structure."""

    references: set[str] = set()

    for field in REFERENCE_FIELDS:
        references.update(_normalize_reference_list(entry.get(field)))

    proof = entry.get("proof")
    if isinstance(proof, dict):
        references.update(_normalize_reference_list(proof.get("references")))
        for step in proof.get("steps") or []:
            if isinstance(step, dict):
                references.update(_normalize_reference_list(step.get("references")))

    return references


def _collect_reference_contexts(entry: dict[str, Any]) -> dict[str, list[str]]:
    """Gather referenced labels mapped to the contexts that produced them."""

    label_to_contexts: defaultdict[str, list[str]] = defaultdict(list)

    for field in REFERENCE_FIELDS:
        for label in _normalize_reference_list(entry.get(field)):
            label_to_contexts[label].append(field)

    proof = entry.get("proof")
    if isinstance(proof, dict):
        for label in _normalize_reference_list(proof.get("references")):
            label_to_contexts[label].append("proof.references")

        for idx, step in enumerate(proof.get("steps") or []):
            if not isinstance(step, dict):
                continue
            descriptor = f"proof.steps[{idx}].references"
            for label in _normalize_reference_list(step.get("references")):
                label_to_contexts[label].append(descriptor)

    return label_to_contexts


def _normalize_reference_list(raw_refs: Any) -> set[str]:
    """Convert a raw reference field into a set of labels."""

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
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _build_label_node_attributes(entry: dict[str, Any] | None) -> dict[str, Any]:
    """Create node metadata for a label-driven graph."""

    if not isinstance(entry, dict):
        return {
            "entity_type": "unknown",
            "title": None,
            "document_id": None,
            "section": None,
            "tags": [],
        }

    def _normalize_str(value: Any) -> str | None:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return None

    registry_ctx = entry.get("registry_context")
    document_hint = None
    if isinstance(registry_ctx, dict):
        for key in ("document_id", "document"):
            document_hint = _normalize_str(registry_ctx.get(key))
            if document_hint:
                break

    document_id = (
        _normalize_str(entry.get("document_id"))
        or _normalize_str(entry.get("document"))
        or _normalize_str(entry.get("source_document"))
        or document_hint
    )

    return {
        "entity_type": _normalize_str(entry.get("type")) or "unknown",
        "title": _normalize_str(entry.get("title")) or _normalize_str(entry.get("term")),
        "document_id": document_id,
        "section": _normalize_str(entry.get("section")),
        "tags": _collect_tags(entry),
    }


__all__ = [
    "build_label_reference_graph",
    "build_tag_reference_graph",
    "load_preprocess_registry",
]
