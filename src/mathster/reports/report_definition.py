"""Render :class:`UnifiedDefinition` objects into Markdown reports."""

from __future__ import annotations

import json
from typing import Any, Iterable

from mathster.preprocess_extraction.data_models import (
    Condition,
    Example,
    NamedProperty,
    Note,
    Parameter,
    Span,
    UnifiedDefinition,
)


__all__ = ["unified_definition_to_markdown"]


def unified_definition_to_markdown(definition: UnifiedDefinition) -> str:
    """Return a comprehensive Markdown report for ``definition``."""

    sections: list[str] = []

    reference_labels = _format_reference_labels(definition.references)
    reference_line = (
        f"**Reference labels:** {reference_labels}"
        if reference_labels
        else "**Reference labels:** _none_"
    )
    sections.extend((reference_line, _build_header_block(definition)))

    term_block = _format_term(definition)
    if term_block:
        sections.append(_section("Term", term_block))

    if definition.nl_definition:
        sections.append(_section("Natural Language Definition", definition.nl_definition.strip()))

    if definition.formal_conditions:
        sections.append(
            _section("Formal Conditions", _format_conditions(definition.formal_conditions))
        )

    if definition.properties:
        sections.append(_section("Named Properties", _format_properties(definition.properties)))

    if definition.parameters:
        sections.append(_section("Parameters", _format_parameters(definition.parameters)))

    if definition.examples:
        sections.append(_section("Examples", _format_examples(definition.examples)))

    if definition.notes:
        sections.append(_section("Notes", _format_notes(definition.notes)))

    if definition.related_refs:
        sections.append(
            _section("Related References", _format_bullet_list(definition.related_refs))
        )

    if definition.tags:
        sections.append(_section("Tags", ", ".join(f"`{tag}`" for tag in definition.tags)))

    if definition.content_markdown:
        sections.append(
            _section(
                "Directive Content",
                f"```markdown\n{definition.content_markdown.strip()}\n```",
            )
        )

    metadata_block = _format_metadata(definition.metadata, definition.registry_context)
    if metadata_block:
        sections.append(_section("Metadata", metadata_block))

    combined = "\n\n".join(part for part in sections if part).strip()
    return combined + "\n"


def _build_header_block(definition: UnifiedDefinition) -> str:
    title = definition.title or definition.term or definition.label
    lines = [f"# {title}"]
    if definition.label:
        lines.append(f"**Label:** `{definition.label}`")
    lines.append(f"**Type:** {definition.type}")

    info_parts: list[str] = []
    if definition.document_id:
        info_parts.append(f"Document: `{definition.document_id}`")
    if definition.section:
        info_parts.append(f"Section: {definition.section}")
    span = _format_span(definition.span)
    if span:
        info_parts.append(span)
    if definition.generated_at:
        info_parts.append(f"Generated at: {definition.generated_at}")
    if definition.alt_labels:
        info_parts.append(
            "Alternate labels: " + ", ".join(f"`{label}`" for label in definition.alt_labels)
        )
    if info_parts:
        lines.append(" • ".join(info_parts))
    return "\n".join(lines)


def _format_term(definition: UnifiedDefinition) -> str | None:
    parts: list[str] = []
    if definition.term:
        parts.append(f"**Term:** {definition.term}")
    if definition.object_type:
        parts.append(f"**Object type:** {definition.object_type}")
    return "\n\n".join(parts) if parts else None


def _format_conditions(conditions: list[Condition]) -> str:
    lines: list[str] = []
    for idx, condition in enumerate(conditions, start=1):
        pieces: list[str] = []
        if condition.type:
            pieces.append(f"[{condition.type}]")
        if condition.text:
            pieces.append(condition.text.strip())
        if condition.latex:
            pieces.append(_wrap_math(condition.latex))
        content = " ".join(pieces) if pieces else f"Condition {idx}"
        lines.append(f"- {content}")
    return "\n".join(lines)


def _format_properties(properties: list[NamedProperty]) -> str:
    lines: list[str] = []
    for prop in properties:
        name = prop.name or "Property"
        description = prop.description.strip() if prop.description else "—"
        lines.append(f"- **{name}:** {description}")
    return "\n".join(lines)


def _format_parameters(parameters: list[Parameter]) -> str:
    lines: list[str] = []
    for param in parameters:
        parts: list[str] = []
        if param.symbol:
            parts.append(f"`{param.symbol}`")
        if param.name:
            parts.append(param.name)
        header = " - ".join(parts) if parts else "Parameter"
        details: list[str] = []
        if param.description:
            details.append(param.description.strip())
        if param.constraints:
            details.append("Constraints: " + ", ".join(param.constraints))
        if param.tags:
            details.append("Tags: " + ", ".join(f"`{tag}`" for tag in param.tags))
        lines.append(f"- **{header}:** {' '.join(details) if details else '—'}")
    return "\n".join(lines)


def _format_examples(examples: list[Example]) -> str:
    blocks: list[str] = []
    for idx, example in enumerate(examples, start=1):
        parts: list[str] = [f"**Example {idx}:**"]
        if example.text:
            parts.append(example.text.strip())
        if example.latex:
            parts.append(_wrap_math(example.latex))
        blocks.append("\n\n".join(parts))
    return "\n\n".join(blocks)


def _format_notes(notes: list[Note]) -> str:
    lines: list[str] = []
    for note in notes:
        prefix = f"[{note.type}] " if note.type else ""
        text = note.text.strip() if note.text else "—"
        lines.append(f"- {prefix}{text}")
    return "\n".join(lines)


def _format_reference_labels(references: list[Any]) -> str | None:
    labels: list[str] = []
    for entry in references:
        label = _extract_label_from_reference(entry)
        if label:
            labels.append(f"`{label}`")
    if not labels:
        return None
    return ", ".join(labels)


def _extract_label_from_reference(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry.strip() or None
    if isinstance(entry, dict):
        for key in ("label", "target", "id", "ref", "reference"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        title = entry.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    return None


def _format_span(span: Span | None) -> str | None:
    if not span:
        return None
    items: list[str] = []
    if span.start_line is not None or span.end_line is not None:
        start = span.start_line if span.start_line is not None else "?"
        end = span.end_line if span.end_line is not None else "?"
        items.append(f"Lines {start}–{end}")
    if span.content_start is not None or span.content_end is not None:
        start = span.content_start if span.content_start is not None else "?"
        end = span.content_end if span.content_end is not None else "?"
        items.append(f"Content {start}–{end}")
    if span.header_lines:
        items.append("Headers: " + ", ".join(str(num) for num in span.header_lines))
    return "; ".join(items) if items else None


def _section(title: str, body: str | None) -> str | None:
    if not body:
        return None
    return f"## {title}\n\n{body.strip()}"


def _format_bullet_list(items: Iterable[str]) -> str | None:
    clean = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    if not clean:
        return None
    return "\n".join(f"- {item}" for item in clean)


def _wrap_math(content: str) -> str:
    stripped = content.strip()
    if "\n" in stripped:
        return f"$${stripped}$$"
    return f"${stripped}$"


def _format_metadata(metadata: dict[str, Any], registry_context: dict[str, Any]) -> str | None:
    if not metadata and not registry_context:
        return None
    payload: dict[str, Any] = {}
    if metadata:
        payload["metadata"] = metadata
    if registry_context:
        payload["registry_context"] = registry_context
    return f"```json\n{json.dumps(payload, indent=2, sort_keys=True)}\n```"
