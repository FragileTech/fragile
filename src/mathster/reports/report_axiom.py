"""Render :class:`UnifiedAxiom` objects into Markdown reports."""

from __future__ import annotations

from typing import Any, Iterable

import json

from mathster.preprocess_extraction.data_models import (
    FailureMode,
    Hypothesis,
    Implication,
    Parameter,
    Span,
    UnifiedAxiom,
)

__all__ = ["unified_axiom_to_markdown"]


def unified_axiom_to_markdown(axiom: UnifiedAxiom) -> str:
    """Return a Markdown report describing ``axiom``."""

    sections: list[str] = []

    reference_labels = _format_reference_labels(axiom.references)
    reference_line = (
        f"**Reference labels:** {reference_labels}" if reference_labels else "**Reference labels:** _none_"
    )
    sections.append(reference_line)

    sections.append(_build_header_block(axiom))

    if axiom.nl_summary:
        sections.append(_section("Natural Language Summary", axiom.nl_summary.strip()))

    if axiom.core_statement:
        sections.append(_section("Core Statement", _format_core_statement(axiom)))

    if axiom.hypotheses:
        sections.append(_section("Hypotheses", _format_hypotheses(axiom.hypotheses)))

    if axiom.implications:
        sections.append(_section("Implications", _format_implications(axiom.implications)))

    if axiom.parameters:
        sections.append(_section("Parameters", _format_parameters(axiom.parameters)))

    if axiom.failure_modes:
        sections.append(_section("Failure Modes", _format_failure_modes(axiom.failure_modes)))

    if axiom.tags:
        sections.append(_section("Tags", ", ".join(f"`{tag}`" for tag in axiom.tags)))

    if axiom.content_markdown:
        sections.append(
            _section(
                "Directive Content",
                f"```markdown\n{axiom.content_markdown.strip()}\n```",
            )
        )

    metadata_block = _format_metadata(axiom.metadata, axiom.registry_context)
    if metadata_block:
        sections.append(_section("Metadata", metadata_block))

    combined = "\n\n".join(part for part in sections if part).strip()
    return combined + "\n"


def _build_header_block(axiom: UnifiedAxiom) -> str:
    title = axiom.title or axiom.label
    lines = [f"# {title}"]
    if axiom.label:
        lines.append(f"**Label:** `{axiom.label}`")
    lines.append(f"**Type:** {axiom.type}")
    if axiom.axiom_class:
        lines.append(f"**Class:** {axiom.axiom_class}")

    info_parts: list[str] = []
    if axiom.document_id:
        info_parts.append(f"Document: `{axiom.document_id}`")
    if axiom.section:
        info_parts.append(f"Section: {axiom.section}")
    span = _format_span(axiom.span)
    if span:
        info_parts.append(span)
    if axiom.generated_at:
        info_parts.append(f"Generated at: {axiom.generated_at}")
    if axiom.alt_labels:
        info_parts.append(
            "Alternate labels: " + ", ".join(f"`{label}`" for label in axiom.alt_labels)
        )
    if info_parts:
        lines.append(" • ".join(info_parts))
    return "\n".join(lines)


def _format_core_statement(axiom: UnifiedAxiom) -> str:
    parts: list[str] = []
    core = axiom.core_statement
    if core:
        if core.text:
            parts.append(core.text.strip())
        if core.latex:
            parts.append(_wrap_math(core.latex))
    return "\n\n".join(parts) if parts else "—"


def _format_hypotheses(hypotheses: list[Hypothesis]) -> str:
    lines: list[str] = []
    for idx, hyp in enumerate(hypotheses, start=1):
        pieces: list[str] = []
        if hyp.text:
            pieces.append(hyp.text.strip())
        if hyp.latex:
            pieces.append(_wrap_math(hyp.latex))
        content = " ".join(pieces) if pieces else f"Hypothesis {idx}"
        lines.append(f"- {content}")
    return "\n".join(lines)


def _format_implications(implications: list[Implication]) -> str:
    lines: list[str] = []
    for idx, imp in enumerate(implications, start=1):
        pieces: list[str] = []
        if imp.text:
            pieces.append(imp.text.strip())
        if imp.latex:
            pieces.append(_wrap_math(imp.latex))
        content = " ".join(pieces) if pieces else f"Implication {idx}"
        lines.append(f"- {content}")
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


def _format_failure_modes(failure_modes: list[FailureMode]) -> str:
    lines: list[str] = []
    for mode in failure_modes:
        description = mode.description.strip() if mode.description else "—"
        impact = f" (impact: {mode.impact})" if mode.impact else ""
        lines.append(f"- {description}{impact}")
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

