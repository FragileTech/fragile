"""Markdown renderer for :class:`UnifiedRemark`."""

from __future__ import annotations

from typing import Any

from mathster.preprocess_extraction.data_models import UnifiedRemark
from mathster.reports.report_utils import (
    format_bullet_list,
    format_metadata,
    format_reference_labels,
    make_section,
)

__all__ = ["unified_remark_to_markdown"]


def unified_remark_to_markdown(remark: UnifiedRemark) -> str:
    """Return a Markdown report describing ``remark``."""

    sections: list[str] = []

    reference_labels = format_reference_labels(remark.references)
    reference_line = (
        f"**Reference labels:** {reference_labels}" if reference_labels else "**Reference labels:** _none_"
    )
    sections.append(reference_line)

    sections.append(_build_header_block(remark))

    if remark.nl_summary:
        sections.append(make_section("Summary", remark.nl_summary.strip()))

    if remark.key_points:
        sections.append(make_section("Key Points", _format_key_points(remark.key_points)))

    if remark.quantitative_notes:
        sections.append(
            make_section("Quantitative Notes", _format_quantitative_notes(remark.quantitative_notes))
        )

    if remark.recommendations:
        sections.append(make_section("Recommendations", _format_recommendations(remark.recommendations)))

    if remark.dependencies:
        sections.append(make_section("Dependencies", format_bullet_list(remark.dependencies)))

    if remark.tags:
        sections.append(make_section("Tags", ", ".join(f"`{tag}`" for tag in remark.tags)))

    if remark.content:
        sections.append(
            make_section(
                "Directive Content",
                f"```markdown\n{remark.content.strip()}\n```",
            )
        )

    metadata_block = format_metadata(_maybe_model_dump(remark.metadata), _maybe_model_dump(remark.registry_context))
    if metadata_block:
        sections.append(make_section("Metadata", metadata_block))

    combined = "\n\n".join(part for part in sections if part).strip()
    return combined + "\n"


def _build_header_block(remark: UnifiedRemark) -> str:
    title = remark.title or remark.label
    lines = [f"# {title}"]
    if remark.label:
        lines.append(f"**Label:** `{remark.label}`")
    if remark.remark_type:
        lines.append(f"**Type:** {remark.remark_type}")
    if remark.section:
        lines.append(f"Section: {remark.section}")
    line_range = _format_line_range(remark.start_line, remark.end_line)
    if line_range:
        lines.append(line_range)
    return "\n".join(lines)


def _format_line_range(start: int | None, end: int | None) -> str | None:
    if start is None and end is None:
        return None
    start_val = start if start is not None else "?"
    end_val = end if end is not None else "?"
    return f"Lines {start_val}–{end_val}"


def _format_key_points(points: Any) -> str:
    lines: list[str] = []
    for idx, point in enumerate(points, start=1):
        text = getattr(point, "text", None)
        latex = getattr(point, "latex", None)
        importance = getattr(point, "importance", None)
        parts: list[str] = [f"**Point {idx}:**"]
        if text:
            parts.append(text.strip())
        if latex:
            parts.append(f"$${latex.strip()}$$")
        if importance:
            parts.append(f"(importance: {importance})")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def _format_quantitative_notes(notes: Any) -> str:
    lines: list[str] = []
    for idx, note in enumerate(notes, start=1):
        text = getattr(note, "text", None)
        latex = getattr(note, "latex", None)
        parts = [f"**Note {idx}:**"]
        if text:
            parts.append(text.strip())
        if latex:
            parts.append(f"$${latex.strip()}$$")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def _format_recommendations(recommendations: Any) -> str:
    lines: list[str] = []
    for rec in recommendations:
        text = getattr(rec, "text", None)
        severity = getattr(rec, "severity", None)
        parts: list[str] = []
        if severity:
            parts.append(f"[{severity}]")
        if text:
            parts.append(text.strip())
        lines.append("- " + " ".join(parts) if parts else "- recommendation")
    return "\n".join(lines)


def _maybe_model_dump(model: Any) -> dict | None:
    if model is None:
        return None
    if hasattr(model, "model_dump"):
        data = model.model_dump(exclude_none=True)
    else:
        data = getattr(model, "dict", lambda **_: None)(exclude_none=True)
    return data or None

