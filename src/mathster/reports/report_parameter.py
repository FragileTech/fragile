"""Markdown renderer for deduplicated parameter entries."""

from __future__ import annotations

from typing import Any, Mapping

from mathster.reports.report_utils import (
    format_bullet_list,
    format_reference_labels,
    make_section,
)

__all__ = ["parameter_entry_to_markdown"]


def parameter_entry_to_markdown(parameter: Mapping[str, Any]) -> str:
    """Return a Markdown report for a deduplicated parameter entry."""

    sections: list[str] = []

    references = parameter.get("references")
    reference_labels = format_reference_labels(references if isinstance(references, list) else None)
    reference_line = (
        f"**Reference labels:** {reference_labels}" if reference_labels else "**Reference labels:** _none_"
    )
    sections.append(reference_line)

    sections.append(_build_header_block(parameter))

    description = parameter.get("description")
    if isinstance(description, str) and description.strip():
        sections.append(make_section("Description", description.strip()))

    constraints = parameter.get("constraints")
    if isinstance(constraints, list) and constraints:
        sections.append(make_section("Constraints", format_bullet_list(constraints)))

    tags = parameter.get("tags")
    if isinstance(tags, list) and tags:
        sections.append(make_section("Tags", ", ".join(f"`{tag}`" for tag in tags if isinstance(tag, str))))

    defined_in = parameter.get("defined_in")
    if isinstance(defined_in, str) and defined_in.strip():
        sections.append(make_section("Defined In", f"`{defined_in.strip()}`"))

    combined = "\n\n".join(part for part in sections if part).strip()
    return combined + "\n"


def _build_header_block(parameter: Mapping[str, Any]) -> str:
    label = parameter.get("label")
    symbol = parameter.get("symbol")
    name = parameter.get("name")
    title = name or symbol or label or "Parameter"
    lines = [f"# {title}"]
    if label:
        lines.append(f"**Label:** `{label}`")
    if symbol:
        lines.append(f"**Symbol:** {symbol}")
    return "\n".join(lines)

