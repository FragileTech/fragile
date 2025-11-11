"""Shared helpers for Markdown report generation."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from mathster.preprocess_extraction.data_models import Span


__all__ = [
    "format_bullet_list",
    "format_metadata",
    "format_reference_labels",
    "format_span",
    "make_section",
    "wrap_math",
]


def format_reference_labels(references: Iterable[Any] | None) -> str | None:
    labels: list[str] = []
    if not references:
        return None
    for entry in references:
        label = _extract_label_from_reference(entry)
        if label:
            labels.append(f"`{label}`")
    if not labels:
        return None
    return ", ".join(labels)


def _extract_label_from_reference(entry: Any) -> str | None:
    if isinstance(entry, str):
        entry = entry.strip()
        return entry or None
    if isinstance(entry, Mapping):
        for key in ("label", "target", "id", "ref", "reference"):
            value = entry.get(key)
            if isinstance(value, str):
                value = value.strip()
                if value:
                    return value
        title = entry.get("title")
        if isinstance(title, str):
            title = title.strip()
            if title:
                return title
    return None


def format_span(span: Span | None) -> str | None:
    if not span:
        return None
    components: list[str] = []
    if span.start_line is not None or span.end_line is not None:
        start = span.start_line if span.start_line is not None else "?"
        end = span.end_line if span.end_line is not None else "?"
        components.append(f"Lines {start}–{end}")
    if span.content_start is not None or span.content_end is not None:
        start = span.content_start if span.content_start is not None else "?"
        end = span.content_end if span.content_end is not None else "?"
        components.append(f"Content {start}–{end}")
    if span.header_lines:
        components.append("Headers: " + ", ".join(str(num) for num in span.header_lines))
    return "; ".join(components) if components else None


def make_section(title: str, body: str | None) -> str | None:
    if not body:
        return None
    return f"## {title}\n\n{body.strip()}"


def wrap_math(content: str) -> str:
    stripped = content.strip()
    if "\n" in stripped:
        return f"$${stripped}$$"
    return f"${stripped}$"


def format_metadata(
    metadata: Mapping[str, Any] | None,
    registry_context: Mapping[str, Any] | None,
) -> str | None:
    if not metadata and not registry_context:
        return None
    payload: dict[str, Any] = {}
    if metadata:
        payload["metadata"] = metadata
    if registry_context:
        payload["registry_context"] = registry_context
    return f"```json\n{json.dumps(payload, indent=2, sort_keys=True)}\n```"


def format_bullet_list(items: Iterable[str] | None) -> str | None:
    if not items:
        return None
    cleaned = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    if not cleaned:
        return None
    return "\n".join(f"- {item}" for item in cleaned)
