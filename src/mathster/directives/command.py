"""Directive extraction orchestration helpers.

These utilities power the ``mathster parse`` CLI subcommand while remaining
independent of Click. They expose a simple function that:

1. Splits a MyST markdown file into sections
2. Serializes every directive hint with absolute line numbers
3. Validates and optionally previews each section
4. Writes ``chapter_{k}.json`` payloads under a ``directives/`` workspace
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mathster.directives import preview_hints, validate_hints
from mathster.directives.directive_parser import (
    DirectiveHint,
    DocumentSection,
    split_into_sections,
)


DirectiveSummary = dict[str, Any]


def _format_section_heading(section: DocumentSection) -> str:
    return f"{'#' * section.level} {section.title}"


def _slice_numbered_lines(lines: list[str], start: int | None, end: int | None) -> str:
    """Return numbered line snippet using 1-indexed inclusive/exclusive bounds."""
    if start is None or end is None or start <= 0 or end <= 0:
        return ""
    start_idx = max(start - 1, 0)
    end_idx = min(end, len(lines))
    return "\n".join(lines[start_idx:end_idx])


def _serialize_hint(
    section: DocumentSection,
    hint: DirectiveHint,
    numbered_lines: list[str],
) -> dict:
    """Convert a ``DirectiveHint`` into an absolute-line-number dictionary."""
    offset = section.start_line - 1
    hint_dict = hint.to_dict()

    def adjust(value: int | None) -> int | None:
        if value is None:
            return None
        return value + offset

    hint_dict["start_line"] = adjust(hint_dict.get("start_line"))
    hint_dict["end_line"] = adjust(hint_dict.get("end_line"))
    hint_dict["content_start"] = adjust(hint_dict.get("content_start"))
    hint_dict["content_end"] = adjust(hint_dict.get("content_end"))

    header_lines: list[int] = []
    for raw_line in hint_dict.get("header_lines", []):
        adjusted = adjust(raw_line)
        if adjusted is not None:
            header_lines.append(adjusted)
    hint_dict["header_lines"] = header_lines

    hint_dict["section"] = _format_section_heading(section)
    hint_dict["content"] = _slice_numbered_lines(
        numbered_lines,
        hint_dict.get("content_start"),
        hint_dict.get("content_end"),
    )
    hint_dict["raw_directive"] = _slice_numbered_lines(
        numbered_lines,
        hint_dict.get("start_line"),
        hint_dict.get("end_line"),
    )
    return hint_dict


def run_directive_extraction(
    markdown_file: Path,
    output_dir: Path | None = None,
    *,
    preview: bool = True,
    validate: bool = True,
) -> dict:
    """Extract directive hints and persist them to chapter JSON files.

    Returns a summary dictionary containing the output directory, written files,
    and aggregate counts. Raises ``ValueError`` if the document has no sections.
    """
    markdown_path = Path(markdown_file)
    document_text = markdown_path.read_text(encoding="utf-8")
    sections = split_into_sections(document_text, str(markdown_path))
    if not sections:
        raise ValueError(f"No sections detected in {markdown_path}")

    base_dir = output_dir or (markdown_path.parent / markdown_path.stem)
    directives_dir = base_dir / "directives"
    directives_dir.mkdir(parents=True, exist_ok=True)

    numbered_lines = [f"{idx + 1}: {line}" for idx, line in enumerate(document_text.splitlines())]

    chapter_summaries: list[DirectiveSummary] = []
    total_directives = 0

    for index, section in enumerate(sections):
        hints = [_serialize_hint(section, hint, numbered_lines) for hint in section.directives]
        if validate:
            ok, errors = validate_hints(hints)
            validation = {"ok": ok, "errors": errors}
        else:
            validation = {"ok": True, "errors": []}

        payload = {
            "chapter_index": index,
            "section_id": _format_section_heading(section),
            "directive_count": len(hints),
            "hints": hints,
            "validation": validation,
        }

        output_path = directives_dir / f"chapter_{index}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        chapter_summary: DirectiveSummary = {
            "index": index,
            "section_id": payload["section_id"],
            "path": output_path,
            "directive_count": len(hints),
            "validation": validation,
        }
        if preview and hints:
            chapter_summary["preview"] = preview_hints(hints)

        chapter_summaries.append(chapter_summary)
        total_directives += len(hints)

    return {
        "output_dir": directives_dir,
        "chapters": chapter_summaries,
        "total_chapters": len(sections),
        "total_directives": total_directives,
    }
