"""Remark report generator: render RemarkBox to Markdown.

Usage (CLI):
    python -m fragile.proofs.reports.remark_report \
        --input path/to/remark-*.json \
        --output path/to/remark-*.md

Programmatic:
    from fragile.proofs.reports.remark_report import remark_to_markdown
    md = remark_to_markdown(data)  # dict or RemarkBox
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from fragile.proofs.core.article_system import SourceLocation
from fragile.proofs.core.enriched_types import RemarkBox, RemarkType


def _join_nonempty(items: Iterable[str], sep: str = ", ") -> str:
    vals = [s for s in items if s]
    return sep.join(vals) if vals else "—"


def _format_source(src: SourceLocation | None) -> str:
    if not src:
        return "—"
    parts = [src.document_id]
    if src.section:
        parts.append(src.section)
    if src.directive_label:
        parts.append(f"#{src.directive_label}")
    if src.line_range:
        parts.append(f"lines {src.line_range[0]}–{src.line_range[1]}")
    parts.append(f"({src.file_path})")
    return " ".join(parts)


def remark_to_markdown(data: RemarkBox | dict[str, Any]) -> str:
    """Render a single RemarkBox (or raw dict) to Markdown string."""
    remark = data if isinstance(data, RemarkBox) else RemarkBox.model_validate(data)

    header = [
        f"# Remark: {remark.label}",
        "",
    ]

    meta_rows = [
        "| Field | Value |",
        "|-------|-------|",
        f"| Label | `{remark.label}` |",
        f"| Type | `{RemarkType(remark.remark_type).value}` |",
        f"| Relates To | {_join_nonempty(remark.relates_to)} |",
        f"| Provides Intuition For | {_join_nonempty(remark.provides_intuition_for)} |",
        f"| Source | {_format_source(remark.source)} |",
    ]

    sections: list[str] = [
        "",
        "## Content",
        "",
        remark.content.strip(),
        "",
    ]

    if remark.validation_errors:
        sections.extend([
            "## Validation Notes",
            "",
            *[f"- {msg}" for msg in remark.validation_errors],
            "",
        ])

    lines = header + meta_rows + sections
    return "\n".join(lines)


def save_remark_markdown(data: RemarkBox | dict[str, Any], output_path: str | Path) -> Path:
    """Save rendered Markdown to a file and return the path."""
    md = remark_to_markdown(data)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    return out


def _load_json(path: str | Path) -> dict[str, Any]:
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _main_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render RemarkBox JSON to Markdown")
    parser.add_argument("--input", required=True, help="Path to RemarkBox JSON file")
    parser.add_argument("--output", required=True, help="Path to output Markdown file")
    args = parser.parse_args()

    data = _load_json(args.input)
    save_remark_markdown(data, args.output)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    _main_cli()
