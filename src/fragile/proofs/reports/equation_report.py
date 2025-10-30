"""Equation report generator: render EquationBox to Markdown.

This module provides utilities to transform enriched EquationBox data
into a nicely formatted Markdown document for documentation/review.

Usage (CLI):
    python -m fragile.proofs.reports.equation_report \
        --input path/to/eq-langevin.json \
        --output path/to/eq-langevin.md

Programmatic:
    from fragile.proofs.reports.equation_report import equation_to_markdown
    md = equation_to_markdown(data)  # dict or EquationBox
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from fragile.proofs.core.article_system import SourceLocation
from fragile.proofs.core.enriched_types import EquationBox


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


def _equation_display(eq: EquationBox) -> str:
    # Prefer dual statement if present; otherwise use raw LaTeX content
    if eq.dual_statement is not None:
        latex = eq.dual_statement.to_latex()
    else:
        latex = eq.latex_content

    # MyST markdown requires exactly one blank line before $$ blocks
    return f"\n$$\n{latex}\n$$"  # blank line before and after


def equation_to_markdown(data: EquationBox | dict[str, Any]) -> str:
    """Render a single EquationBox (or raw dict) to Markdown string."""
    eq = data if isinstance(data, EquationBox) else EquationBox.model_validate(data)

    header = [
        f"# Equation: {eq.label}",
        "",
    ]

    meta_rows = [
        "| Field | Value |",
        "|-------|-------|",
        f"| Label | `{eq.label}` |",
        f"| Number | `{eq.equation_number or '—'}` |",
        f"| Introduces | {_join_nonempty(eq.introduces_symbols)} |",
        f"| Uses | {_join_nonempty(eq.uses_symbols)} |",
        f"| Referenced By | {_join_nonempty(eq.referenced_by)} |",
        f"| Appears In Theorems | {_join_nonempty(eq.appears_in_theorems)} |",
        f"| Source | {_format_source(eq.source)} |",
    ]

    sections: list[str] = []

    if eq.context_before:
        sections.extend([
            "## Context Before",
            "",
            eq.context_before.strip(),
            "",
        ])

    sections.extend([
        "## Equation",
        _equation_display(eq),
        "",
    ])

    if eq.context_after:
        sections.extend([
            "## Context After",
            "",
            eq.context_after.strip(),
            "",
        ])

    if eq.validation_errors:
        sections.extend([
            "## Validation Notes",
            "",
            *[f"- {msg}" for msg in eq.validation_errors],
            "",
        ])

    lines = header + meta_rows + [""] + sections
    return "\n".join(lines)


def save_equation_markdown(data: EquationBox | dict[str, Any], output_path: str | Path) -> Path:
    """Save rendered Markdown to a file and return the path."""
    md = equation_to_markdown(data)
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

    parser = argparse.ArgumentParser(description="Render EquationBox JSON to Markdown")
    parser.add_argument("--input", required=True, help="Path to EquationBox JSON file")
    parser.add_argument("--output", required=True, help="Path to output Markdown file")
    args = parser.parse_args()

    data = _load_json(args.input)
    save_equation_markdown(data, args.output)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    _main_cli()
