"""Parameter report generator: render ParameterBox to Markdown.

Usage (CLI):
    python -m fragile.mathster.reports.parameter_report \
        --input path/to/param-gamma.json \
        --output path/to/param-gamma.md

Programmatic:
    from fragile.mathster.reports.parameter_report import parameter_to_markdown
    md = parameter_to_markdown(data)  # dict or ParameterBox
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from mathster.core.article_system import SourceLocation
from mathster.core.enriched_types import ParameterBox, ParameterScope
from mathster.core.math_types import ParameterType


def _join_nonempty(items: Iterable[str], sep: str = ", ") -> str:
    vals = [s for s in items if s]
    return sep.join(vals) if vals else "—"


def _format_source(src: SourceLocation | None) -> str:
    if not src:
        return "—"
    parts = [src.document_id]
    if src.section:
        parts.append(src.section)
    if src.label:
        parts.append(f"#{src.label}")
    if src.line_range:
        parts.append(f"lines {src.line_range[0]}–{src.line_range[1]}")
    parts.append(f"({src.file_path})")
    return " ".join(parts)


def _format_math_inline(latex: str) -> str:
    # Wrap LaTeX for inline rendering; assume already escaped
    return f"${latex}$"


def parameter_to_markdown(data: ParameterBox | dict[str, Any]) -> str:
    """Render a single ParameterBox (or raw dict) to Markdown string."""
    param = data if isinstance(data, ParameterBox) else ParameterBox.model_validate(data)

    header = [
        f"# Parameter: {param.symbol} ({param.label})",
        "",
    ]

    meta_rows = [
        "| Field | Value |",
        "|-------|-------|",
        f"| Label | `{param.label}` |",
        f"| Symbol | `{param.symbol}` |",
        f"| LaTeX | {_format_math_inline(param.latex)} |",
        f"| Domain | `{ParameterType(param.domain).value}` |",
        f"| Scope | `{ParameterScope(param.scope).value}` |",
        f"| Constraints | {_join_nonempty(param.constraints)} |",
        f"| Default | `{param.default_value or '—'}` |",
        f"| Appears In | {_join_nonempty(param.appears_in)} |",
        f"| Source | {_format_source(param.source)} |",
    ]

    sections: list[str] = [
        "",
        "## Meaning",
        "",
        param.meaning.strip(),
        "",
    ]

    if param.full_definition_text:
        sections.extend([
            "## Full Definition",
            "",
            param.full_definition_text.strip(),
            "",
        ])

    if param.validation_errors:
        sections.extend([
            "## Validation Notes",
            "",
            *[f"- {msg}" for msg in param.validation_errors],
            "",
        ])

    lines = header + meta_rows + sections
    return "\n".join(lines)


def save_parameter_markdown(data: ParameterBox | dict[str, Any], output_path: str | Path) -> Path:
    """Save rendered Markdown to a file and return the path."""
    md = parameter_to_markdown(data)
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

    parser = argparse.ArgumentParser(description="Render ParameterBox JSON to Markdown")
    parser.add_argument("--input", required=True, help="Path to ParameterBox JSON file")
    parser.add_argument("--output", required=True, help="Path to output Markdown file")
    args = parser.parse_args()

    data = _load_json(args.input)
    save_parameter_markdown(data, args.output)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    _main_cli()
