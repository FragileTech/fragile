"""Definition report generator: render refined Definition dicts to Markdown.

Definitions currently arrive as dictionaries under refined_data/definitions.
This reporter formats the key fields into a clean MyST Markdown card.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def _join(items: Iterable[str], sep: str = ", ") -> str:
    vals = [str(s) for s in items if s]
    return sep.join(vals) if vals else "—"


def _format_source(src: dict[str, Any] | None | None) -> str:
    if not src:
        return "—"
    parts: list[str] = []
    if src.get("document_id"):
        parts.append(src["document_id"])
    if src.get("section"):
        parts.append(str(src["section"]))
    if src.get("subsection"):
        parts.append(str(src["subsection"]))
    if src.get("label"):
        parts.append(f"#{src['label']}")
    # line_start/line_end OR line_range
    if "line_range" in src and isinstance(src["line_range"], list | tuple):
        a, b = src["line_range"]
        parts.append(f"lines {a}–{b}")
    else:
        ls, le = src.get("line_start"), src.get("line_end")
        if ls and le:
            parts.append(f"lines {ls}–{le}")
    if src.get("file_path"):
        parts.append(f"({src['file_path']})")
    return " ".join(parts) if parts else "—"


def _math_block(tex: str | None) -> str:
    if not tex:
        return "—"
    return f"\n$$\n{tex}\n$$\n"  # ensure blank line before $$


def definition_to_markdown(data: dict[str, Any]) -> str:
    label = data.get("label", "")
    name = data.get("name", label)
    nls = data.get("natural_language_statement")
    desc = data.get("description")
    inputs_objs = data.get("input_objects", [])
    inputs_ax = data.get("input_axioms", [])
    inputs_params = data.get("input_parameters", [])
    source = data.get("source") or data.get("source_location")

    # Optional math signature pulled from NLS if a signature field exists
    signature = None
    raw_fb = data.get("raw_fallback", {})
    if isinstance(raw_fb, dict):
        math_content = raw_fb.get("mathematical_content", {})
        signature = math_content.get("signature")

    header = [f"# Definition: {name} ({label})", ""]

    meta = [
        "| Field | Value |",
        "|-------|-------|",
        f"| Label | `{label}` |",
        f"| Source | {_format_source(source)} |",
        "",
    ]

    body: list[str] = []

    if nls:
        body.extend(["## Statement", "", str(nls), ""])

    if signature:
        body.extend(["## Signature", _math_block(signature)])

    if desc:
        body.extend(["## Description", "", str(desc), ""])

    if inputs_objs or inputs_ax or inputs_params:
        body.append("## Inputs/Dependencies")
        if inputs_objs:
            body.append(f"- Objects: {_join(inputs_objs)}")
        if inputs_ax:
            body.append(f"- Axioms: {_join(inputs_ax)}")
        if inputs_params:
            body.append(f"- Parameters: {_join(inputs_params)}")
        body.append("")

    return "\n".join(header + meta + body)


def save_definition_markdown(data: dict[str, Any], output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(definition_to_markdown(data))
    return out
