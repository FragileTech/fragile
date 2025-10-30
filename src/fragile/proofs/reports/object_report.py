"""MathematicalObject report generator: render objects to Markdown.

Accepts either a dict loaded from JSON (refined_data/objects) or a
MathematicalObject instance, and produces a readable MyST Markdown view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from fragile.proofs.core.article_system import SourceLocation
from fragile.proofs.core.math_types import MathematicalObject


def _join(items: Iterable[str], sep: str = ", ") -> str:
    vals = [str(s) for s in items if s]
    return sep.join(vals) if vals else "—"


def _format_source(src: SourceLocation | None | dict[str, Any] | None) -> str:
    if not src:
        return "—"
    if isinstance(src, dict):
        parts = [src.get("document_id") or ""]
        if src.get("section"):
            parts.append(str(src["section"]))
        if src.get("directive_label"):
            parts.append(f"#{src['directive_label']}")
        if src.get("line_range"):
            a, b = src["line_range"]
            parts.append(f"lines {a}–{b}")
        if src.get("file_path"):
            parts.append(f"({src['file_path']})")
        return " ".join([p for p in parts if p]) or "—"
    parts = [src.document_id]
    if src.section:
        parts.append(src.section)
    if src.directive_label:
        parts.append(f"#{src.directive_label}")
    if src.line_range:
        parts.append(f"lines {src.line_range[0]}–{src.line_range[1]}")
    parts.append(f"({src.file_path})")
    return " ".join(parts)


def _render_expression(expr: str | None) -> str:
    """Render an object's mathematical expression preserving existing math.

    Heuristics:
    - If expr already contains '$' or a newline, assume author provided proper
      Markdown/LaTeX; include verbatim without forcing $$ wrappers.
    - Otherwise, if it looks like a short formula, wrap in $$ for display math.
    """
    if not expr:
        return "—"
    s = str(expr).strip()
    if "$" in s or "\n" in s:
        return "\n" + s + "\n"
    # Simple formula heuristic
    if any(ch in s for ch in ["=", "\\", "∈", "≤", "≥"]) and len(s) < 200:
        return "\n$$\n" + s + "\n$$\n"
    return "\n" + s + "\n"


def object_to_markdown(data: MathematicalObject | dict[str, Any]) -> str:
    """Render a MathematicalObject or compatible dict to Markdown."""
    if isinstance(data, MathematicalObject):
        obj = data
        raw = None
    else:
        raw = data
        # Try to coerce minimally; refined JSON may have extra fields
        try:
            obj = MathematicalObject.model_validate(raw)
        except Exception:
            obj = None

    label = obj.label if obj else raw.get("label", "")
    name = obj.name if obj else raw.get("name", "")
    obj_type = (obj.object_type.value if obj else raw.get("object_type", "")).capitalize()
    expr = obj.mathematical_expression if obj else raw.get("mathematical_expression", "")

    tags = (obj.tags if obj else raw.get("tags", [])) or []
    chapter = (obj.chapter if obj else raw.get("chapter")) or "—"
    document = (obj.document if obj else raw.get("document")) or "—"
    definition_label = (obj.definition_label if obj else raw.get("definition_label")) or "—"
    source = obj.source if obj else raw.get("source")

    header = [f"# Object: {name} ({label})", ""]

    meta = [
        "| Field | Value |",
        "|-------|-------|",
        f"| Label | `{label}` |",
        f"| Type | `{obj_type}` |",
        f"| Definition | `{definition_label}` |",
        f"| Tags | {_join(tags)} |",
        f"| Chapter | `{chapter}` |",
        f"| Document | `{document}` |",
        f"| Source | {_format_source(source)} |",
        "",
    ]

    body = [
        "## Expression",
        _render_expression(expr),
    ]

    # Attributes (if available)
    if obj and obj.current_attributes:
        body.append("## Current Attributes")
        for a in obj.current_attributes:
            body.append(f"- `{a.label}`: ${a.expression}$")
        body.append("")

    return "\n".join(header + meta + body)


def save_object_markdown(
    data: MathematicalObject | dict[str, Any], output_path: str | Path
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(object_to_markdown(data))
    return out
