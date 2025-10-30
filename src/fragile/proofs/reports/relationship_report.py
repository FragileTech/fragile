"""Relationship report generator: render Relationship to Markdown."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fragile.proofs.core.math_types import Relationship


def _math_block(tex: str | None) -> str:
    if not tex:
        return "—"
    return f"\n$$\n{tex}\n$$\n"  # ensure blank line before $$


def relationship_to_markdown(data: Relationship | dict[str, Any]) -> str:
    rel = data if isinstance(data, Relationship) else None
    raw = None if isinstance(data, Relationship) else data
    if raw and rel is None:
        try:
            rel = Relationship.model_validate(raw)
        except Exception:
            pass

    label = rel.label if rel else raw.get("label", "")
    rtype = (
        rel.relationship_type.value
        if rel
        else raw.get("relationship_type") or raw.get("type") or "relationship"
    )
    bidirectional = rel.bidirectional if rel else bool(raw.get("bidirectional", False))
    src = rel.source_object if rel else raw.get("source_object") or raw.get("source")
    tgt = rel.target_object if rel else raw.get("target_object") or raw.get("target")
    expr = rel.expression if rel else raw.get("expression", "")
    established_by = rel.established_by if rel else raw.get("established_by", "—")
    tags = (rel.tags if rel else raw.get("tags", [])) or []

    header = [f"# Relationship: {label}", ""]

    meta = [
        "| Field | Value |",
        "|-------|-------|",
        f"| Label | `{label}` |",
        f"| Type | `{rtype}` |",
        f"| Direction | `{'Bidirectional' if bidirectional else 'Directed'}` |",
        f"| Source | `{src}` |",
        f"| Target | `{tgt}` |",
        f"| Established By | `{established_by}` |",
        f"| Tags | {', '.join(tags) if tags else '—'} |",
        "",
    ]

    body = ["## Expression", _math_block(expr)]

    return "\n".join(header + meta + body)


def save_relationship_markdown(
    data: Relationship | dict[str, Any], output_path: str | Path
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(relationship_to_markdown(data))
    return out
