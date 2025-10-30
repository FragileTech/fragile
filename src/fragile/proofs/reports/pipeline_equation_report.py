"""Pipeline Equation reporter: render Stage-1/2 equation JSON to Markdown."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _equation_block(latex: str) -> str:
    s = str(latex).strip()
    # Ensure blank line and $$ wrappers
    return "\n$$\n" + s + "\n$$\n"


def pipeline_equation_to_markdown(data: dict[str, Any]) -> str:
    label = data.get("label") or data.get("temp_id") or "eq"
    eqn_label = data.get("equation_label") or data.get("equation_number") or "—"
    latex = data.get("latex_content") or data.get("latex") or ""
    before = data.get("context_before")
    after = data.get("context_after")

    header = [f"# Equation: {label}", ""]
    meta = [
        "| Field | Value |",
        "|-------|-------|",
        f"| Label | `{label}` |",
        f"| Number | `{eqn_label}` |",
        "",
    ]

    body: list[str] = []
    if before:
        body.extend(["## Context Before", "", str(before), ""])
    body.extend(["## Equation", _equation_block(latex)])
    if after:
        body.extend(["## Context After", "", str(after), ""])

    return "\n".join(header + meta + body)


def save_pipeline_equation_markdown(data: dict[str, Any], output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pipeline_equation_to_markdown(data))
    return out
