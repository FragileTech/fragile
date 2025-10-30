"""Pipeline Remark reporter: render Stage-1/2 remark JSON to Markdown.

Accepts pipeline/raw remark dicts with keys like 'remark_type' and 'full_text'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def pipeline_remark_to_markdown(data: dict[str, Any]) -> str:
    label = data.get("label") or data.get("temp_id") or "remark"
    rtype = data.get("remark_type") or data.get("type") or "remark"
    text = data.get("content") or data.get("full_text") or ""

    header = [f"# Remark: {label}", ""]
    meta = [
        "| Field | Value |",
        "|-------|-------|",
        f"| Label | `{label}` |",
        f"| Type | `{rtype}` |",
        "",
    ]

    # Preserve inline math and formatting as provided by source
    body = [
        "## Content",
        "",
        str(text),
        "",
    ]

    return "\n".join(header + meta + body)


def save_pipeline_remark_markdown(data: dict[str, Any], output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pipeline_remark_to_markdown(data))
    return out
