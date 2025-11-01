"""Pipeline Parameter report generator: render Stage-1/2 param JSON to Markdown.

Handles pipeline_data/parameters JSON schema (name, symbol, parameter_type, ...)
which is different from the enriched ParameterBox reporter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def pipeline_parameter_to_markdown(data: dict[str, Any]) -> str:
    label = data.get("label", "")
    name = data.get("name", label)
    symbol = data.get("symbol", "")
    ptype = data.get("parameter_type") or data.get("type") or "—"
    constraints = data.get("constraints") or "—"
    default_value = data.get("default_value") or data.get("default") or "—"
    domain = data.get("domain") or "—"

    header = [f"# Parameter: {name} ({label})", ""]
    meta = [
        "| Field | Value |",
        "|-------|-------|",
        f"| Label | `{label}` |",
        f"| Symbol | `{symbol}` |",
        f"| Type | `{ptype}` |",
        f"| Domain | `{domain}` |",
        f"| Constraints | {constraints} |",
        f"| Default | `{default_value}` |",
        "",
    ]
    return "\n".join(header + meta)


def save_pipeline_parameter_markdown(data: dict[str, Any], output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pipeline_parameter_to_markdown(data))
    return out
