"""Axiom report generator: render Axiom to Markdown.

Accepts either a dict from refined_data/axioms or an Axiom instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from fragile.proofs.core.article_system import SourceLocation
from fragile.proofs.core.math_types import Axiom, AxiomaticParameter


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


def _math_block(tex: str | None) -> str:
    if not tex:
        return "—"
    return f"\n$$\n{tex}\n$$\n"  # ensure blank line before $$


def _dual_to_latex(dual: Any) -> str | None:
    try:
        if dual is None:
            return None
        if hasattr(dual, "to_latex"):
            return dual.to_latex()
        # If dict-like DualStatement
        if isinstance(dual, dict):
            lhs = dual.get("lhs", {}).get("latex") or ""
            rel = dual.get("relation", "=")
            rhs = dual.get("rhs", {}).get("latex") or ""
            ctx = dual.get("context")
            expr = f"{lhs} {rel} {rhs}"
            if ctx:
                expr += f" \\quad \\text{{({ctx})}}"
            return expr
    except Exception:
        pass
    return None


def _parameters_table(params: list | None | None) -> list[str]:
    if not params:
        return []
    lines = [
        "| Symbol | Description | Constraints |",
        "|--------|-------------|-------------|",
    ]
    for p in params:
        if isinstance(p, AxiomaticParameter):
            sym = p.symbol
            desc = p.description
            cons = p.constraints or "—"
        else:  # dict-like
            sym = p.get("symbol", "")
            desc = p.get("description", "")
            cons = p.get("constraints") or "—"
        lines.append(f"| `{sym}` | {desc} | {cons} |")
    return [*lines, ""]


def axiom_to_markdown(data: Axiom | dict[str, Any]) -> str:
    axiom = data if isinstance(data, Axiom) else None
    raw = None if isinstance(data, Axiom) else data
    if raw and axiom is None:
        try:
            axiom = Axiom.model_validate(raw)
        except Exception:
            pass

    label = axiom.label if axiom else raw.get("label", "")
    name = axiom.name if axiom else raw.get("name") or raw.get("statement") or ""
    framework = axiom.foundational_framework if axiom else raw.get("foundational_framework", "")
    math_expr = axiom.mathematical_expression if axiom else raw.get("mathematical_expression", "")
    chapter = (axiom.chapter if axiom else raw.get("chapter")) or "—"
    document = (axiom.document if axiom else raw.get("document")) or "—"
    source = axiom.source if axiom else raw.get("source")
    core = axiom.core_assumption if axiom else raw.get("core_assumption")
    condition = axiom.condition if axiom else raw.get("condition")
    parameters = axiom.parameters if axiom else raw.get("parameters")
    fma = axiom.failure_mode_analysis if axiom else raw.get("failure_mode_analysis")

    header = [f"# Axiom: {name} ({label})", ""]

    meta = [
        "| Field | Value |",
        "|-------|-------|",
        f"| Label | `{label}` |",
        f"| Framework | `{framework}` |",
        f"| Chapter | `{chapter}` |",
        f"| Document | `{document}` |",
        f"| Source | {_format_source(source)} |",
        "",
    ]

    body: list[str] = []

    # Plain text statement (if provided)
    stmt_text = axiom.statement if axiom else raw.get("statement")
    if stmt_text:
        body.extend(["## Statement (Text)", "", str(stmt_text), ""])

    # Core assumption / statement
    core_latex = _dual_to_latex(core)
    if core_latex:
        body.extend(["## Core Assumption", _math_block(core_latex)])
    elif math_expr:
        # Preserve author-provided math/markdown without forcing $$ if it mixes prose
        s = str(math_expr).strip()
        if "$" in s or "\n" in s:
            body.extend(["## Statement", "", s, ""])  # verbatim markdown
        else:
            body.extend(["## Statement", _math_block(s)])

    # Condition (if any)
    cond_latex = _dual_to_latex(condition)
    if cond_latex:
        body.extend(["## Condition", _math_block(cond_latex)])

    # Parameters
    body.extend(["## Parameters", "", *_parameters_table(parameters)])

    # Failure mode analysis
    if fma:
        body.extend(["## Failure Mode Analysis", "", str(fma), ""])

    return "\n".join(header + meta + body)


def save_axiom_markdown(data: Axiom | dict[str, Any], output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(axiom_to_markdown(data))
    return out
