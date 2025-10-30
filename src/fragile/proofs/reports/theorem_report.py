"""Theorem/Lemma/Proposition report generator: render TheoremBox to Markdown.

Accepts either a dict from refined_data/theorems (or similar) or a TheoremBox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from fragile.proofs.core.article_system import SourceLocation
from fragile.proofs.core.math_types import Relationship, TheoremBox


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


def _relationships_list(
    rels: list[Relationship] | None | list[dict[str, Any]] | None,
) -> list[str]:
    lines: list[str] = []
    if not rels:
        return lines
    for r in rels:
        if isinstance(r, Relationship):
            arrow = "⟷" if r.bidirectional else "→"
            lines.append(
                f"- `{r.source_object}` {arrow} `{r.target_object}` ({r.relationship_type.value})"
            )
        else:
            arrow = "⟷" if r.get("bidirectional") else "→"
            rtype = r.get("relationship_type") or r.get("type") or "relationship"
            src = r.get("source_object") or r.get("source")
            tgt = r.get("target_object") or r.get("target")
            lines.append(f"- `{src}` {arrow} `{tgt}` ({rtype})")
    lines.append("")
    return lines


def theorem_to_markdown(data: TheoremBox | dict[str, Any]) -> str:
    thm = data if isinstance(data, TheoremBox) else None
    raw = None if isinstance(data, TheoremBox) else data
    if raw and thm is None:
        try:
            thm = TheoremBox.model_validate(raw)
        except Exception:
            pass

    label = thm.label if thm else raw.get("label", "")
    name = (thm.name if thm else raw.get("name") or raw.get("label", "")).strip()
    stype = (thm.statement_type if thm else raw.get("statement_type", "theorem")).capitalize()
    out_type = thm.output_type.value if thm else (raw.get("output_type") or "Property")
    chapter = (thm.chapter if thm else raw.get("chapter")) or "—"
    document = (thm.document if thm else raw.get("document")) or "—"
    source = thm.source if thm else raw.get("source")
    eq_label = (thm.equation_label if thm else raw.get("equation_label")) or "—"
    status = thm.proof_status if thm else raw.get("proof_status") or "unproven"

    inputs_objs = thm.input_objects if thm else raw.get("input_objects", [])
    inputs_ax = thm.input_axioms if thm else raw.get("input_axioms", [])
    inputs_params = thm.input_parameters if thm else raw.get("input_parameters", [])
    nat_stmt = thm.natural_language_statement if thm else raw.get("natural_language_statement")
    assumptions = thm.assumptions if thm else raw.get("assumptions", [])
    conclusion = thm.conclusion if thm else raw.get("conclusion")
    rels = (
        thm.relations_established
        if thm
        else raw.get("relations_established") or raw.get("relations")
    )
    internal_lemmas = thm.internal_lemmas if thm else raw.get("internal_lemmas", [])

    header = [f"# {stype}: {name} ({label})", ""]

    meta = [
        "| Field | Value |",
        "|-------|-------|",
        f"| Label | `{label}` |",
        f"| Output Type | `{out_type}` |",
        f"| Equation Label | `{eq_label}` |",
        f"| Proof Status | `{status}` |",
        f"| Chapter | `{chapter}` |",
        f"| Document | `{document}` |",
        f"| Source | {_format_source(source)} |",
        "",
    ]

    body: list[str] = []

    if nat_stmt:
        # Preserve prose + inline math as provided
        body.extend(["## Statement (Natural Language)", "", str(nat_stmt), ""])

    if assumptions:
        body.append("## Assumptions")
        for a in assumptions:
            latex = _dual_to_latex(a)
            if latex:
                body.append(_math_block(latex))
        body.append("")

    concl_latex = _dual_to_latex(conclusion)
    if concl_latex:
        body.extend(["## Conclusion", _math_block(concl_latex)])

    if inputs_objs or inputs_ax or inputs_params:
        body.append("## Inputs")
        if inputs_objs:
            body.append(f"- Objects: {_join(inputs_objs)}")
        if inputs_ax:
            body.append(f"- Axioms: {_join(inputs_ax)}")
        if inputs_params:
            body.append(f"- Parameters: {_join(inputs_params)}")
        body.append("")

    rel_lines = _relationships_list(rels)
    if rel_lines:
        body.append("## Relationships Established")
        body += rel_lines

    if internal_lemmas:
        body.append("## Internal Lemmas/Propositions")
        for lab in internal_lemmas:
            body.append(f"- `{lab}`")
        body.append("")

    return "\n".join(header + meta + body)


def save_theorem_markdown(data: TheoremBox | dict[str, Any], output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(theorem_to_markdown(data))
    return out
