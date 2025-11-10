"""Utilities to render :class:`UnifiedMathematicalEntity` instances as Markdown."""

from __future__ import annotations

from typing import Any

from mathster.preprocess_extraction.data_models import (
    Assumption,
    Conclusion,
    Equation,
    Hypothesis,
    UnifiedMathematicalEntity,
    UnifiedProof,
    Variable,
)

from mathster.reports.report_utils import (
    format_bullet_list,
    format_metadata,
    format_reference_labels,
    format_span,
    make_section,
    wrap_math,
)

__all__ = ["unified_theorem_to_markdown", "format_unified_proof"]


def unified_theorem_to_markdown(theorem: UnifiedMathematicalEntity) -> str:
    """Return a Markdown report describing ``theorem``."""

    sections: list[str] = []

    reference_labels = format_reference_labels(theorem.references)
    reference_line = (
        f"**Reference labels:** {reference_labels}" if reference_labels else "**Reference labels:** _none_"
    )
    sections.append(reference_line)

    sections.append(_build_header_block(theorem))

    if theorem.nl_statement:
        sections.append(make_section("Natural Language Statement", theorem.nl_statement.strip()))

    if theorem.conclusion:
        sections.append(make_section("Conclusion", _format_conclusion(theorem.conclusion)))

    if theorem.hypotheses:
        sections.append(make_section("Hypotheses", _format_hypotheses(theorem.hypotheses)))

    if theorem.variables:
        sections.append(make_section("Variables", _format_variables(theorem.variables)))

    if theorem.equations:
        sections.append(make_section("Equations", _format_equations(theorem.equations)))

    if theorem.implicit_assumptions:
        sections.append(
            make_section("Implicit Assumptions", _format_assumptions(theorem.implicit_assumptions))
        )

    if theorem.local_refs:
        sections.append(make_section("Local References", format_bullet_list(theorem.local_refs)))

    if theorem.tags:
        sections.append(make_section("Tags", ", ".join(f"`{tag}`" for tag in theorem.tags)))

    if theorem.proof:
        sections.append(make_section("Proof", format_unified_proof(theorem.proof)))

    if theorem.content_markdown:
        sections.append(
            make_section(
                "Directive Content",
                f"```markdown\n{theorem.content_markdown.strip()}\n```",
            )
        )

    metadata_block = format_metadata(theorem.metadata, theorem.registry_context)
    if metadata_block:
        sections.append(make_section("Metadata", metadata_block))

    combined = "\n\n".join(part for part in sections if part).strip()
    return combined + "\n"


def _build_header_block(theorem: UnifiedMathematicalEntity) -> str:
    title = theorem.title or theorem.label
    lines = [f"# {title}"]
    if theorem.label:
        lines.append(f"**Label:** `{theorem.label}`")
    lines.append(f"**Type:** {theorem.type}")

    info_parts: list[str] = []
    if theorem.document_id:
        info_parts.append(f"Document: `{theorem.document_id}`")
    if theorem.section:
        info_parts.append(f"Section: {theorem.section}")
    span = format_span(theorem.span)
    if span:
        info_parts.append(span)
    if theorem.generated_at:
        info_parts.append(f"Generated at: {theorem.generated_at}")
    if theorem.alt_labels:
        info_parts.append("Alternate labels: " + ", ".join(f"`{lbl}`" for lbl in theorem.alt_labels))
    if info_parts:
        lines.append(" • ".join(info_parts))
    return "\n".join(lines)


def _format_hypotheses(hypotheses: list[Hypothesis]) -> str:
    rendered: list[str] = []
    for idx, hyp in enumerate(hypotheses, start=1):
        parts: list[str] = []
        if hyp.text:
            parts.append(hyp.text.strip())
        if hyp.latex:
            parts.append(wrap_math(hyp.latex))
        text = " ".join(parts) if parts else f"Hypothesis {idx}"
        rendered.append(f"- {text}")
    return "\n".join(rendered)


def _format_variables(variables: list[Variable]) -> str:
    rendered: list[str] = []
    for var in variables:
        pieces: list[str] = []
        if var.symbol:
            pieces.append(f"`{var.symbol}`")
        if var.name:
            pieces.append(var.name)
        header = " - ".join(pieces) if pieces else "Variable"
        body_parts: list[str] = []
        if var.description:
            body_parts.append(var.description.strip())
        if var.constraints:
            body_parts.append("Constraints: " + ", ".join(var.constraints))
        if var.tags:
            body_parts.append("Tags: " + ", ".join(f"`{tag}`" for tag in var.tags))
        rendered.append(f"- **{header}:** {' '.join(body_parts) if body_parts else '—'}")
    return "\n".join(rendered)


def _format_equations(equations: list[Equation]) -> str:
    rendered: list[str] = []
    for eq in equations:
        label = f"**{eq.label}:**" if eq.label else "Equation"
        rendered.append(f"{label}\n\n$${eq.latex.strip()}$$")
    return "\n\n".join(rendered)


def _format_assumptions(assumptions: list[Assumption]) -> str:
    lines: list[str] = []
    for assumption in assumptions:
        text = assumption.text.strip()
        if assumption.confidence is not None:
            text += f" (confidence: {assumption.confidence:.2f})"
        lines.append(f"- {text}")
    return "\n".join(lines)


def _format_conclusion(conclusion: Conclusion) -> str:
    parts: list[str] = []
    if conclusion.text:
        parts.append(conclusion.text.strip())
    if conclusion.latex:
        parts.append(wrap_math(conclusion.latex))
    return "\n\n".join(parts) if parts else "—"


def format_unified_proof(proof: UnifiedProof) -> str:
    lines: list[str] = []
    lines.append(f"**Proof label:** `{proof.label}`")
    if proof.proof_type:
        lines.append(f"**Type:** {proof.proof_type}")
    if proof.proof_status:
        lines.append(f"**Status:** {proof.proof_status}")
    if proof.strategy_summary:
        lines.append(f"**Strategy:** {proof.strategy_summary.strip()}")
    if proof.conclusion:
        lines.append("### Proof Conclusion\n\n" + _format_conclusion(proof.conclusion))
    if proof.key_equations:
        lines.append("### Key Equations\n\n" + _format_key_equations(proof.key_equations))
    if proof.assumptions:
        lines.append("### Assumptions\n\n" + _format_proof_assumptions(proof.assumptions))
    if proof.steps:
        lines.append("### Steps\n\n" + _format_proof_steps(proof.steps))
    if proof.math_tools:
        lines.append("### Math Tools\n\n" + _format_math_tools(proof.math_tools))
    if proof.references:
        lab = format_reference_labels(list(proof.references))
        if lab:
            lines.append(f"**References:** {lab}")
    return "\n\n".join(segment for segment in lines if segment)


def _format_key_equations(equations: list[Any]) -> str:
    blocks: list[str] = []
    for eq in equations:
        label = getattr(eq, "label", None)
        latex = getattr(eq, "latex", None)
        role = getattr(eq, "role", None)
        header = f"**{label}:**" if label else "Equation"
        parts = [header]
        if role:
            parts.append(f"Role: {role}")
        if latex:
            parts.append(f"$${latex.strip()}$$")
        blocks.append("\n\n".join(parts))
    return "\n\n".join(blocks)


def _format_proof_assumptions(assumptions: list[Any]) -> str:
    rendered: list[str] = []
    for assumption in assumptions:
        text = getattr(assumption, "text", None)
        latex = getattr(assumption, "latex", None)
        pieces = []
        if text:
            pieces.append(text.strip())
        if latex:
            pieces.append(wrap_math(latex))
        rendered.append("- " + " ".join(pieces) if pieces else "- assumption")
    return "\n".join(rendered)


def _format_proof_steps(steps: list[Any]) -> str:
    rendered: list[str] = []
    enumerated = list(enumerate(steps))
    enumerated.sort(
        key=lambda item: (
            float("inf") if getattr(item[1], "order", None) is None else getattr(item[1], "order"),
            item[0],
        )
    )
    for idx, step in enumerated:
        heading_parts: list[str] = []
        order = getattr(step, "order", None)
        if order is not None:
            heading_parts.append(f"{order}")
        kind = getattr(step, "kind", None)
        if kind:
            heading_parts.append(kind)
        heading = " / ".join(heading_parts) if heading_parts else f"Step {idx + 1}"
        body_parts: list[str] = []
        text = getattr(step, "text", None)
        if text:
            body_parts.append(text.strip())
        latex = getattr(step, "latex", None)
        if latex:
            body_parts.append(wrap_math(latex))
        derived = getattr(step, "derived_statement", None)
        if derived:
            body_parts.append(f"Derived: {derived}")
        refs = getattr(step, "references", None)
        if refs:
            ref_line = format_reference_labels(list(refs))
            if ref_line:
                body_parts.append(f"Refs: {ref_line}")
        body = "\n   ".join(body_parts if body_parts else ["—"])
        rendered.append(f"{len(rendered) + 1}. **{heading}**\n   {body}")
    return "\n".join(rendered)


def _format_math_tools(tools: list[Any]) -> str:
    lines: list[str] = []
    for tool in tools:
        name = getattr(tool, "toolName", None) or getattr(tool, "name", None)
        description = getattr(tool, "description", None)
        role = getattr(tool, "roleInProof", None)
        parts: list[str] = []
        if name:
            parts.append(f"**{name}**")
        if role:
            parts.append(f"role: {role}")
        if description:
            parts.append(description.strip())
        lines.append("- " + " — ".join(parts) if parts else "- tool")
    return "\n".join(lines)
