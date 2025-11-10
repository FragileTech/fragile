"""Markdown renderer for :class:`Algorithm` preprocess entries."""

from __future__ import annotations

from typing import Any

from mathster.preprocess_extraction.data_models import (
    Algorithm,
    AlgorithmParameter,
    AlgorithmSignature,
    AlgorithmStep,
    FailureMode,
    GuardCondition,
)

from mathster.reports.report_utils import (
    format_bullet_list,
    format_reference_labels,
    make_section,
)

__all__ = ["unified_algorithm_to_markdown"]


def unified_algorithm_to_markdown(algorithm: Algorithm) -> str:
    """Return a Markdown report describing ``algorithm``."""

    sections: list[str] = []

    reference_labels = format_reference_labels(algorithm.references)
    reference_line = (
        f"**Reference labels:** {reference_labels}" if reference_labels else "**Reference labels:** _none_"
    )
    sections.append(reference_line)

    sections.append(_build_header_block(algorithm))

    if algorithm.nl_summary:
        sections.append(make_section("Summary", algorithm.nl_summary.strip()))

    if algorithm.signature:
        sections.append(make_section("Signature", _format_signature(algorithm.signature)))

    if algorithm.steps:
        sections.append(make_section("Steps", _format_steps(algorithm.steps)))

    if algorithm.guard_conditions:
        sections.append(make_section("Guard Conditions", _format_guards(algorithm.guard_conditions)))

    if algorithm.failure_modes:
        sections.append(make_section("Failure Modes", _format_failure_modes(algorithm.failure_modes)))

    if algorithm.tags:
        sections.append(make_section("Tags", ", ".join(f"`{tag}`" for tag in algorithm.tags)))

    if algorithm.raw and algorithm.raw.content:
        sections.append(
            make_section(
                "Directive Content",
                f"```markdown\n{algorithm.raw.content.strip()}\n```",
            )
        )

    combined = "\n\n".join(part for part in sections if part).strip()
    return combined + "\n"


def _build_header_block(algorithm: Algorithm) -> str:
    title = algorithm.title or algorithm.label
    lines = [f"# {title}"]
    if algorithm.label:
        lines.append(f"**Label:** `{algorithm.label}`")
    if algorithm.complexity:
        lines.append(f"**Complexity:** {algorithm.complexity}")

    info_parts: list[str] = []
    if algorithm.doc_meta and algorithm.doc_meta.document_id:
        info_parts.append(f"Document: `{algorithm.doc_meta.document_id}`")
    if algorithm.doc_meta and algorithm.doc_meta.stage:
        info_parts.append(f"Stage: {algorithm.doc_meta.stage}")
    if algorithm.raw and algorithm.raw.section:
        info_parts.append(f"Section: {algorithm.raw.section}")
    line_range = _format_line_range(
        algorithm.raw.start_line if algorithm.raw else None,
        algorithm.raw.end_line if algorithm.raw else None,
    )
    if line_range:
        info_parts.append(line_range)
    if info_parts:
        lines.append(" • ".join(info_parts))
    return "\n".join(lines)


def _format_line_range(start: int | None, end: int | None) -> str | None:
    if start is None and end is None:
        return None
    start_val = start if start is not None else "?"
    end_val = end if end is not None else "?"
    return f"Lines {start_val}–{end_val}"


def _format_signature(signature: AlgorithmSignature) -> str:
    lines: list[str] = []
    if signature.input:
        lines.append("**Input:**" )
        lines.append(format_bullet_list(signature.input) or "- —")
    if signature.output:
        lines.append("\n**Output:**")
        lines.append(format_bullet_list(signature.output) or "- —")
    if signature.parameters:
        lines.append("\n**Parameters:**")
        lines.append(_format_algorithm_parameters(signature.parameters))
    return "\n".join(lines).strip()


def _format_algorithm_parameters(parameters: list[AlgorithmParameter]) -> str:
    lines: list[str] = []
    for param in parameters:
        pieces: list[str] = []
        if param.name:
            pieces.append(f"**{param.name}**")
        if param.type:
            pieces.append(f"type: {param.type}")
        if param.default is not None:
            pieces.append(f"default: {param.default}")
        if param.description:
            pieces.append(param.description.strip())
        lines.append("- " + "; ".join(pieces) if pieces else "- parameter")
    return "\n".join(lines)


def _format_steps(steps: list[AlgorithmStep]) -> str:
    rendered: list[str] = []
    ordered = sorted(steps, key=lambda step: step.order)
    for idx, step in enumerate(ordered, start=1):
        body = step.text.strip() if step.text else "—"
        if step.comment:
            body += f"\n   _Comment:_ {step.comment.strip()}"
        rendered.append(f"{idx}. {body}")
    return "\n".join(rendered)


def _format_guards(guards: list[GuardCondition]) -> str:
    lines: list[str] = []
    for guard in guards:
        pieces: list[str] = []
        if guard.condition:
            pieces.append(f"Condition: {guard.condition}")
        if guard.description:
            pieces.append(guard.description.strip())
        if guard.action:
            pieces.append(f"Action: {guard.action}")
        if guard.severity:
            pieces.append(f"Severity: {guard.severity}")
        lines.append("- " + " | ".join(pieces) if pieces else "- guard")
    return "\n".join(lines)


def _format_failure_modes(failure_modes: list[FailureMode]) -> str:
    lines: list[str] = []
    for mode in failure_modes:
        description = mode.description.strip() if mode.description else "—"
        impact = f" (impact: {mode.impact})" if mode.impact else ""
        lines.append(f"- {description}{impact}")
    return "\n".join(lines)

