"""Markdown renderer for :class:`UnifiedProof`."""

from __future__ import annotations

from mathster.preprocess_extraction.data_models import UnifiedProof
from mathster.reports.report_theorem import format_unified_proof
from mathster.reports.report_utils import (
    format_metadata,
    format_reference_labels,
    format_span,
    make_section,
)

__all__ = ["unified_proof_to_markdown"]


def unified_proof_to_markdown(proof: UnifiedProof) -> str:
    """Return a Markdown report describing ``proof``."""

    sections: list[str] = []

    reference_labels = format_reference_labels(proof.references)
    reference_line = (
        f"**Reference labels:** {reference_labels}" if reference_labels else "**Reference labels:** _none_"
    )
    sections.append(reference_line)

    sections.append(_build_header_block(proof))
    sections.append(make_section("Proof Body", format_unified_proof(proof)))

    if proof.content_markdown:
        sections.append(
            make_section(
                "Directive Content",
                f"```markdown\n{proof.content_markdown.strip()}\n```",
            )
        )

    metadata_block = format_metadata(proof.metadata, proof.registry_context)
    if metadata_block:
        sections.append(make_section("Metadata", metadata_block))

    combined = "\n\n".join(part for part in sections if part).strip()
    return combined + "\n"


def _build_header_block(proof: UnifiedProof) -> str:
    title = proof.title or proof.label
    lines = [f"# {title}"]
    if proof.label:
        lines.append(f"**Label:** `{proof.label}`")
    lines.append(f"**Type:** {proof.type}")
    if proof.proves:
        lines.append(f"**Proves:** {proof.proves}")

    info_parts: list[str] = []
    if proof.document_id:
        info_parts.append(f"Document: `{proof.document_id}`")
    if proof.section:
        info_parts.append(f"Section: {proof.section}")
    span = format_span(proof.span)
    if span:
        info_parts.append(span)
    if proof.generated_at:
        info_parts.append(f"Generated at: {proof.generated_at}")
    if info_parts:
        lines.append(" • ".join(info_parts))
    return "\n".join(lines)

