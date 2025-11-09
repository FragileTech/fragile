#!/usr/bin/env python3
"""
Shared DSPy signature definitions for Mathster agents.
"""

from __future__ import annotations

from typing import Any, Literal

import dspy
from pydantic import BaseModel, Field

__all__ = [
    "ImplicitReference",
    "MathematicalTool",
    "Parameter",
    "ExtractSignature",
    "ExtractWithParametersSignature",
    "ParseAlgorithmDirectiveSplit",
    "ParseAssumptionDirectiveSplit",
    "ParseAxiomDirectiveSplit",
    "ParseConjectureDirectiveSplit",
    "ParseDefinitionDirectiveSplit",
    "ParseProofDirectiveSplit",
    "ParseRemarkDirectiveSplit",
    "ParseTheoremDirectiveSplit",
    "to_jsonable",
]


class ImplicitReference(BaseModel):
    """Reference without an explicit label inside the directive."""

    label: str = Field(..., description="Label of the referenced artifact.")
    context: str | None = Field(
        default=None,
        description="Optional brief context describing how the reference is used.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description='array of 3-10 keyword strings for search (e.g., ["greedy","pairing","diversity"]).',
    )


class ExtractSignature(dspy.Signature):
    """Shared base signature for directive extraction agents."""

    label_str: str = dspy.OutputField(desc="Directive label if present, else empty.")
    title_str: str = dspy.OutputField(desc="Human-facing title if present, else empty.")
    references: list[str] = dspy.OutputField(
        desc='JSON array of labels referencing other artifacts (e.g., ["def-...", "thm-...", ...])'
    )
    tags: list[str] = dspy.OutputField(
        desc='JSON array of 3-10 keyword strings for search (e.g., ["greedy","pairing","diversity"]).'
    )
    implicit_references: list[ImplicitReference] = dspy.OutputField(
        desc='JSON array of objects representing references with no explicit label [{"label": str, "context": str|null, "tags": [str,...]}, ...]'
    )


class Parameter(BaseModel):
    """Structured representation for directive parameters."""

    symbol: str = Field(..., description="Parameter symbol.")
    description: str | None = Field(default=None, description="Parameter description.")
    constraints: list[str] = Field(default_factory=list, description="Parameter constraints.")
    tags: list[str] = Field(
        default_factory=list,
        description='array of 3-10 keyword strings for search (e.g., ["greedy","pairing","diversity"]).',
    )


class MathematicalTool(BaseModel):
    """Structured description of a mathematical tool referenced in a proof."""

    toolName: str = Field(..., description="Commonly accepted name of the tool.")
    field: str = Field(..., description="Primary mathematical discipline (e.g., Optimal Transport).")
    description: str = Field(..., description="Brief general-purpose definition of the tool.")
    roleInProof: str = Field(..., description="Specific role/use within this proof.")
    levelOfAbstraction: Literal["Concept", "Technique", "Theorem/Lemma", "Notation"] | None = Field(
        default=None,
        description="Classification of the tool's nature.",
    )
    relatedTools: list[str] = Field(
        default_factory=list,
        description="Optional list of other tools referenced in this proof that are closely related.",
    )


class ExtractWithParametersSignature(ExtractSignature):
    """Extension of ExtractSignature including parameter extraction."""

    parameters: list[Parameter] = dspy.OutputField(
        desc='JSON array [{"symbol": str, "description": str|null, "constraints": [str,...], "tags": [str,...]}, ...]'
    )


class ParseAlgorithmDirectiveSplit(ExtractSignature):
    """
    Convert a `::{prf:algorithm}` directive into a structured representation.
    """

    directive_text = dspy.InputField(desc="Raw algorithm directive text.")
    context_hints = dspy.InputField(desc="Optional context window.", optional=True)

    label_str = dspy.OutputField(desc="Algorithm label.")
    title_str = dspy.OutputField(desc="Title if present.")
    complexity_str = dspy.OutputField(desc="Complexity classification.")
    nl_summary_str = dspy.OutputField(desc="Concise summary of the algorithm.")

    signature_json = dspy.OutputField(
        desc='JSON object {"input": [str,...], "output": [str,...], "parameters": [str,...]}'
    )
    steps_json = dspy.OutputField(
        desc='JSON array [{"order": int|null, "text": str|null, "latex": str|null, "comment": str|null}, ...]'
    )
    guard_conditions_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...]'
    )
    failure_modes_json = dspy.OutputField(
        desc='JSON array [{"description": str|null, "impact": str|null}, ...]'
    )


class ParseAssumptionDirectiveSplit(ExtractWithParametersSignature):
    """
    Convert a `::{prf:assumption}` directive into structured analysis artifacts.
    """

    directive_text = dspy.InputField(desc="Raw assumption directive text.")
    context_hints = dspy.InputField(desc="Optional nearby prose.", optional=True)

    label_str = dspy.OutputField(desc="Directive label (`assump-*`).")
    title_str = dspy.OutputField(desc="Title/heading if present.")
    scope_str = dspy.OutputField(desc="Scope classification (global/local/model/etc.).")
    nl_summary_str = dspy.OutputField(desc="Concise natural-language summary.")

    bullet_items_json = dspy.OutputField(
        desc='JSON array [{"name": str|null, "text": str|null, "latex": str|null}, ...]'
    )
    conditions_json = dspy.OutputField(
        desc='JSON array [{"type": str|null, "text": str|null, "latex": str|null}, ...]'
    )
    notes_json = dspy.OutputField(desc='JSON array [{"type": str|null, "text": str|null}, ...]')


class ParseAxiomDirectiveSplit(ExtractWithParametersSignature):
    """
    Convert a `::{prf:axiom}` directive into structured semantic content.
    """

    directive_text = dspy.InputField(desc="Raw axiom directive text.")
    context_hints = dspy.InputField(desc="Optional nearby context snippet.", optional=True)

    label_str = dspy.OutputField(desc="Directive label.")
    title_str = dspy.OutputField(desc="Title if present.")
    axiom_class_str = dspy.OutputField(desc="Classification (structural, regularity, etc.).")
    nl_summary_str = dspy.OutputField(desc="Concise natural-language summary.")

    core_statement_json = dspy.OutputField(
        desc='JSON object {"text": str|null, "latex": str|null}'
    )
    hypotheses_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...]'
    )
    implications_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...]'
    )
    failure_modes_json = dspy.OutputField(
        desc='JSON array [{"description": str|null, "impact": str|null}, ...]'
    )


class ParseConjectureDirectiveSplit(ExtractWithParametersSignature):
    """
    Convert a `::{prf:conjecture}` directive into a structured representation.
    """

    directive_text = dspy.InputField(desc="Raw conjecture directive text.")
    context_hints = dspy.InputField(desc="Optional nearby prose.", optional=True)

    label_str = dspy.OutputField(desc="Conjecture label.")
    title_str = dspy.OutputField(desc="Title if present.")
    conjecture_type_str = dspy.OutputField(desc="Conjecture classification.")
    status_str = dspy.OutputField(desc="Status (open/partial/etc.).")
    nl_summary_str = dspy.OutputField(desc="Concise summary.")

    statement_json = dspy.OutputField(desc='JSON object {"text": str|null, "latex": str|null}')
    evidence_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...]'
    )
    obstacles_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...]'
    )
    recommended_paths_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "priority": str|null}, ...]'
    )


class ParseDefinitionDirectiveSplit(ExtractWithParametersSignature):
    """
    Transform a raw `::{prf:definition}` directive into a structured bundle.
    """

    directive_text = dspy.InputField(desc="Raw definition directive text (header/body).")
    context_hints = dspy.InputField(
        desc="Optional nearby prose for scope/motivation.", optional=True
    )

    label_str = dspy.OutputField(desc="Definition label (def-*).")
    term_str = dspy.OutputField(desc="Term being defined.")
    object_type_str = dspy.OutputField(
        desc="Category of mathematical object (set/function/operator/process/...)."
    )
    nl_definition_str = dspy.OutputField(desc="Concise natural-language paraphrase.")

    formal_conditions_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...]'
    )
    properties_json = dspy.OutputField(
        desc='JSON array [{"name": str|null, "description": str|null}, ...]'
    )
    examples_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...]'
    )
    notes_json = dspy.OutputField(desc='JSON array [{"type": str|null, "text": str|null}, ...]')


class ParseProofDirectiveSplit(ExtractSignature):
    """
    Convert a `::{prf:proof}` directive into a structured description.
    """

    directive_text = dspy.InputField(desc="Raw proof directive text (header/body).")
    context_hints = dspy.InputField(desc="Optional nearby prose window.", optional=True)

    label_str = dspy.OutputField(desc="Proof label (`proof-*`).")
    proves_label_str = dspy.OutputField(desc="Label proved (e.g., `thm-main`).")
    proof_type_str = dspy.OutputField(
        desc="Technique: direct/contradiction/induction/reference/construction/variational/probabilistic/other."
    )
    proof_status_str = dspy.OutputField(desc="Status: complete/sketch/omitted/by-reference.")
    strategy_summary_str = dspy.OutputField(desc="Short strategy description (1-2 sentences).")
    math_tools: list[MathematicalTool] = dspy.OutputField(
        desc=(
            "list of mathematical tool objects. Each must include "
            '{"toolName": str, "field": str, "description": str, "roleInProof": str, '
            '"levelOfAbstraction": "Concept|Technique|Theorem/Lemma|Notation"|null, '
            '"relatedTools": [str,...]}.'
        )
    )
    conclusion_json = dspy.OutputField(desc='JSON object {"text": str|null, "latex": str|null}.')
    assumptions_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...].'
    )
    steps_json = dspy.OutputField(
        desc='JSON array [{"order": int|null, "kind": str|null, "text": str|null, "latex": str|null, "references": [str,...], "derived_statement": str|null}, ...].'
    )
    key_equations_json = dspy.OutputField(
        desc='JSON array [{"label": str|null, "latex": str, "role": str|null}, ...]. If labels are not present use eq-{equation-name}'
    )
    cases_json = dspy.OutputField(
        desc='JSON array [{"name": str|null, "condition": str|null, "summary": str|null}, ...].'
    )
    remarks_json = dspy.OutputField(desc='JSON array [{"type": str|null, "text": str|null}, ...].')
    gaps_json = dspy.OutputField(
        desc='JSON array [{"description": str, "severity": str|null, "location_hint": str|null}, ...].'
    )


class ParseRemarkDirectiveSplit(ExtractSignature):
    """
    Convert a `::{prf:remark}` directive into structured content.
    """

    directive_text = dspy.InputField(desc="Raw remark directive text.")
    context_hints = dspy.InputField(desc="Optional nearby prose.", optional=True)

    label_str = dspy.OutputField(desc="Remark label.")
    title_str = dspy.OutputField(desc="Title if present.")
    remark_type_str = dspy.OutputField(desc="Remark classification.")
    nl_summary_str = dspy.OutputField(desc="Natural-language summary.")

    key_points_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null, "importance": str|null}, ...]'
    )
    quantitative_notes_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...]'
    )
    recommendations_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "severity": str|null}, ...]'
    )
    dependencies_json = dspy.OutputField(desc='JSON array of strings ["sec-2.1","rem-other",...]')


class ParseTheoremDirectiveSplit(ExtractSignature):
    """
    Transform one raw theorem directive into a compact structured representation.
    """

    directive_text = dspy.InputField(desc="Raw theorem directive text (header + body).")
    context_hints = dspy.InputField(
        desc="Tiny local context to infer implicit assumptions.", optional=True
    )

    type_str = dspy.OutputField(desc="One of: 'theorem','lemma','proposition',…")
    label_str = dspy.OutputField(desc="Directive label if present, else empty.")
    title_str = dspy.OutputField(desc="Human-facing title if present, else empty.")
    nl_statement_str = dspy.OutputField(desc="Concise natural-language statement only.")

    equations_json = dspy.OutputField(
        desc='JSON array: [{"label": string|null, "latex": string}, ...]'
    )
    hypotheses_json = dspy.OutputField(
        desc='JSON array: [{"text": string, "latex": string|null}, ...]'
    )
    conclusion_json = dspy.OutputField(
        desc='JSON object: {"text": string|null, "latex": string|null}'
    )
    variables_json: list[Parameter] = dspy.OutputField(
        desc='list: [{"symbol": string, "role": string|null, "constraints": [string,...], "tags": [string,...]}, ...]'
    )
    implicit_assumptions_json = dspy.OutputField(
        desc='JSON array: [{"text": string, "confidence": number|null}, ...]'
    )
    proof_json = dspy.OutputField(
        desc='JSON object: {"availability": "...", "steps":[{"kind": "...", "text": "...", "latex": "..."}]}'
    )


def to_jsonable(value: Any) -> Any:
    """
    Convert nested BaseModel/list/dict structures into JSON-serializable primitives.
    """

    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return value
