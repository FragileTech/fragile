"""Unified data models for mathematical entity preprocessing.

This module consolidates ALL Pydantic models for mathematical entity preprocessing
into a single source of truth.

Architecture:
    - Section 1: Shared nested models (used across multiple entity types)
    - Section 2: Theorem-like entities (theorem, lemma, corollary, proposition)
    - Sections 3-8: Entity-specific models (algorithm, assumption, axiom, definition, proof, remark)

Design Philosophy:
    - DRY: Single source of truth for all data structures
    - Type Safety: Strict Pydantic validation throughout
    - Extensibility: Easy to add new entity types
    - Organization: Clear sections by entity type
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger(__name__)


def _reference_identity(value: Any) -> str:
    """Return a stable identity for a reference entry."""

    if isinstance(value, dict):
        label = value.get("label")
        if isinstance(label, str):
            return f"dict:{label}"
        try:
            return "dict:" + json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            return "dict:" + repr(value)
    return f"value:{value!r}"


def _merge_reference_labels(*sources: Any) -> list[Any]:
    """Merge multiple reference lists while preserving order and removing duplicates."""

    merged: list[Any] = []
    seen: set[str] = set()

    for source in sources:
        if not source:
            continue
        if isinstance(source, list | tuple | set):
            iterator = source
        else:
            iterator = [source]
        for entry in iterator:
            if entry is None:
                continue
            key = _reference_identity(entry)
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry)
    return merged


# ============================================================================
# SECTION 1: SHARED NESTED MODELS
# Used by multiple entity types across the preprocessing pipeline
# ============================================================================


class Equation(BaseModel):
    """Mathematical equation with optional reference label."""

    label: str | None = None
    latex: str


class Hypothesis(BaseModel):
    """Assumption or condition in a mathematical statement.

    Can be expressed in natural language, LaTeX, or both.
    """

    text: str | None = None
    latex: str | None = None


class Conclusion(BaseModel):
    """Main result of a mathematical statement.

    Can be expressed in natural language, LaTeX, or both.
    """

    text: str | None = None
    latex: str | None = None


class Variable(BaseModel):
    """Mathematical symbol with semantic metadata.

    Includes symbol notation, description, constraints, and classification tags.
    """

    symbol: str | None = None
    name: str | None = None
    description: str | None = None
    constraints: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class Assumption(BaseModel):
    """Implicit assumption with optional confidence score.

    Used for assumptions not explicitly stated but required for validity.
    Used by theorem-like entities (theorem, lemma, corollary, proposition).
    """

    text: str
    confidence: float | None = None


class ProofStep(BaseModel):
    """Single step in a proof (extended version).

    Attributes:
        order: Optional ordering (float allows fractional steps)
        kind: Type of step (e.g., "calculation", "argument", "reference")
        text: Natural language description
        latex: LaTeX representation
        references: Cross-references to other entities (extended field)
        derived_statement: Statement derived in this step (extended field)
    """

    order: float | None = None
    kind: str | None = None
    text: str | None = None
    latex: str | None = None
    references: list[str] = Field(default_factory=list)
    derived_statement: str | None = None


class Proof(BaseModel):
    """Proof structure with availability status and steps.

    Attributes:
        availability: Status like "not-provided", "sketched", "complete"
        steps: Sequence of proof steps
    """

    availability: str | None = None
    steps: list[ProofStep] = Field(default_factory=list)


class Span(BaseModel):
    """Line number positioning for directive location in source file.

    Tracks precise location of mathematical entities in markdown documents.

    Attributes:
        start_line: First line of directive block
        end_line: Last line of directive block
        content_start: First line of actual content (after directive header)
        content_end: Last line of content (before closing marker)
        header_lines: Lines containing section headers
    """

    start_line: int | None = None
    end_line: int | None = None
    content_start: int | None = None
    content_end: int | None = None
    header_lines: list[int] = Field(default_factory=list)


class Parameter(BaseModel):
    """Mathematical parameter with semantic metadata.

    Shared across assumptions, axioms, and definitions.
    """

    symbol: str | None = None
    name: str | None = None
    description: str | None = None
    constraints: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class Note(BaseModel):
    """Explanatory note or comment.

    Shared across assumptions and definitions.
    """

    type: str | None = None
    text: str | None = None


class FailureMode(BaseModel):
    """Failure mode description.

    Shared across algorithms and axioms.
    """

    description: str
    impact: str | None = None

    class Config:
        extra = "allow"


class Condition(BaseModel):
    """Formal condition or constraint.

    Shared across assumptions and definitions.
    """

    type: str | None = None  # Optional for flexibility
    text: str | None = None
    latex: str | None = None


# ============================================================================
# SECTION 2: THEOREM-LIKE ENTITIES
# Base class and concrete types (theorem, lemma, corollary, proposition)
# ============================================================================


class UnifiedMathematicalEntity(BaseModel):
    """Base class for all theorem-like mathematical entities.

    This class provides the complete structure shared by theorems, lemmas,
    corollaries, and propositions. Subclasses need only override the `type`
    field default value.

    Architecture:
        - Identity fields: label, title, type
        - Extracted semantic content: equations, hypotheses, conclusion, etc.
        - Raw directive content: content_markdown, raw_directive
        - Positioning metadata: document_id, section, span
        - Provenance: references, registry_context, generated_at

    Design:
        - Flat structure for all positioning fields (no nested locators)
        - Strict Pydantic validation
        - Shared methods: strip_line_numbers(), from_instances()
    """

    # === IDENTITY ===
    label: str = Field(..., description="Unique identifier (e.g., 'thm-kl-convergence')")
    title: str | None = Field(default=None, description="Human-readable name")
    type: str = Field(
        default="entity",
        description="Entity type discriminator (theorem, lemma, corollary, proposition)",
    )

    # === EXTRACTED SEMANTIC CONTENT ===
    nl_statement: str | None = Field(
        default=None,
        description="Natural language summary from LLM extraction",
    )
    equations: list[Equation] = Field(
        default_factory=list,
        description="LaTeX equations with optional labels",
    )
    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description="Conditions and assumptions",
    )
    conclusion: Conclusion | None = Field(
        default=None,
        description="Main result statement",
    )
    variables: list[Variable] = Field(
        default_factory=list,
        description="Mathematical symbols with semantic metadata",
    )
    implicit_assumptions: list[Assumption] = Field(
        default_factory=list,
        description="Implicit assumptions with confidence scores",
    )
    local_refs: list[str] = Field(
        default_factory=list,
        description="Cross-references to other entities",
    )
    proof: UnifiedProof | None = Field(
        default=None,
        description="Attached proof object built from proof directives/extraction",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Classification tags for search/filtering",
    )

    # === RAW DIRECTIVE CONTENT ===
    content_markdown: str | None = Field(
        default=None,
        description="Cleaned markdown content (line numbers stripped)",
    )
    raw_directive: str | None = Field(
        default=None,
        description="Full raw directive block with all metadata",
    )

    # === POSITIONING & PROVENANCE ===
    document_id: str | None = Field(
        default=None,
        description="Source document identifier (e.g., '03_cloning')",
    )
    section: str | None = Field(
        default=None,
        description="Section heading context",
    )
    span: Span | None = Field(
        default=None,
        description="Line number positioning in source file",
    )
    references: list[Any] = Field(
        default_factory=list,
        description="Cross-references from directive metadata",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata from directive",
    )
    registry_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Stage, chapter, and section context from registry",
    )
    generated_at: str | None = Field(
        default=None,
        description="ISO timestamp of generation",
    )
    alt_labels: list[str] = Field(
        default_factory=list,
        description="Alternative labels if mismatch between extracted and directive",
    )

    # ========================================================================
    # SHARED METHODS
    # ========================================================================

    _proof_lookup: ClassVar[dict[str, list[UnifiedProof]]] = {}

    @classmethod
    def attach_proof_lookup(cls, proof_lookup: dict[str, list[UnifiedProof]] | None) -> None:
        """Install a lookup of proofs keyed by the statements they prove."""
        UnifiedMathematicalEntity._proof_lookup = proof_lookup or {}

    @classmethod
    def _select_proof_for_label(cls, label: str) -> UnifiedProof | None:
        proofs = UnifiedMathematicalEntity._proof_lookup.get(label)
        if not proofs:
            return None
        if len(proofs) > 1:
            logger.warning(
                "Multiple proofs found for %s; selecting the longest entry.",
                label,
            )
        return max(proofs, key=UnifiedMathematicalEntity._proof_length_score)

    @staticmethod
    def _proof_length_score(proof: UnifiedProof) -> int:
        """Heuristic length used to pick the most detailed proof."""
        step_count = len(proof.steps or [])
        step_text_len = sum(len(step.text or "") for step in proof.steps or [])
        content_len = len(proof.content_markdown or proof.raw_directive or "")
        summary_len = len(proof.strategy_summary or "")
        return step_count * 1000 + step_text_len + content_len + summary_len

    @staticmethod
    def strip_line_numbers(text: str | None) -> str | None:
        """Remove leading 'NNN: ' prefixes from directive content.

        Directive content from the registry includes line numbers like:
            615: Content here
            616: More content

        This method strips those prefixes to get clean markdown.

        Args:
            text: Raw directive content with line number prefixes

        Returns:
            Cleaned text without line number prefixes, or None if input is None
        """
        if not text:
            return text
        out_lines = []
        for line in text.splitlines():
            # Remove leading integer + colon + optional space (e.g., "615: " or "615:")
            out_lines.append(re.sub(r"^\s*\d+:\s?", "", line))
        return "\n".join(out_lines).strip()

    @classmethod
    def from_instances(
        cls,
        directive: dict[str, Any],
        extracted: dict[str, Any],
    ) -> UnifiedMathematicalEntity:
        """Merge directive and extracted data into unified entity.

        This method combines two data sources:
            1. Directive: Raw positioning metadata from directive parsing
            2. Extracted: Semantic content from LLM extraction

        The merge strategy:
            - Identity: Prefer extracted label, fallback to directive
            - Semantic content: Entirely from extracted
            - Raw content: From directive (with line number stripping)
            - Positioning: From directive
            - Provenance: From directive container metadata

        Args:
            directive: Dict from directives/*.json (may be container with 'items')
            extracted: Dict from extract/*_extracted.json

        Returns:
            Unified entity with merged data

        Raises:
            Exception: If data cannot be parsed or merged
        """
        # Handle directive container vs item
        directive_container = None
        directive_item = directive
        if "items" in directive and isinstance(directive["items"], list):
            directive_container = directive
            # Match by label
            ex_label = extracted.get("label")
            matched = next(
                (it for it in directive["items"] if it.get("label") == ex_label),
                None,
            )
            directive_item = matched or (directive["items"][0] if directive["items"] else {})

        # === IDENTITY ===
        label_ex = extracted.get("label")
        label_dir = directive_item.get("label")
        label = label_ex or label_dir or ""

        title = extracted.get("title") or directive_item.get("title")

        # === SEMANTIC CONTENT (from extracted) ===
        equations = [Equation(**e) for e in extracted.get("equations", []) or []]
        hypotheses = [Hypothesis(**h) for h in extracted.get("hypotheses", []) or []]

        conclusion_data = extracted.get("conclusion")
        conclusion = Conclusion(**conclusion_data) if isinstance(conclusion_data, dict) else None

        variables = [Variable(**v) for v in extracted.get("variables", []) or []]
        implicit_assumptions = [
            Assumption(**a) for a in extracted.get("implicit_assumptions", []) or []
        ]
        local_refs = extracted.get("local_refs", []) or []

        proof = cls._select_proof_for_label(label)

        tags = extracted.get("tags", []) or []
        nl_statement = extracted.get("nl_statement")

        # === RAW CONTENT (from directive, cleaned) ===
        raw_content = directive_item.get("content")
        content_markdown = cls.strip_line_numbers(raw_content) if raw_content else None
        raw_directive = directive_item.get("raw_directive")

        # === POSITIONING (from directive) ===
        span = (
            Span(
                start_line=directive_item.get("start_line"),
                end_line=directive_item.get("end_line"),
                content_start=directive_item.get("content_start"),
                content_end=directive_item.get("content_end"),
                header_lines=directive_item.get("header_lines", []) or [],
            )
            if directive_item
            else None
        )

        # === PROVENANCE (from directive container) ===
        if directive_container:
            document_id = directive_container.get("document_id")
            generated_at = directive_container.get("generated_at")
        else:
            # Fallback: recover from item's registry context
            rc = directive_item.get("_registry_context", {}) or {}
            document_id = rc.get("document_id")
            generated_at = None

        registry_context = directive_item.get("_registry_context", {}) or {}
        metadata = directive_item.get("metadata", {}) or {}
        section = directive_item.get("section")
        references = _merge_reference_labels(
            directive_item.get("references"),
            extracted.get("references"),
        )
        if proof and proof.references:
            references = _merge_reference_labels(references, proof.references)

        # === ALT LABELS (if mismatch) ===
        alt_labels = []
        if label_ex and label_dir and label_ex != label_dir:
            alt_labels = [label_ex, label_dir]

        # === CONSTRUCT UNIFIED ENTITY ===
        return cls(
            # Identity
            label=label,
            title=title,
            type=extracted.get("type", cls.model_fields["type"].default),
            # Semantic content
            nl_statement=nl_statement,
            equations=equations,
            hypotheses=hypotheses,
            conclusion=conclusion,
            variables=variables,
            implicit_assumptions=implicit_assumptions,
            local_refs=local_refs,
            proof=proof,
            tags=tags,
            # Raw content
            content_markdown=content_markdown,
            raw_directive=raw_directive,
            # Positioning & provenance
            document_id=document_id,
            section=section,
            span=span,
            references=references,
            metadata=metadata,
            registry_context=registry_context,
            generated_at=generated_at,
            alt_labels=alt_labels,
        )


class UnifiedTheorem(UnifiedMathematicalEntity):
    """Theorem entity.

    A theorem is a major mathematical statement that has been proven or is
    presented for proof.
    """

    type: str = Field(
        default="theorem",
        description="Entity type discriminator",
    )


class UnifiedLemma(UnifiedMathematicalEntity):
    """Lemma entity.

    A lemma is a subsidiary mathematical statement proven to support a main
    theorem or result.
    """

    type: str = Field(
        default="lemma",
        description="Entity type discriminator",
    )


class UnifiedCorollary(UnifiedMathematicalEntity):
    """Corollary entity.

    A corollary is a mathematical statement that follows directly from a
    previously proven theorem or lemma.
    """

    type: str = Field(
        default="corollary",
        description="Entity type discriminator",
    )


class UnifiedProposition(UnifiedMathematicalEntity):
    """Proposition entity.

    A proposition is a mathematical statement of intermediate importance,
    typically more significant than a lemma but less central than a theorem.

    Note: This class uses the same flat structure as Theorem/Lemma/Corollary,
    replacing the previous nested RawLocator design for consistency.
    """

    type: str = Field(
        default="proposition",
        description="Entity type discriminator",
    )


# ============================================================================
# SECTION 3: ALGORITHM-SPECIFIC MODELS
# ============================================================================


class DocumentMetadata(BaseModel):
    """Document-level metadata."""

    document_id: str | None = None
    stage: str | None = None
    generated_at: str | None = None

    class Config:
        extra = "allow"


class AlgorithmParameter(BaseModel):
    """A flexible parameter representation for algorithms."""

    name: str
    type: str | None = None
    default: Any | None = None
    description: str | None = None

    @classmethod
    def from_any(cls, obj: str | dict[str, Any]) -> AlgorithmParameter:
        """Create from string name or dict."""
        if isinstance(obj, str):
            return cls(name=obj)
        name = obj.get("name") or obj.get("id") or obj.get("param") or "param"
        return cls(
            name=name,
            type=obj.get("type"),
            default=obj.get("default"),
            description=obj.get("description"),
        )


class AlgorithmSignature(BaseModel):
    """Structured I/O signature of an algorithm."""

    input: list[str] = Field(default_factory=list)
    output: list[str] = Field(default_factory=list)
    parameters: list[AlgorithmParameter] = Field(default_factory=list)

    @field_validator("parameters", mode="before")
    @classmethod
    def _coerce_parameters(cls, v):
        if v is None:
            return []
        coerced: list[AlgorithmParameter] = []
        for item in v:
            coerced.append(AlgorithmParameter.from_any(item))
        return coerced


class AlgorithmStep(BaseModel):
    """Single step in an algorithm."""

    order: int
    text: str
    comment: str | None = None

    class Config:
        extra = "allow"


class GuardCondition(BaseModel):
    """Guard condition in an algorithm."""

    condition: str | None = None
    description: str | None = None
    action: str | None = None
    severity: str | None = None

    class Config:
        extra = "allow"


class RawAlgorithm(BaseModel):
    """Verbatim raw directive slice from algorithm.json."""

    label: str | None = None
    title: str | None = None
    section: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    content_start: int | None = None
    content_end: int | None = None
    header_lines: list[int] | None = None
    raw_directive: str | None = None
    content: str | None = None

    class Config:
        extra = "allow"


class ExtractedAlgorithm(BaseModel):
    """Structured fields from algorithm_extracted.json."""

    label: str | None = None
    title: str | None = None
    complexity: str | None = None
    nl_summary: str | None = None
    signature: AlgorithmSignature | None = None
    steps: list[AlgorithmStep] = Field(default_factory=list)
    guard_conditions: list[GuardCondition] = Field(default_factory=list)
    references: list[str | dict[str, Any]] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    class Config:
        extra = "allow"


class Algorithm(BaseModel):
    """Unified Algorithm model combining raw directive and structured extraction."""

    label: str
    title: str | None = None
    complexity: str | None = None
    nl_summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    signature: AlgorithmSignature | None = None
    steps: list[AlgorithmStep] = Field(default_factory=list)
    guard_conditions: list[GuardCondition] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    references: list[str | dict[str, Any]] = Field(default_factory=list)
    raw: RawAlgorithm | None = None
    extracted: ExtractedAlgorithm | None = None
    doc_meta: DocumentMetadata | None = None

    class Config:
        extra = "allow"

    @classmethod
    def from_instances(
        cls, raw: dict[str, Any], extracted: dict[str, Any], doc_meta: dict[str, Any] | None = None
    ) -> Algorithm:
        """Merge raw and extracted algorithm data."""
        # Implementation preserved from process_algorithms.py
        raw_obj = RawAlgorithm(**raw) if raw else None
        extracted_obj = ExtractedAlgorithm(**extracted) if extracted else None
        doc_meta_obj = DocumentMetadata(**doc_meta) if doc_meta else None

        label = extracted.get("label") or (raw.get("label") if raw else None) or "unknown"
        title = extracted.get("title") or (raw.get("title") if raw else None)

        raw_references = raw.get("references") if raw else None
        extracted_references = extracted_obj.references if extracted_obj else None
        merged_references = _merge_reference_labels(raw_references, extracted_references)

        return cls(
            label=label,
            title=title,
            complexity=extracted.get("complexity"),
            nl_summary=extracted.get("nl_summary"),
            tags=extracted.get("tags", []),
            signature=extracted_obj.signature if extracted_obj else None,
            steps=extracted_obj.steps if extracted_obj else [],
            guard_conditions=extracted_obj.guard_conditions if extracted_obj else [],
            failure_modes=extracted_obj.failure_modes if extracted_obj else [],
            references=merged_references,
            raw=raw_obj,
            extracted=extracted_obj,
            doc_meta=doc_meta_obj,
        )


# ============================================================================
# SECTION 4: ASSUMPTION-SPECIFIC MODELS
# ============================================================================


class BulletItem(BaseModel):
    """Bullet point item in an assumption."""

    name: str | None = None
    text: str | None = None
    latex: str | None = None


class UnifiedAssumption(BaseModel):
    """Unified assumption model."""

    label: str
    type: str = Field(default="assumption")
    title: str | None = None
    scope: str | None = None
    nl_summary: str | None = None
    content_markdown: str | None = None
    raw_directive: str | None = None
    bullet_items: list[BulletItem] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    parameters: list[Parameter] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    references: list[Any] = Field(default_factory=list)
    document_id: str | None = None
    section: str | None = None
    span: Span | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    registry_context: dict[str, Any] = Field(default_factory=dict)
    generated_at: str | None = None
    alt_labels: list[str] = Field(default_factory=list)

    @staticmethod
    def _strip_line_numbers(text: str | None) -> str | None:
        """Remove leading 'NNN: ' prefixes from directive content."""
        if not text:
            return text
        out_lines = []
        for line in text.splitlines():
            out_lines.append(re.sub(r"^\s*\d+:\s?", "", line))
        return "\n".join(out_lines).strip()

    @classmethod
    def from_instances(
        cls, directive: dict[str, Any], extracted: dict[str, Any]
    ) -> UnifiedAssumption:
        """Merge directive and extracted assumption data."""
        # Implementation preserved from process_assumptions.py
        directive_container = None
        directive_item = directive
        if "items" in directive and isinstance(directive["items"], list):
            directive_container = directive
            ex_label = extracted.get("label")
            matched = next(
                (it for it in directive["items"] if it.get("label") == ex_label),
                None,
            )
            directive_item = matched or (directive["items"][0] if directive["items"] else {})

        label_ex = extracted.get("label")
        label_dir = directive_item.get("label")
        label = label_ex or label_dir or ""
        title = extracted.get("title") or directive_item.get("title")

        raw_content = directive_item.get("content")
        content_markdown = cls._strip_line_numbers(raw_content) if raw_content else None

        span = (
            Span(
                start_line=directive_item.get("start_line"),
                end_line=directive_item.get("end_line"),
                content_start=directive_item.get("content_start"),
                content_end=directive_item.get("content_end"),
                header_lines=directive_item.get("header_lines", []) or [],
            )
            if directive_item
            else None
        )

        if directive_container:
            document_id = directive_container.get("document_id")
            generated_at = directive_container.get("generated_at")
        else:
            rc = directive_item.get("_registry_context", {}) or {}
            document_id = rc.get("document_id")
            generated_at = None

        registry_context = directive_item.get("_registry_context", {}) or {}
        metadata = directive_item.get("metadata", {}) or {}
        section = directive_item.get("section")
        references = _merge_reference_labels(
            directive_item.get("references"),
            extracted.get("references"),
        )

        bullet_items = [BulletItem(**b) for b in extracted.get("bullet_items", []) or []]
        conditions = [Condition(**c) for c in extracted.get("conditions", []) or []]
        parameters = [Parameter(**p) for p in extracted.get("parameters", []) or []]
        notes = [Note(**n) for n in extracted.get("notes", []) or []]

        alt_labels = []
        if label_ex and label_dir and label_ex != label_dir:
            alt_labels = [label_ex, label_dir]

        return cls(
            label=label,
            title=title,
            type=extracted.get("type", "assumption"),
            scope=extracted.get("scope"),
            nl_summary=extracted.get("nl_summary"),
            content_markdown=content_markdown,
            raw_directive=directive_item.get("raw_directive"),
            bullet_items=bullet_items,
            conditions=conditions,
            parameters=parameters,
            notes=notes,
            tags=extracted.get("tags", []),
            references=references,
            document_id=document_id,
            section=section,
            span=span,
            metadata=metadata,
            registry_context=registry_context,
            generated_at=generated_at,
            alt_labels=alt_labels,
        )


# ============================================================================
# SECTION 5: AXIOM-SPECIFIC MODELS
# ============================================================================


class CoreStatement(BaseModel):
    """Core statement of an axiom."""

    text: str | None = None
    latex: str | None = None


class Implication(BaseModel):
    """Implication or consequence of an axiom."""

    text: str | None = None
    latex: str | None = None


class UnifiedAxiom(BaseModel):
    """Unified axiom model."""

    label: str
    title: str | None = None
    type: str = Field(default="axiom")
    axiom_class: str | None = None
    nl_summary: str | None = None
    core_statement: CoreStatement | None = None
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    implications: list[Implication] = Field(default_factory=list)
    parameters: list[Parameter] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    content_markdown: str | None = None
    raw_directive: str | None = None
    references: list[Any] = Field(default_factory=list)
    document_id: str | None = None
    section: str | None = None
    span: Span | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    registry_context: dict[str, Any] = Field(default_factory=dict)
    generated_at: str | None = None
    alt_labels: list[str] = Field(default_factory=list)

    @staticmethod
    def _strip_line_numbers(text: str | None) -> str | None:
        """Remove leading 'NNN: ' prefixes from directive content."""
        if not text:
            return text
        out_lines = []
        for line in text.splitlines():
            out_lines.append(re.sub(r"^\s*\d+:\s?", "", line))
        return "\n".join(out_lines).strip()

    @classmethod
    def from_instances(cls, directive: dict[str, Any], extracted: dict[str, Any]) -> UnifiedAxiom:
        """Merge directive and extracted axiom data."""
        # Implementation preserved from process_axioms.py
        directive_container = None
        directive_item = directive
        if "items" in directive and isinstance(directive["items"], list):
            directive_container = directive
            ex_label = extracted.get("label")
            matched = next(
                (it for it in directive["items"] if it.get("label") == ex_label),
                None,
            )
            directive_item = matched or (directive["items"][0] if directive["items"] else {})

        label_ex = extracted.get("label")
        label_dir = directive_item.get("label")
        label = label_ex or label_dir or ""
        title = extracted.get("title") or directive_item.get("title")

        raw_content = directive_item.get("content")
        content_markdown = cls._strip_line_numbers(raw_content) if raw_content else None

        span = (
            Span(
                start_line=directive_item.get("start_line"),
                end_line=directive_item.get("end_line"),
                content_start=directive_item.get("content_start"),
                content_end=directive_item.get("content_end"),
                header_lines=directive_item.get("header_lines", []) or [],
            )
            if directive_item
            else None
        )

        if directive_container:
            document_id = directive_container.get("document_id")
            generated_at = directive_container.get("generated_at")
        else:
            rc = directive_item.get("_registry_context", {}) or {}
            document_id = rc.get("document_id")
            generated_at = None

        registry_context = directive_item.get("_registry_context", {}) or {}
        metadata = directive_item.get("metadata", {}) or {}
        section = directive_item.get("section")
        references = _merge_reference_labels(
            directive_item.get("references"),
            extracted.get("references"),
        )

        core_statement_data = extracted.get("core_statement")
        core_statement = (
            CoreStatement(**core_statement_data) if isinstance(core_statement_data, dict) else None
        )

        hypotheses = [Hypothesis(**h) for h in extracted.get("hypotheses", []) or []]
        implications = [Implication(**i) for i in extracted.get("implications", []) or []]
        parameters = [Parameter(**p) for p in extracted.get("parameters", []) or []]
        failure_modes = [FailureMode(**f) for f in extracted.get("failure_modes", []) or []]

        alt_labels = []
        if label_ex and label_dir and label_ex != label_dir:
            alt_labels = [label_ex, label_dir]

        return cls(
            label=label,
            title=title,
            type=extracted.get("type", "axiom"),
            axiom_class=extracted.get("axiom_class"),
            nl_summary=extracted.get("nl_summary"),
            core_statement=core_statement,
            hypotheses=hypotheses,
            implications=implications,
            parameters=parameters,
            failure_modes=failure_modes,
            tags=extracted.get("tags", []),
            content_markdown=content_markdown,
            raw_directive=directive_item.get("raw_directive"),
            references=references,
            document_id=document_id,
            section=section,
            span=span,
            metadata=metadata,
            registry_context=registry_context,
            generated_at=generated_at,
            alt_labels=alt_labels,
        )


# ============================================================================
# SECTION 6: DEFINITION-SPECIFIC MODELS
# ============================================================================


class NamedProperty(BaseModel):
    """Named property in a definition."""

    name: str | None = None
    description: str | None = None


class Example(BaseModel):
    """Example illustrating a definition."""

    text: str | None = None
    latex: str | None = None


class UnifiedDefinition(BaseModel):
    """Unified definition model."""

    label: str
    type: str = Field(default="definition")
    title: str | None = None
    term: str | None = None
    object_type: str | None = None
    nl_definition: str | None = None
    content_markdown: str | None = None
    raw_directive: str | None = None
    formal_conditions: list[Condition] = Field(default_factory=list)
    properties: list[NamedProperty] = Field(default_factory=list)
    parameters: list[Parameter] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)
    related_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    document_id: str | None = None
    section: str | None = None
    span: Span | None = None
    references: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    registry_context: dict[str, Any] = Field(default_factory=dict)
    generated_at: str | None = None
    alt_labels: list[str] = Field(default_factory=list)

    @staticmethod
    def _strip_line_numbers(text: str | None) -> str | None:
        """Remove leading 'NNN: ' prefixes from directive content."""
        if not text:
            return text
        out_lines = []
        for line in text.splitlines():
            out_lines.append(re.sub(r"^\s*\d+:\s?", "", line))
        return "\n".join(out_lines).strip()

    @classmethod
    def from_instances(
        cls, directive: dict[str, Any], extracted: dict[str, Any]
    ) -> UnifiedDefinition:
        """Merge directive and extracted definition data."""
        # Implementation preserved from process_definitions.py
        directive_container = None
        directive_item = directive
        if "items" in directive and isinstance(directive["items"], list):
            directive_container = directive
            ex_label = extracted.get("label")
            matched = next(
                (it for it in directive["items"] if it.get("label") == ex_label),
                None,
            )
            directive_item = matched or (directive["items"][0] if directive["items"] else {})

        label_ex = extracted.get("label")
        label_dir = directive_item.get("label")
        label = label_ex or label_dir or ""
        title = extracted.get("title") or directive_item.get("title")

        raw_content = directive_item.get("content")
        content_markdown = cls._strip_line_numbers(raw_content) if raw_content else None

        span = (
            Span(
                start_line=directive_item.get("start_line"),
                end_line=directive_item.get("end_line"),
                content_start=directive_item.get("content_start"),
                content_end=directive_item.get("content_end"),
                header_lines=directive_item.get("header_lines", []) or [],
            )
            if directive_item
            else None
        )

        if directive_container:
            document_id = directive_container.get("document_id")
            generated_at = directive_container.get("generated_at")
        else:
            rc = directive_item.get("_registry_context", {}) or {}
            document_id = rc.get("document_id")
            generated_at = None

        registry_context = directive_item.get("_registry_context", {}) or {}
        metadata = directive_item.get("metadata", {}) or {}
        section = directive_item.get("section")
        references = _merge_reference_labels(
            directive_item.get("references"),
            extracted.get("references"),
        )

        formal_conditions = [Condition(**c) for c in extracted.get("formal_conditions", []) or []]
        properties = [NamedProperty(**p) for p in extracted.get("properties", []) or []]
        parameters = [Parameter(**p) for p in extracted.get("parameters", []) or []]
        examples = [Example(**e) for e in extracted.get("examples", []) or []]
        notes = [Note(**n) for n in extracted.get("notes", []) or []]

        alt_labels = []
        if label_ex and label_dir and label_ex != label_dir:
            alt_labels = [label_ex, label_dir]

        return cls(
            label=label,
            title=title,
            type=extracted.get("type", "definition"),
            term=extracted.get("term"),
            object_type=extracted.get("object_type"),
            nl_definition=extracted.get("nl_definition"),
            content_markdown=content_markdown,
            raw_directive=directive_item.get("raw_directive"),
            formal_conditions=formal_conditions,
            properties=properties,
            parameters=parameters,
            examples=examples,
            notes=notes,
            related_refs=extracted.get("related_refs", []),
            tags=extracted.get("tags", []),
            document_id=document_id,
            section=section,
            span=span,
            references=references,
            metadata=metadata,
            registry_context=registry_context,
            generated_at=generated_at,
            alt_labels=alt_labels,
        )


# ============================================================================
# SECTION 7: PROOF-SPECIFIC MODELS
# ============================================================================


class ProofAssumption(BaseModel):
    """Assumption used in a proof (different from Assumption for theorems).

    Note: Renamed from 'Assumption' to avoid collision with data_models.Assumption.
    """

    text: str
    latex: str | None = None


class KeyEquation(BaseModel):
    """Key equation in a proof."""

    label: str | None = None
    latex: str
    role: str | None = None


class MathTool(BaseModel):
    """Mathematical tool used in a proof."""

    toolName: str | None = None
    field: str | None = None
    description: str | None = None
    roleInProof: str | None = None
    levelOfAbstraction: str | None = None
    relatedTools: list[str] = Field(default_factory=list)


class CaseItem(BaseModel):
    """Case analysis item in a proof."""

    name: str | None = None
    condition: str | None = None
    summary: str | None = None


class Remark(BaseModel):
    """Remark or comment in a proof."""

    type: str | None = None
    text: str | None = None


class Gap(BaseModel):
    """Gap or missing piece in a proof."""

    description: str
    severity: str | None = None
    location_hint: str | None = None


class UnifiedProof(BaseModel):
    """Unified proof model."""

    label: str
    title: str | None = None
    type: str = Field(default="proof")
    proves: str | None = None
    proof_type: str | None = None
    proof_status: str | None = None
    content_markdown: str | None = None
    raw_directive: str | None = None
    strategy_summary: str | None = None
    conclusion: Conclusion | None = None
    assumptions: list[ProofAssumption] = Field(default_factory=list)
    steps: list[ProofStep] = Field(default_factory=list)
    key_equations: list[KeyEquation] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    math_tools: list[MathTool] = Field(default_factory=list)
    cases: list[CaseItem] = Field(default_factory=list)
    remarks: list[Remark] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    document_id: str | None = None
    section: str | None = None
    span: Span | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    registry_context: dict[str, Any] = Field(default_factory=dict)
    generated_at: str | None = None
    alt_labels: list[str] = Field(default_factory=list)

    @staticmethod
    def _strip_line_numbers(text: str | None) -> str | None:
        """Remove leading 'NNN: ' prefixes from directive content."""
        if not text:
            return text
        out_lines = []
        for line in text.splitlines():
            out_lines.append(re.sub(r"^\s*\d+:\s?", "", line))
        return "\n".join(out_lines).strip()

    @classmethod
    def from_instances(cls, directive: dict[str, Any], extracted: dict[str, Any]) -> UnifiedProof:
        """Merge directive and extracted proof data."""
        # Implementation preserved from process_proofs.py
        directive_container = None
        directive_item = directive
        if "items" in directive and isinstance(directive["items"], list):
            directive_container = directive
            ex_label = extracted.get("label")
            matched = next(
                (it for it in directive["items"] if it.get("label") == ex_label),
                None,
            )
            directive_item = matched or (directive["items"][0] if directive["items"] else {})

        label_ex = extracted.get("label")
        label_dir = directive_item.get("label")
        label = label_ex or label_dir or ""
        title = extracted.get("title") or directive_item.get("title")

        raw_content = directive_item.get("content")
        content_markdown = cls._strip_line_numbers(raw_content) if raw_content else None

        span = (
            Span(
                start_line=directive_item.get("start_line"),
                end_line=directive_item.get("end_line"),
                content_start=directive_item.get("content_start"),
                content_end=directive_item.get("content_end"),
                header_lines=directive_item.get("header_lines", []) or [],
            )
            if directive_item
            else None
        )

        if directive_container:
            document_id = directive_container.get("document_id")
            generated_at = directive_container.get("generated_at")
        else:
            rc = directive_item.get("_registry_context", {}) or {}
            document_id = rc.get("document_id")
            generated_at = None

        registry_context = directive_item.get("_registry_context", {}) or {}
        metadata = directive_item.get("metadata", {}) or {}
        section = directive_item.get("section")
        references_list = _merge_reference_labels(
            directive_item.get("references"),
            extracted.get("references"),
        )

        conclusion_data = extracted.get("conclusion")
        conclusion = Conclusion(**conclusion_data) if isinstance(conclusion_data, dict) else None

        assumptions = [ProofAssumption(**a) for a in extracted.get("assumptions", []) or []]
        steps = [ProofStep(**s) for s in extracted.get("steps", []) or []]
        key_equations = [KeyEquation(**e) for e in extracted.get("key_equations", []) or []]
        math_tools = [MathTool(**t) for t in extracted.get("math_tools", []) or []]
        cases = [CaseItem(**c) for c in extracted.get("cases", []) or []]
        remarks = [Remark(**r) for r in extracted.get("remarks", []) or []]
        gaps = [Gap(**g) for g in extracted.get("gaps", []) or []]

        alt_labels = []
        if label_ex and label_dir and label_ex != label_dir:
            alt_labels = [label_ex, label_dir]

        return cls(
            label=label,
            title=title,
            type=extracted.get("type", "proof"),
            proves=extracted.get("proves"),
            proof_type=extracted.get("proof_type"),
            proof_status=extracted.get("proof_status"),
            content_markdown=content_markdown,
            raw_directive=directive_item.get("raw_directive"),
            strategy_summary=extracted.get("strategy_summary"),
            conclusion=conclusion,
            assumptions=assumptions,
            steps=steps,
            key_equations=key_equations,
            references=references_list,
            math_tools=math_tools,
            cases=cases,
            remarks=remarks,
            gaps=gaps,
            tags=extracted.get("tags", []),
            document_id=document_id,
            section=section,
            span=span,
            metadata=metadata,
            registry_context=registry_context,
            generated_at=generated_at,
            alt_labels=alt_labels,
        )


# ============================================================================
# SECTION 8: REMARK-SPECIFIC MODELS
# ============================================================================


class KeyPoint(BaseModel):
    """Key point in a remark."""

    text: str
    latex: str | None = None
    importance: str | None = None


class QuantitativeNote(BaseModel):
    """Quantitative note in a remark."""

    text: str
    latex: str | None = None


class Recommendation(BaseModel):
    """Recommendation in a remark."""

    text: str
    severity: str | None = None


class RawMetadata(BaseModel):
    """Raw metadata for remarks."""

    label: str | None = None
    class_: str | None = Field(default=None, alias="class")


class RegistryContext(BaseModel):
    """Registry context information."""

    stage: str | None = None
    document_id: str | None = None
    chapter_index: int | None = None
    chapter_file: str | None = None
    section_id: str | None = None


class UnifiedRemark(BaseModel):
    """Unified remark model."""

    label: str
    title: str | None = None
    remark_type: str | None = None
    nl_summary: str | None = None
    key_points: list[KeyPoint] = Field(default_factory=list)
    quantitative_notes: list[QuantitativeNote] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    references: list[Any] = Field(default_factory=list)
    content: str | None = None
    section: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    header_lines: list[int] = Field(default_factory=list)
    content_start: int | None = None
    content_end: int | None = None
    raw_directive: str | None = None
    directive_type: str | None = None
    metadata: RawMetadata | None = None
    registry_context: RegistryContext | None = Field(default=None, alias="_registry_context")

    class Config:
        # Pydantic v1/v2 compatibility
        populate_by_name = True

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        """Remove duplicates while preserving order."""
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    @classmethod
    def from_instances(
        cls, raw_item: dict[str, Any], extracted_item: dict[str, Any]
    ) -> UnifiedRemark:
        """Merge raw and extracted remark data."""
        # Implementation preserved from process_remarks.py
        label = extracted_item.get("label") or raw_item.get("label") or ""
        title = extracted_item.get("title") or raw_item.get("title")

        key_points = [KeyPoint(**k) for k in extracted_item.get("key_points", []) or []]
        quantitative_notes = [
            QuantitativeNote(**q) for q in extracted_item.get("quantitative_notes", []) or []
        ]
        recommendations = [
            Recommendation(**r) for r in extracted_item.get("recommendations", []) or []
        ]

        dependencies = cls._dedupe(
            (extracted_item.get("dependencies") or []) + (raw_item.get("dependencies") or [])
        )
        tags = cls._dedupe((extracted_item.get("tags") or []) + (raw_item.get("tags") or []))
        references = cls._dedupe(
            (extracted_item.get("references") or []) + (raw_item.get("references") or [])
        )

        metadata_dict = raw_item.get("metadata")
        metadata = RawMetadata(**metadata_dict) if metadata_dict else None

        registry_context_dict = raw_item.get("_registry_context")
        registry_context = (
            RegistryContext(**registry_context_dict) if registry_context_dict else None
        )

        return cls(
            label=label,
            title=title,
            remark_type=extracted_item.get("remark_type"),
            nl_summary=extracted_item.get("nl_summary"),
            key_points=key_points,
            quantitative_notes=quantitative_notes,
            recommendations=recommendations,
            dependencies=dependencies,
            tags=tags,
            references=references,
            content=raw_item.get("content"),
            section=raw_item.get("section"),
            start_line=raw_item.get("start_line"),
            end_line=raw_item.get("end_line"),
            header_lines=raw_item.get("header_lines", []),
            content_start=raw_item.get("content_start"),
            content_end=raw_item.get("content_end"),
            raw_directive=raw_item.get("raw_directive"),
            directive_type=raw_item.get("directive_type"),
            metadata=metadata,
            _registry_context=registry_context,
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "Algorithm",
    "AlgorithmParameter",
    "AlgorithmSignature",
    "AlgorithmStep",
    "Assumption",
    # Section 4: Assumption-specific
    "BulletItem",
    "CaseItem",
    "Conclusion",
    "Condition",
    # Section 5: Axiom-specific
    "CoreStatement",
    # Section 3: Algorithm-specific
    "DocumentMetadata",
    # Section 1: Shared nested models
    "Equation",
    "Example",
    "ExtractedAlgorithm",
    "FailureMode",
    "Gap",
    "GuardCondition",
    "Hypothesis",
    "Implication",
    "KeyEquation",
    # Section 8: Remark-specific
    "KeyPoint",
    "MathTool",
    # Section 6: Definition-specific
    "NamedProperty",
    "Note",
    "Parameter",
    "Proof",
    # Section 7: Proof-specific
    "ProofAssumption",
    "ProofStep",
    "QuantitativeNote",
    "RawAlgorithm",
    "RawMetadata",
    "Recommendation",
    "RegistryContext",
    "Remark",
    "Span",
    "UnifiedAssumption",
    "UnifiedAxiom",
    "UnifiedCorollary",
    "UnifiedDefinition",
    "UnifiedLemma",
    # Section 2: Theorem-like entities
    "UnifiedMathematicalEntity",
    "UnifiedProof",
    "UnifiedProposition",
    "UnifiedRemark",
    "UnifiedTheorem",
    "Variable",
]
