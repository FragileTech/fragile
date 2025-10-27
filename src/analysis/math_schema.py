#!/usr/bin/env python3
"""
Pydantic models for the Fragile Gas Mathematical Documentation Schema.

This module provides Python type-safe equivalents to math_schema.json,
enabling validation, serialization, and programmatic document creation.

Generated from: math_schema.json v1.0.0
"""

from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, constr, field_validator


# ==================== Basic Types ====================

Label = constr(pattern=r"^[a-z][a-z0-9-]*[a-z0-9]$")
"""Unique identifier for cross-referencing (kebab-case).

Examples: 'def-walker', 'thm-main-convergence', 'lem-keystone'
"""

MathExpression = str
"""LaTeX mathematical expression.

Use $ for inline and $$ for display math.
Must be valid LaTeX that compiles with standard AMS packages.
"""


# ==================== Enums and Literals ====================

DirectiveType = Literal[
    "definition",
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "axiom",
    "assumption",
    "proof",
    "remark",
    "observation",
    "conjecture",
    "property",
    "algorithm",
    "example",
    "equation",
]

ReferenceRole = Literal[
    "prerequisite",
    "uses",
    "proves",
    "extends",
    "contradicts",
    "generalizes",
    "specializes",
    "equivalent",
    "motivation",
    "application",
    "counterexample",
]

AssumptionType = Literal[
    "regularity",
    "boundedness",
    "continuity",
    "differentiability",
    "integrability",
    "existence",
    "uniqueness",
    "compatibility",
    "non-degeneracy",
    "structural",
    "technical",
]

VerificationType = Literal[
    "symbolic",
    "numerical",
    "simulation",
    "visualization",
    "unit-test",
    "integration-test",
]

VerificationStatus = Literal["verified", "partial", "planned", "failed", "not-applicable"]

ProofType = Literal[
    "direct",
    "contradiction",
    "contrapositive",
    "induction",
    "construction",
    "computation",
    "case-analysis",
    "variational",
    "probabilistic",
    "combinatorial",
]

ProofDifficulty = Literal["routine", "standard", "technical", "intricate", "deep"]

BoundType = Literal[
    "upper",
    "lower",
    "two-sided",
    "asymptotic",
    "exponential",
    "polynomial",
    "logarithmic",
]

BoundTightness = Literal["tight", "optimal", "improvable", "crude"]

Importance = Literal["foundational", "main-result", "technical-lemma", "auxiliary"]

RemarkType = Literal[
    "clarification",
    "warning",
    "connection",
    "historical-note",
    "interpretation",
    "limitation",
    "extension",
    "special-case",
]

PropertyType = Literal[
    "algebraic",
    "topological",
    "geometric",
    "analytic",
    "probabilistic",
    "combinatorial",
    "structural",
]

AxiomCategory = Literal[
    "viability",
    "environmental",
    "measurement-quality",
    "algorithmic-dynamics",
    "regularity",
    "geometric",
    "structural",
]

ParameterType = Literal["scalar", "bound", "rate", "probability", "dimension", "index"]

ParameterSensitivity = Literal["low", "medium", "high", "critical"]

ConfidenceLevel = Literal["low", "medium", "high", "very-high"]

EvidenceType = Literal[
    "numerical",
    "heuristic",
    "special-cases",
    "analogous-results",
    "physical-intuition",
    "computational",
    "analytical",
]

RigorLevel = Literal["sketch", "informal", "rigorous", "publication-ready", "published"]

PeerReviewStatus = Literal["pending", "completed", "revised", "approved"]

# Review-related types
Severity = Literal["critical", "major", "minor", "suggestion"]

IssueDistinction = Literal["missing-proof", "incorrect-claim", "ambiguous", "unclear"]

IssueStatus = Literal["open", "addressed", "verified", "wontfix"]

EstimatedDifficulty = Literal[
    "straightforward", "moderate", "requires-new-proof", "fundamental-rework"
]

Verdict = Literal["ready", "minor-revisions", "major-revisions", "reject"]

Reviewer = Literal["gemini-2.5-pro", "codex", "claude-synthesis", "human-expert", "other"]

DevelopmentStage = Literal["sketch", "partial", "complete", "verified", "published"]

DetailLevel = Literal["outline", "sketch", "detailed", "complete", "exhaustive"]

ProgrammingLanguage = Literal["python", "pytorch", "julia", "c++", "other"]


# ==================== Core Mathematical Structures ====================


class CrossReference(BaseModel):
    """Reference to another mathematical object."""

    label: Label = Field(..., description="Label of the referenced item")
    type: DirectiveType = Field(..., description="Type of the referenced mathematical object")
    role: Optional[ReferenceRole] = Field(
        None, description="How this reference is used in the current context"
    )
    description: Optional[str] = Field(
        None,
        description="Brief explanation of how this reference is used (e.g., 'provides the bound on variance')",
    )
    location: Optional[str] = Field(
        None,
        description="Where in the current item this reference is used (e.g., 'Step 2.3', 'Part 1', 'Equation (5)')",
    )


class PropertyConstant(BaseModel):
    """Constant appearing in a mathematical property."""

    symbol: str = Field(..., description="Mathematical symbol (e.g., 'κ_x', 'C_bound')")
    value: str = Field(..., description="Explicit value or bound expression")
    dependencies: Optional[List[str]] = Field(
        None, description="Parameters this constant depends on"
    )


class MathematicalProperty(BaseModel):
    """A mathematical property with optional quantitative information."""

    name: str = Field(
        ...,
        description="Name of the property (e.g., 'N-uniform stability', 'Exponential convergence', 'Lipschitz continuity')",
    )
    statement: MathExpression = Field(..., description="Mathematical statement of the property")
    quantitative: Optional[bool] = Field(
        None, description="Whether this property includes explicit constants/bounds"
    )
    constants: Optional[Dict[str, PropertyConstant]] = Field(
        None, description="Explicit constants appearing in this property"
    )
    scope: Optional[str] = Field(
        None,
        description="Scope or regime where this property holds (e.g., 'for large N', 'in the high-friction limit')",
    )


class MathematicalAssumption(BaseModel):
    """An assumption or hypothesis in a theorem/lemma."""

    statement: MathExpression = Field(..., description="Mathematical statement of the assumption")
    type: AssumptionType = Field(..., description="Category of the assumption")
    label: Optional[str] = Field(
        None, description="Optional label for referencing this specific assumption"
    )
    justification: Optional[str] = Field(
        None, description="Why this assumption is necessary or reasonable"
    )
    references: Optional[List[CrossReference]] = Field(
        None, description="References establishing or validating this assumption"
    )


class VerificationResults(BaseModel):
    """Results from computational verification."""

    summary: Optional[str] = Field(None, description="Brief summary of verification results")
    passed: Optional[bool] = Field(None, description="Whether verification passed")
    metrics: Optional[Dict[str, Any]] = Field(
        None, description="Quantitative metrics (e.g., error bounds, convergence rates)"
    )
    plots: Optional[List[str]] = Field(None, description="Paths to generated plots/figures")


class ComputationalVerification(BaseModel):
    """Computational verification of a mathematical claim."""

    type: VerificationType = Field(..., description="Type of computational verification")
    status: VerificationStatus = Field(..., description="Verification status")
    script_path: Optional[str] = Field(
        None, description="Path to Python/SymPy verification script relative to docs root"
    )
    notebook_path: Optional[str] = Field(
        None, description="Path to Jupyter notebook with interactive verification"
    )
    description: Optional[str] = Field(
        None, description="What aspect is being verified computationally"
    )
    results: Optional[VerificationResults] = Field(
        None, description="Results from running the verification"
    )


# ==================== Review and Development Tracking ====================


class ReviewIssue(BaseModel):
    """A specific issue identified during peer review."""

    severity: Severity = Field(
        ...,
        description="Issue severity level: CRITICAL (invalidates theorem), MAJOR (significant gap), MINOR (subtle error), SUGGESTION (improvement)",
    )
    title: str = Field(..., description="Brief descriptive title of the issue")
    location: Dict[str, Optional[str]] = Field(
        ...,
        description="Location information (section, line_range, equation_number, proof_step)",
    )
    problem: str = Field(..., description="Clear description of what is wrong or missing")
    mechanism: Optional[str] = Field(
        None, description="WHY this fails - precise mechanism of the error"
    )
    evidence: Optional[str] = Field(
        None, description="Quoted passage, counterexample, or calculation demonstrating the issue"
    )
    impact: Optional[str] = Field(
        None, description="What downstream results are affected by this issue"
    )
    suggested_fix: Optional[str] = Field(
        None, description="Concrete suggestion for how to fix this issue"
    )
    distinction: Optional[IssueDistinction] = Field(
        None, description="Classification of the issue type"
    )
    status: Optional[IssueStatus] = Field(default="open", description="Current status of the issue")
    estimated_difficulty: Optional[EstimatedDifficulty] = Field(
        None, description="Estimated difficulty to resolve this issue"
    )
    affected_labels: Optional[List[str]] = Field(
        None, description="Labels of other directives affected by this issue"
    )


class ReviewScore(BaseModel):
    """Review score from a single reviewer."""

    reviewer: Reviewer = Field(..., description="Reviewer identifier")
    review_date: date = Field(..., description="Date of review")
    rigor: int = Field(..., ge=1, le=10, description="Mathematical rigor score (1-10)")
    soundness: int = Field(..., ge=1, le=10, description="Logical soundness score (1-10)")
    consistency: int = Field(
        ..., ge=1, le=10, description="Framework consistency score (1-10)"
    )
    verdict: Verdict = Field(..., description="Overall publication readiness verdict")
    issues_identified: Optional[List[ReviewIssue]] = Field(
        None, description="List of issues identified during review"
    )
    strengths: Optional[List[str]] = Field(
        None, description="Positive aspects highlighted by reviewer"
    )
    recommended_actions: Optional[List[str]] = Field(
        None, description="Prioritized list of recommended actions"
    )
    comments: Optional[str] = Field(None, description="Additional reviewer comments")


class UniqueFindings(BaseModel):
    """Issues found by only one reviewer."""

    gemini_only: Optional[List[ReviewIssue]] = Field(
        None, description="Issues identified only by Gemini"
    )
    codex_only: Optional[List[ReviewIssue]] = Field(
        None, description="Issues identified only by Codex"
    )


class AggregateScore(BaseModel):
    """Aggregate scores from multiple reviewers."""

    rigor: float = Field(..., ge=1.0, le=10.0, description="Average rigor score")
    soundness: float = Field(..., ge=1.0, le=10.0, description="Average soundness score")
    consistency: float = Field(..., ge=1.0, le=10.0, description="Average consistency score")


class DualReviewAnalysis(BaseModel):
    """Dual review analysis combining Gemini and Codex reviews."""

    gemini_review: Optional[ReviewScore] = Field(None, description="Gemini 2.5 Pro review")
    codex_review: Optional[ReviewScore] = Field(None, description="Codex review")
    consensus_issues: Optional[List[ReviewIssue]] = Field(
        None, description="Issues identified by both reviewers (high confidence)"
    )
    discrepancies: Optional[List[str]] = Field(
        None, description="Contradictory findings requiring manual verification"
    )
    unique_findings: Optional[UniqueFindings] = Field(
        None, description="Issues found by only one reviewer"
    )
    aggregate_score: Optional[AggregateScore] = Field(
        None, description="Average scores across both reviewers"
    )
    final_verdict: Optional[Verdict] = Field(
        None, description="Final publication readiness verdict after synthesis"
    )
    blocking_issues: Optional[List[ReviewIssue]] = Field(
        None, description="CRITICAL issues that block publication"
    )
    synthesis_notes: Optional[str] = Field(
        None, description="Claude's synthesis and recommendations from dual review"
    )


class VerificationFlags(BaseModel):
    """Verification status flags."""

    logic_verified: Optional[bool] = Field(None, description="Logical correctness verified")
    computation_verified: Optional[bool] = Field(
        None, description="Computational aspects verified"
    )
    framework_consistent: Optional[bool] = Field(
        None, description="Consistent with framework axioms and definitions"
    )


class QualityMetrics(BaseModel):
    """Quality assessment metrics."""

    rigor_level: Optional[int] = Field(
        None, ge=1, le=10, description="Self-assessed rigor level (1-10)"
    )
    detail_level: Optional[DetailLevel] = Field(None, description="Level of detail provided")


class DevelopmentStatus(BaseModel):
    """Development maturity and completeness tracking."""

    stage: DevelopmentStage = Field(
        ..., description="Current stage of development (sketch → verified → published)"
    )
    completeness: int = Field(
        ..., ge=0, le=100, description="Completeness percentage (0-100)"
    )
    verification_status: Optional[VerificationFlags] = Field(
        None, description="What has been verified"
    )
    quality_metrics: Optional[QualityMetrics] = Field(
        None, description="Quality assessment metrics"
    )
    last_updated: Optional[date] = Field(None, description="When this status was last updated")
    notes: Optional[str] = Field(None, description="Development notes or TODO items")


class SourceSketch(BaseModel):
    """Information about the source sketch."""

    file_path: str = Field(..., description="Path to the sketch file")
    label: Optional[str] = Field(None, description="Label of the sketch")
    date_created: Optional[date] = Field(None, description="When the sketch was created")
    agent: Optional[str] = Field(None, description="Agent that created the sketch (e.g., 'gemini')")


class ExpansionHistoryEntry(BaseModel):
    """Single entry in expansion history."""

    expansion_date: date = Field(..., description="Date of this expansion step")
    stage: DevelopmentStage = Field(..., description="Stage reached at this step")
    agent: Optional[str] = Field(
        None, description="Agent performing expansion (e.g., 'claude', 'human')"
    )
    description: str = Field(..., description="Description of changes made")


class SketchCoverage(BaseModel):
    """Coverage of sketch items in the full proof."""

    coverage_percentage: int = Field(
        ..., ge=0, le=100, description="Percentage of sketch items covered"
    )
    uncovered_items: Optional[List[str]] = Field(
        None, description="List of sketch items not yet fully addressed"
    )


class StrategyOption(BaseModel):
    """Alternative proof strategy."""

    name: str = Field(..., description="Name of the strategy")
    source: Optional[str] = Field(
        None, description="Where this strategy came from (sketch/reviewer/alternative)"
    )
    chosen: bool = Field(..., description="Whether this strategy was chosen")
    rationale: Optional[str] = Field(None, description="Why this was chosen/rejected")


class StrategyComparison(BaseModel):
    """Comparison of different proof strategies."""

    strategies: List[StrategyOption] = Field(
        ..., description="Different strategies considered"
    )
    rationale: Optional[str] = Field(
        None, description="Overall rationale for strategy selection"
    )


class SketchProofLinkage(BaseModel):
    """Link between a proof sketch and its full expansion."""

    source_sketch: Optional[SourceSketch] = Field(
        None, description="Information about the original sketch"
    )
    expansion_history: Optional[List[ExpansionHistoryEntry]] = Field(
        None, description="History of expansion from sketch to full proof"
    )
    sketch_coverage: Optional[SketchCoverage] = Field(
        None, description="How much of the sketch is covered"
    )
    strategy_comparison: Optional[StrategyComparison] = Field(
        None, description="Comparison of sketch strategy vs. final strategy"
    )
    divergences: Optional[List[str]] = Field(
        None, description="Places where the full proof diverged from the sketch"
    )


# ==================== Proof Step Structure ====================


class ProofStep(BaseModel):
    """A single step in a proof (can have recursive substeps)."""

    id: str = Field(..., description="Step identifier (e.g., 'Step 1', 'Step 2.1')")
    title: Optional[str] = Field(None, description="Optional step title")
    content: str = Field(..., description="Mathematical content of this step")
    type: Optional[str] = Field(None, description="Type of reasoning (e.g., 'direct', 'case-1')")
    techniques: Optional[List[str]] = Field(
        None,
        description="Mathematical techniques used (e.g., 'contradiction', 'induction', 'Cauchy-Schwarz')",
    )
    justification: Optional[Union[str, List[Union[str, CrossReference]]]] = Field(
        None, description="Why this step is valid (references or text)"
    )
    intermediate_result: Optional[MathExpression] = Field(
        None, description="Key intermediate result from this step"
    )
    substeps: Optional[List["ProofStep"]] = Field(
        None, description="Substeps for hierarchical proof structure"
    )


# Enable forward reference resolution
ProofStep.model_rebuild()


# ==================== Source Location ====================


class SourceLocation(BaseModel):
    """Location in source documentation."""

    document: Optional[str] = Field(
        None, description="Source document (e.g., '01_fragile_gas_framework.md')"
    )
    section: Optional[str] = Field(None, description="Section reference (e.g., '§2.3.5')")
    equation: Optional[str] = Field(None, description="Equation reference (e.g., 'Eq. (17)')")
    page: Optional[int] = Field(None, description="Page number (if from PDF/paper)")


class BaseDirective(BaseModel):
    """Base class for all mathematical directives."""

    type: DirectiveType = Field(..., description="Type of this directive")
    label: Label = Field(..., description="Unique label for cross-referencing")
    title: str = Field(..., description="Human-readable title")
    statement: str = Field(..., description="Main statement (LaTeX-formatted)")
    tags: Optional[List[str]] = Field(
        None, description="Tags for categorization (e.g., 'cloning', 'convergence', 'fundamental')"
    )
    source: Optional[SourceLocation] = Field(
        None, description="Where this directive appears in source documentation"
    )
    related_concepts: Optional[List[CrossReference]] = Field(
        None, description="Related mathematical concepts"
    )


# ==================== Definition ====================


class DefinedObject(BaseModel):
    """An object being defined."""

    name: str = Field(
        ..., description="Name of the object being defined (e.g., 'Walker', 'Swarm State Space')"
    )
    symbol: Optional[str] = Field(
        None,
        description="Primary mathematical symbol (e.g., 'w', '\\mathcal{S}', '\\Sigma_N')",
    )
    mathematical_definition: MathExpression = Field(
        ..., description="Formal mathematical definition"
    )
    type: Optional[
        Literal[
            "set",
            "function",
            "operator",
            "measure",
            "metric",
            "space",
            "relation",
            "constant",
            "variable",
            "parameter",
            "structure",
        ]
    ] = Field(None, description="Mathematical type of the defined object")
    properties: Optional[List[Union[str, MathematicalProperty]]] = Field(
        None, description="Key properties that follow immediately from the definition"
    )


class DefinitionExample(BaseModel):
    """Example illustrating a definition."""

    description: str
    instance: MathExpression
    verification: Optional[str] = Field(None, description="Why this is a valid example")


class DefinitionCounterexample(BaseModel):
    """Counterexample for a definition."""

    description: str
    instance: MathExpression
    reason: Optional[str] = Field(
        None, description="Why this fails to satisfy the definition"
    )


class Definition(BaseDirective):
    """A mathematical definition."""

    type: Literal["definition"] = "definition"
    defined_objects: List[DefinedObject] = Field(
        ..., description="Objects being defined (can define multiple related objects)"
    )
    motivation: Optional[str] = Field(
        None, description="Why this definition is introduced (mathematical or intuitive motivation)"
    )
    examples: Optional[List[DefinitionExample]] = Field(
        None, description="Concrete examples illustrating the definition"
    )
    counterexamples: Optional[List[DefinitionCounterexample]] = Field(
        None, description="Examples that fail to satisfy the definition (clarifies boundaries)"
    )
    peer_review: Optional[DualReviewAnalysis] = Field(
        None, description="Dual review analysis (Gemini + Codex) tracking publication readiness"
    )
    development_status: Optional[DevelopmentStatus] = Field(
        None, description="Development maturity and completeness tracking"
    )
    sketch_linkage: Optional[SketchProofLinkage] = Field(
        None, description="Link to proof sketch and expansion history (if applicable)"
    )


# ==================== Axiom ====================


class AxiomaticParameter(BaseModel):
    """Parameter introduced by an axiom."""

    symbol: str = Field(
        ..., description="Parameter symbol (e.g., 'κ_revival', 'L_R', 'p_worst-case')"
    )
    description: str = Field(..., description="What this parameter quantifies")
    type: ParameterType = Field(..., description="Type of parameter")
    conditions: Optional[Union[str, List[MathExpression]]] = Field(
        None,
        description="Required conditions on this parameter (e.g., 'κ_revival > 1', 'L_R < ∞')",
    )
    typical_values: Optional[str] = Field(
        None, description="Typical or recommended parameter values"
    )
    sensitivity: Optional[ParameterSensitivity] = Field(
        None, description="How sensitive the framework is to this parameter"
    )


class FailureMode(BaseModel):
    """What happens when an axiom is violated."""

    condition: Optional[MathExpression] = Field(
        None, description="Parameter configuration that causes failure"
    )
    consequence: str = Field(..., description="What goes wrong when axiom is violated")
    diagnostic: Optional[str] = Field(
        None, description="How to detect this failure mode in practice"
    )


class Axiom(BaseDirective):
    """A foundational axiom of the framework."""

    type: Literal["axiom"] = "axiom"
    axiomatic_parameters: Optional[List[AxiomaticParameter]] = Field(
        None, description="Quantifiable parameters introduced by this axiom"
    )
    category: Optional[AxiomCategory] = Field(
        None, description="Axiom category in the framework hierarchy"
    )
    failure_modes: Optional[List[FailureMode]] = Field(
        None, description="What happens when the axiom is violated (for debugging)"
    )
    peer_review: Optional[DualReviewAnalysis] = Field(
        None, description="Dual review analysis (Gemini + Codex) tracking publication readiness"
    )
    development_status: Optional[DevelopmentStatus] = Field(
        None, description="Development maturity and completeness tracking"
    )
    sketch_linkage: Optional[SketchProofLinkage] = Field(
        None, description="Link to proof sketch and expansion history (if applicable)"
    )


# ==================== Theorem-like Directives ====================


class QuantitativeBound(BaseModel):
    """A quantitative bound in a theorem conclusion."""

    bound: MathExpression
    type: BoundType
    tightness: Optional[BoundTightness] = Field(None, description="How tight the bound is")


class TheoremConclusion(BaseModel):
    """Conclusion of a theorem."""

    statement: MathExpression = Field(..., description="Main conclusion of the theorem")
    properties_established: Optional[List[Union[str, MathematicalProperty]]] = Field(
        None, description="Key properties established by this theorem (for downstream use)"
    )
    quantitative_bounds: Optional[Dict[str, Union[str, QuantitativeBound]]] = Field(
        None, description="Explicit quantitative bounds established"
    )


class ProofReference(BaseModel):
    """Reference to where a proof can be found."""

    label: Optional[Label] = Field(None, description="Label of the proof directive")
    inline: Optional[bool] = Field(None, description="Whether proof is inline")
    sketch: Optional[bool] = Field(None, description="Whether only a sketch is provided")
    reference: Optional[str] = Field(
        None, description="External reference (e.g., 'Rudin (1987), Theorem 3.14')"
    )


class Theorem(BaseDirective):
    """A mathematical theorem."""

    type: Literal["theorem"] = "theorem"
    hypotheses: List[MathematicalAssumption] = Field(
        ..., description="All hypotheses/assumptions required for the theorem to hold"
    )
    conclusion: TheoremConclusion = Field(..., description="Theorem conclusion")
    proof_reference: Optional[ProofReference] = Field(
        None, description="Where to find the proof"
    )
    importance: Optional[Importance] = Field(
        None, description="Importance level in the framework"
    )
    peer_review: Optional[DualReviewAnalysis] = Field(
        None, description="Dual review analysis (Gemini + Codex) tracking publication readiness"
    )
    development_status: Optional[DevelopmentStatus] = Field(
        None, description="Development maturity and completeness tracking"
    )
    sketch_linkage: Optional[SketchProofLinkage] = Field(
        None, description="Link to proof sketch and expansion history (if applicable)"
    )


class Lemma(BaseDirective):
    """A supporting lemma."""

    type: Literal["lemma"] = "lemma"
    hypotheses: List[MathematicalAssumption] = Field(..., description="Lemma hypotheses")
    conclusion: TheoremConclusion = Field(..., description="Lemma conclusion")
    proof_reference: Optional[ProofReference] = Field(
        None, description="Where to find the proof"
    )
    supports: Optional[List[CrossReference]] = Field(
        None, description="Main results this lemma supports"
    )
    importance: Optional[Importance] = Field(
        None, description="Importance level in the framework"
    )
    peer_review: Optional[DualReviewAnalysis] = Field(
        None, description="Dual review analysis (Gemini + Codex) tracking publication readiness"
    )
    development_status: Optional[DevelopmentStatus] = Field(
        None, description="Development maturity and completeness tracking"
    )
    sketch_linkage: Optional[SketchProofLinkage] = Field(
        None, description="Link to proof sketch and expansion history (if applicable)"
    )


class Proposition(BaseDirective):
    """A mathematical proposition."""

    type: Literal["proposition"] = "proposition"
    hypotheses: List[MathematicalAssumption] = Field(..., description="Proposition hypotheses")
    conclusion: TheoremConclusion = Field(..., description="Proposition conclusion")
    proof_reference: Optional[ProofReference] = Field(
        None, description="Where to find the proof"
    )
    importance: Optional[Importance] = Field(
        None, description="Importance level in the framework"
    )
    peer_review: Optional[DualReviewAnalysis] = Field(
        None, description="Dual review analysis (Gemini + Codex) tracking publication readiness"
    )
    development_status: Optional[DevelopmentStatus] = Field(
        None, description="Development maturity and completeness tracking"
    )
    sketch_linkage: Optional[SketchProofLinkage] = Field(
        None, description="Link to proof sketch and expansion history (if applicable)"
    )


class CorollaryConclusion(BaseModel):
    """Conclusion of a corollary."""

    statement: MathExpression
    properties_established: Optional[List[MathematicalProperty]] = None


class ImmediateProof(BaseModel):
    """Proof that follows immediately."""

    immediate: Literal[True] = True
    justification: Optional[str] = Field(
        None, description="Brief explanation of why this follows immediately"
    )


class InlineProof(BaseModel):
    """Inline proof marker."""

    inline: Literal[True] = True


class Corollary(BaseDirective):
    """A corollary following from theorems/lemmas."""

    type: Literal["corollary"] = "corollary"
    follows_from: List[CrossReference] = Field(
        ..., description="Theorems/lemmas this corollary immediately follows from"
    )
    conclusion: CorollaryConclusion = Field(..., description="Corollary conclusion")
    proof_reference: Optional[Union[ImmediateProof, InlineProof]] = None
    peer_review: Optional[DualReviewAnalysis] = Field(
        None, description="Dual review analysis (Gemini + Codex) tracking publication readiness"
    )
    development_status: Optional[DevelopmentStatus] = Field(
        None, description="Development maturity and completeness tracking"
    )
    sketch_linkage: Optional[SketchProofLinkage] = Field(
        None, description="Link to proof sketch and expansion history (if applicable)"
    )


# ==================== Proof ====================


class AlternativeApproach(BaseModel):
    """Alternative proof approach."""

    description: str
    advantages: Optional[str] = None
    disadvantages: Optional[str] = None


class Proof(BaseDirective):
    """A mathematical proof."""

    type: Literal["proof"] = "proof"
    proves: CrossReference = Field(
        ..., description="What theorem/lemma/proposition this proves"
    )
    proof_type: Optional[ProofType] = Field(None, description="Overall proof strategy type")
    strategy: str = Field(
        ...,
        description="High-level overview of the proof approach (1-3 paragraphs explaining the main idea)",
    )
    prerequisites: Optional[List[CrossReference]] = Field(
        None, description="All results used in this proof (explicitly listed)"
    )
    steps: List[ProofStep] = Field(
        ..., description="Detailed proof steps in hierarchical structure", min_length=1
    )
    key_insights: Optional[List[str]] = Field(
        None,
        description="Main mathematical insights that make the proof work (pedagogical)",
    )
    alternative_approaches: Optional[List[AlternativeApproach]] = Field(
        None, description="Other possible proof strategies and why this one was chosen"
    )
    computational_verification: Optional[List[ComputationalVerification]] = Field(
        None, description="Computational verifications for different parts of the proof"
    )
    difficulty: Optional[ProofDifficulty] = Field(None, description="Mathematical difficulty level")
    rigor_level: Optional[int] = Field(
        None, ge=1, le=10, description="Self-assessed rigor level (1=sketch, 10=publication-ready)"
    )
    peer_review: Optional[DualReviewAnalysis] = Field(
        None, description="Dual review analysis (Gemini + Codex) tracking publication readiness"
    )
    development_status: Optional[DevelopmentStatus] = Field(
        None, description="Development maturity and completeness tracking"
    )
    sketch_linkage: Optional[SketchProofLinkage] = Field(
        None, description="Link to proof sketch and expansion history (if applicable)"
    )


# ==================== Algorithm ====================


class AlgorithmInput(BaseModel):
    """Input to an algorithm."""

    name: str
    type: str = Field(
        ..., description="Mathematical type (e.g., 'swarm state S ∈ Σ_N', 'parameter γ > 0')"
    )
    description: Optional[str] = None
    constraints: Optional[List[MathExpression]] = None


class AlgorithmOutput(BaseModel):
    """Output from an algorithm."""

    name: str
    type: str
    description: Optional[str] = None
    guarantees: Optional[List[MathExpression]] = Field(
        None, description="Mathematical properties guaranteed for output"
    )


class AlgorithmStep(BaseModel):
    """Single step in an algorithm."""

    step_number: Optional[int] = None
    description: str
    pseudocode: Optional[str] = None
    mathematical_operation: Optional[MathExpression] = None
    complexity: Optional[str] = Field(None, description="Time/space complexity for this step")


class AlgorithmComplexity(BaseModel):
    """Computational complexity analysis."""

    time: Optional[str] = Field(
        None, description="Overall time complexity (e.g., 'O(N log N)', 'O(N²)')"
    )
    space: Optional[str] = Field(None, description="Space complexity")
    worst_case: Optional[str] = None
    average_case: Optional[str] = None


class AlgorithmImplementation(BaseModel):
    """Implementation details."""

    path: Optional[str] = Field(None, description="Path to implementation code")
    language: Optional[ProgrammingLanguage] = None
    entry_point: Optional[str] = Field(
        None, description="Function/class name implementing this algorithm"
    )
    tests: Optional[str] = Field(None, description="Path to test suite")


class Algorithm(BaseDirective):
    """An algorithm specification."""

    type: Literal["algorithm"] = "algorithm"
    inputs: List[AlgorithmInput] = Field(..., description="Algorithm inputs")
    outputs: List[AlgorithmOutput] = Field(..., description="Algorithm outputs")
    steps: List[AlgorithmStep] = Field(..., description="Algorithm steps")
    complexity: Optional[AlgorithmComplexity] = None
    correctness_proof: Optional[CrossReference] = Field(
        None, description="Reference to proof of correctness"
    )
    implementation: Optional[AlgorithmImplementation] = None
    peer_review: Optional[DualReviewAnalysis] = Field(
        None, description="Dual review analysis (Gemini + Codex) tracking publication readiness"
    )
    development_status: Optional[DevelopmentStatus] = Field(
        None, description="Development maturity and completeness tracking"
    )
    sketch_linkage: Optional[SketchProofLinkage] = Field(
        None, description="Link to proof sketch and expansion history (if applicable)"
    )


# ==================== Other Directive Types ====================


class Remark(BaseDirective):
    """A mathematical remark or note."""

    type: Literal["remark"] = "remark"
    remark_type: Optional[RemarkType] = None
    relates_to: Optional[List[CrossReference]] = Field(
        None, description="What results this remark comments on"
    )
    peer_review: Optional[DualReviewAnalysis] = Field(
        None, description="Dual review analysis (Gemini + Codex) tracking publication readiness"
    )
    development_status: Optional[DevelopmentStatus] = Field(
        None, description="Development maturity and completeness tracking"
    )
    sketch_linkage: Optional[SketchProofLinkage] = Field(
        None, description="Link to proof sketch and expansion history (if applicable)"
    )


class ObservationEvidence(BaseModel):
    """Evidence for an observation."""

    type: EvidenceType
    description: str
    reference: Optional[str] = None


class Observation(BaseDirective):
    """An empirical or mathematical observation."""

    type: Literal["observation"] = "observation"
    empirical: Optional[bool] = Field(
        None, description="Whether this is an empirical observation (vs. mathematical)"
    )
    evidence: Optional[List[ObservationEvidence]] = Field(
        None, description="Supporting evidence for this observation"
    )
    peer_review: Optional[DualReviewAnalysis] = Field(
        None, description="Dual review analysis (Gemini + Codex) tracking publication readiness"
    )
    development_status: Optional[DevelopmentStatus] = Field(
        None, description="Development maturity and completeness tracking"
    )
    sketch_linkage: Optional[SketchProofLinkage] = Field(
        None, description="Link to proof sketch and expansion history (if applicable)"
    )


class ConjectureEvidence(BaseModel):
    """Evidence for a conjecture."""

    type: EvidenceType
    description: str


class Conjecture(BaseDirective):
    """A mathematical conjecture."""

    type: Literal["conjecture"] = "conjecture"
    confidence: ConfidenceLevel = Field(..., description="Confidence level in this conjecture")
    evidence: List[ConjectureEvidence] = Field(..., description="Supporting evidence")
    partial_results: Optional[List[CrossReference]] = Field(
        None, description="Partial results supporting this conjecture"
    )
    obstacles: Optional[List[str]] = Field(
        None, description="Known obstacles to proving this conjecture"
    )
    peer_review: Optional[DualReviewAnalysis] = Field(
        None, description="Dual review analysis (Gemini + Codex) tracking publication readiness"
    )
    development_status: Optional[DevelopmentStatus] = Field(
        None, description="Development maturity and completeness tracking"
    )
    sketch_linkage: Optional[SketchProofLinkage] = Field(
        None, description="Link to proof sketch and expansion history (if applicable)"
    )


class Example(BaseDirective):
    """A worked example."""

    type: Literal["example"] = "example"
    demonstrates: List[CrossReference] = Field(
        ..., description="What concepts/results this example illustrates"
    )
    setup: Optional[str] = Field(None, description="Description of the example setup")
    calculation: Optional[str] = Field(None, description="Worked calculation or construction")
    conclusion: Optional[str] = Field(None, description="What this example shows")
    computational_verification: Optional[ComputationalVerification] = None
    peer_review: Optional[DualReviewAnalysis] = Field(
        None, description="Dual review analysis (Gemini + Codex) tracking publication readiness"
    )
    development_status: Optional[DevelopmentStatus] = Field(
        None, description="Development maturity and completeness tracking"
    )
    sketch_linkage: Optional[SketchProofLinkage] = Field(
        None, description="Link to proof sketch and expansion history (if applicable)"
    )


class Property(BaseDirective):
    """A mathematical property."""

    type: Literal["property"] = "property"
    applies_to: List[CrossReference] = Field(
        ..., description="What objects have this property"
    )
    property_type: Optional[PropertyType] = None
    peer_review: Optional[DualReviewAnalysis] = Field(
        None, description="Dual review analysis (Gemini + Codex) tracking publication readiness"
    )
    development_status: Optional[DevelopmentStatus] = Field(
        None, description="Development maturity and completeness tracking"
    )
    sketch_linkage: Optional[SketchProofLinkage] = Field(
        None, description="Link to proof sketch and expansion history (if applicable)"
    )


# ==================== Dependency Graph ====================


class DependencyEdge(BaseModel):
    """Edge in the dependency graph."""

    from_: Label = Field(..., alias="from", description="Source label")
    to: Label = Field(..., description="Target label")
    relationship: str = Field(
        ...,
        description="Type of dependency (e.g., 'requires', 'builds-on', 'proves', 'uses')",
    )
    critical: Optional[bool] = Field(
        None, description="Whether this is a critical dependency for the proof/result"
    )
    description: Optional[str] = Field(None, description="Brief explanation of the dependency")

    class Config:
        populate_by_name = True


class DependencyGraph(BaseModel):
    """Dependency graph showing relationships between directives."""

    edges: List[DependencyEdge] = Field(..., description="Edges in the dependency graph")


# ==================== Metadata ====================


class ReviewStatusEntry(BaseModel):
    """Review status for a single reviewer."""

    date: Optional[date] = None
    status: Optional[PeerReviewStatus] = None
    review_file: Optional[str] = Field(None, description="Path to review output")


class DocumentPeerReviewStatus(BaseModel):
    """Peer review status tracking for the document."""

    gemini_review: Optional[ReviewStatusEntry] = None
    codex_review: Optional[ReviewStatusEntry] = None
    dual_review_consensus: Optional[bool] = Field(
        None, description="Whether both reviewers agree on approval"
    )


class DirectiveSummary(BaseModel):
    """Summary of directive-level readiness."""

    total_directives: int
    reviewed_directives: int
    ready_count: int
    minor_revisions_count: int
    major_revisions_count: int
    reject_count: int


class BlockingIssue(BaseModel):
    """Critical issue blocking publication."""

    directive_label: str
    directive_type: str
    issue: ReviewIssue


class DevelopmentSummary(BaseModel):
    """Summary of development maturity across directives."""

    sketch_count: int
    partial_count: int
    complete_count: int
    verified_count: int
    published_count: int
    average_completeness: float = Field(..., ge=0.0, le=100.0)


class PublicationReadinessAggregate(BaseModel):
    """Aggregate publication readiness metrics across all directives."""

    overall_verdict: Literal["ready", "minor-revisions", "major-revisions", "reject", "not-reviewed"]
    aggregate_scores: Optional[AggregateScore] = None
    directive_summary: Optional[DirectiveSummary] = None
    blocking_issues: Optional[List[BlockingIssue]] = None
    development_summary: Optional[DevelopmentSummary] = None
    last_updated: Optional[datetime] = None


class Metadata(BaseModel):
    """Document-level metadata."""

    title: str = Field(
        ..., description="Document title (e.g., 'The Keystone Principle and the Contractive Nature of Cloning')"
    )
    document_id: str = Field(
        ..., description="Unique document identifier (e.g., '03_cloning', '11_geometric_gas')"
    )
    version: constr(pattern=r"^\d+\.\d+(\.\d+)?$") = Field(  # type: ignore
        ..., description="Semantic version (e.g., '1.0.0', '2.3.1')"
    )
    chapter: Optional[int] = Field(None, description="Chapter number in the overall framework")
    authors: Optional[List[str]] = None
    date_created: Optional[date] = None
    date_modified: Optional[date] = None
    rigor_level: Optional[RigorLevel] = Field(None, description="Overall rigor level of the document")
    peer_review_status: Optional[DocumentPeerReviewStatus] = Field(
        None, description="Dual review protocol status (as per CLAUDE.md)"
    )
    dependencies: Optional[List[str]] = Field(
        None, description="Other documents this one depends on (prerequisite reading)"
    )
    abstract: Optional[str] = Field(
        None, description="Brief abstract summarizing the document (TLDR section)"
    )
    publication_readiness_aggregate: Optional[PublicationReadinessAggregate] = Field(
        None, description="Aggregate publication readiness metrics across all directives"
    )


# ==================== Main Document ====================


# Union of all directive types
Directive = Union[
    Definition,
    Axiom,
    Theorem,
    Lemma,
    Proposition,
    Corollary,
    Proof,
    Algorithm,
    Remark,
    Observation,
    Conjecture,
    Example,
    Property,
]


class MathematicalDocument(BaseModel):
    """Complete mathematical document."""

    metadata: Metadata = Field(..., description="Document metadata")
    directives: List[Directive] = Field(..., description="Mathematical directives")
    dependency_graph: Optional[DependencyGraph] = Field(
        None, description="Dependency graph showing relationships between directives"
    )
    notation_index: Optional[Dict[str, Any]] = Field(
        None, description="Index of mathematical notation used"
    )
    constants_glossary: Optional[Dict[str, Any]] = Field(
        None, description="Glossary of constants defined and used"
    )

    class Config:
        json_schema_extra = {
            "title": "Fragile Gas Mathematical Documentation Schema",
            "description": "Comprehensive schema for rigorous mathematical documentation in the Fragile Gas framework.",
            "version": "1.0.0",
        }
