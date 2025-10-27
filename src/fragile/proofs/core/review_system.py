"""
Review System for Automated Proof Development Workflow.

This module implements a comprehensive review and evaluation system for mathematical
objects in the proof pipeline. Reviews are stored in an external registry and linked
by object ID, enabling efficient iteration tracking and validation integration.

Architecture:
- ReviewIssue: Individual problem identified in review
- ValidationResult: Result from automated validators
- Review: Complete review of a mathematical object with scores and issues
- ReviewComparison: Comparison of two reviews (for dual review protocol)

All types follow Lean-compatible patterns from docs/LEAN_EMULATION_GUIDE.md:
- frozen=True (immutability)
- Pure functions (no side effects)
- Total functions (Optional[T] instead of exceptions)
- Explicit types (no Any where possible)

Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================


class ReviewSeverity(str, Enum):
    """
    Severity levels for review issues.

    Maps to Lean:
        inductive ReviewSeverity where
          | critical : ReviewSeverity
          | major : ReviewSeverity
          ...
    """

    CRITICAL = "critical"  # Invalidates theorem/proof
    MAJOR = "major"  # Significant gap requiring rework
    MINOR = "minor"  # Subtle error or improvement
    SUGGESTION = "suggestion"  # Optional enhancement
    VALIDATION_FAILURE = "validation_failure"  # Automated validation failed
    FRAMEWORK_INCONSISTENCY = "framework_inconsistency"  # Conflicts with framework docs


class ReviewSource(str, Enum):
    """
    Source of a review (LLM, validator, or human).

    Maps to Lean:
        inductive ReviewSource where
          | gemini : ReviewSource
          | codex : ReviewSource
          ...
    """

    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    CODEX = "codex"
    CLAUDE_SYNTHESIS = "claude-synthesis"
    SYMPY_VALIDATOR = "sympy-validator"
    DATAFLOW_VALIDATOR = "dataflow-validator"
    FRAMEWORK_CHECKER = "framework-checker"
    HUMAN_EXPERT = "human-expert"
    OTHER = "other"


class ReviewStatus(str, Enum):
    """Overall status of a reviewed object."""

    READY = "ready"  # No blocking issues
    NEEDS_MINOR_REVISION = "needs-minor-revision"  # Only MINOR/SUGGESTION issues
    NEEDS_MAJOR_REVISION = "needs-major-revision"  # Has MAJOR issues
    BLOCKED = "blocked"  # Has CRITICAL issues or VALIDATION_FAILURE
    NOT_REVIEWED = "not-reviewed"  # No review exists


class IssueResolution(str, Enum):
    """How an issue was resolved."""

    FIXED = "fixed"  # Issue addressed and resolved
    REJECTED = "rejected"  # Reviewer was wrong, no fix needed
    DEFERRED = "deferred"  # Will fix in future iteration
    NEEDS_CLARIFICATION = "needs-clarification"  # Need more info from reviewer


# =============================================================================
# REVIEW ISSUE
# =============================================================================


class ReviewIssue(BaseModel):
    """
    Individual issue identified during review.

    This is the fundamental unit of review feedback: a specific problem
    with location, severity, explanation, and suggested fix.

    Maps to Lean:
        structure ReviewIssue where
          id : String
          severity : ReviewSeverity
          location : String
          title : String
          problem : String
          ...
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    id: str = Field(..., pattern=r"^issue-[a-z0-9-]+$", description="Unique issue ID")

    # Classification
    severity: ReviewSeverity = Field(..., description="Issue severity level")
    location: str = Field(
        ...,
        min_length=1,
        description="Location in object (e.g., 'step-2', 'lines 45-50', 'property prop-lipschitz')",
    )

    # Description
    title: str = Field(..., min_length=1, description="Brief descriptive title")
    problem: str = Field(..., min_length=1, description="Clear description of what is wrong or missing")
    mechanism: Optional[str] = Field(
        None, description="WHY this fails - precise mechanism of the error"
    )
    evidence: Optional[str] = Field(
        None,
        description="Quoted passage, counterexample, or calculation demonstrating the issue",
    )

    # Impact and fix
    impact: List[str] = Field(
        default_factory=list,
        description="Affected objects/theorems/properties (labels)",
    )
    suggested_fix: Optional[str] = Field(
        None, description="Concrete suggestion for how to fix this issue"
    )

    # Metadata
    actionability_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How clear/actionable the fix is (0=unclear, 1=very clear)",
    )
    framework_references: List[str] = Field(
        default_factory=list,
        description="References to framework docs (e.g., 'lem-gronwall-inequality' from glossary.md)",
    )
    validation_failure: Optional[Dict[str, Any]] = Field(
        None, description="Details if this is from automated validation"
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validator → Lean proof obligation: issue ID well-formed."""
        if not v.startswith("issue-"):
            raise ValueError(f"Issue IDs must start with 'issue-': {v}")
        return v

    # Pure function: Check if blocking
    def is_blocking(self) -> bool:
        """
        Pure function: Check if this issue blocks publication.

        Blocking issues are CRITICAL, MAJOR, or VALIDATION_FAILURE.

        Maps to Lean:
            def is_blocking (issue : ReviewIssue) : Bool :=
              issue.severity == ReviewSeverity.critical ||
              issue.severity == ReviewSeverity.major ||
              issue.severity == ReviewSeverity.validation_failure
        """
        return self.severity in (
            ReviewSeverity.CRITICAL,
            ReviewSeverity.MAJOR,
            ReviewSeverity.VALIDATION_FAILURE,
        )

    # Pure function: Get severity weight
    def get_severity_weight(self) -> float:
        """
        Pure function: Get numeric weight for severity (for scoring).

        Maps to Lean:
            def get_severity_weight (s : ReviewSeverity) : Float :=
              match s with
              | critical => 10.0
              | major => 5.0
              ...
        """
        weights = {
            ReviewSeverity.CRITICAL: 10.0,
            ReviewSeverity.MAJOR: 5.0,
            ReviewSeverity.MINOR: 2.0,
            ReviewSeverity.SUGGESTION: 0.5,
            ReviewSeverity.VALIDATION_FAILURE: 8.0,
            ReviewSeverity.FRAMEWORK_INCONSISTENCY: 6.0,
        }
        return weights[self.severity]


# =============================================================================
# LLM RESPONSE
# =============================================================================


class LLMResponse(BaseModel):
    """
    LLM's response to a review issue (how it was addressed).

    Tracks which issues were implemented, rejected, or deferred,
    enabling iteration analysis.

    Maps to Lean:
        structure LLMResponse where
          issue_id : String
          implemented : Bool
          result : IssueResolution
          ...
    """

    model_config = ConfigDict(frozen=True)

    issue_id: str = Field(..., pattern=r"^issue-[a-z0-9-]+$", description="Issue being addressed")
    implemented: bool = Field(..., description="Whether fix was implemented")
    result: IssueResolution = Field(..., description="How issue was resolved")

    # Details
    implementation_notes: Optional[str] = Field(
        None, description="Notes on how issue was fixed"
    )
    rejection_reason: Optional[str] = Field(
        None, description="Why fix was rejected (if result=rejected)"
    )

    # Metadata
    timestamp: datetime = Field(..., description="When response was recorded")
    llm_agent: Optional[str] = Field(
        None, description="Which agent implemented the fix (e.g., 'claude', 'human')"
    )

    @field_validator("issue_id")
    @classmethod
    def validate_issue_id(cls, v: str) -> str:
        """Validator → Lean proof obligation: issue ID well-formed."""
        if not v.startswith("issue-"):
            raise ValueError(f"Issue IDs must start with 'issue-': {v}")
        return v


# =============================================================================
# VALIDATION RESULT
# =============================================================================


class ValidationResult(BaseModel):
    """
    Result from automated validation (dataflow, SymPy, framework checker, etc.).

    Links automated checks to reviews, providing evidence for validation failures.

    Maps to Lean:
        structure ValidationResult where
          validator : String
          passed : Bool
          errors : List String
          timestamp : DateTime
          metadata : HashMap String Any
    """

    model_config = ConfigDict(frozen=True)

    validator: str = Field(
        ...,
        min_length=1,
        description="Validator name (e.g., 'dataflow', 'sympy', 'framework-checker')",
    )
    passed: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(
        default_factory=list, description="Error messages if validation failed"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Non-fatal warnings"
    )
    timestamp: datetime = Field(..., description="When validation was performed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validator-specific metadata (e.g., SymPy simplification results)",
    )

    # Pure function: Get severity
    def get_severity(self) -> ReviewSeverity:
        """
        Pure function: Convert validation result to severity.

        Failed validation → VALIDATION_FAILURE
        Warnings only → SUGGESTION

        Maps to Lean:
            def get_severity (vr : ValidationResult) : ReviewSeverity :=
              if !vr.passed then ReviewSeverity.validation_failure
              else if !vr.warnings.isEmpty then ReviewSeverity.suggestion
              else ReviewSeverity.suggestion  -- No issues
        """
        if not self.passed:
            return ReviewSeverity.VALIDATION_FAILURE
        elif self.warnings:
            return ReviewSeverity.SUGGESTION
        return ReviewSeverity.SUGGESTION  # No issues


# =============================================================================
# REVIEW
# =============================================================================


class Review(BaseModel):
    """
    Complete review of a mathematical object.

    This is the main review data structure, containing:
    - Issues identified
    - Scores (rigor, soundness, completeness, etc.)
    - Validation results
    - Iteration tracking
    - LLM responses

    Maps to Lean:
        structure Review where
          review_id : String
          object_id : String
          object_type : String
          iteration : Nat
          source : ReviewSource
          issues : List ReviewIssue
          validation_results : List ValidationResult
          ...
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    review_id: str = Field(
        ...,
        pattern=r"^review-[a-z0-9-]+-iter\d+-[a-z0-9.-]+$",
        description="Unique review ID (format: review-{object_id}-iter{N}-{source})",
    )
    object_id: str = Field(..., min_length=1, description="ID of object being reviewed")
    object_type: str = Field(
        ...,
        min_length=1,
        description="Type of object (ProofBox, ProofStep, TheoremBox, MathematicalObject, etc.)",
    )

    # Review metadata
    iteration: int = Field(..., ge=0, description="Iteration number (0-indexed)")
    timestamp: datetime = Field(..., description="When review was performed")
    source: ReviewSource = Field(..., description="Source of this review")

    # Review content
    issues: List[ReviewIssue] = Field(default_factory=list, description="Issues identified")
    strengths: List[str] = Field(
        default_factory=list, description="Positive aspects identified"
    )
    overall_assessment: str = Field(
        ..., min_length=1, description="Summary assessment of the object"
    )

    # Scores (0-10 scale)
    rigor_score: float = Field(
        ..., ge=0.0, le=10.0, description="Mathematical rigor score (1-10)"
    )
    soundness_score: float = Field(
        ..., ge=0.0, le=10.0, description="Logical soundness score (1-10)"
    )
    completeness_score: float = Field(
        ..., ge=0.0, le=10.0, description="Completeness score (1-10)"
    )
    clarity_score: float = Field(
        ..., ge=0.0, le=10.0, description="Clarity/readability score (1-10)"
    )
    framework_consistency_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Framework consistency score (1-10)",
    )

    # Computed metrics
    mean_actionability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean actionability score of all issues",
    )
    blocking_issue_count: int = Field(
        ..., ge=0, description="Number of CRITICAL/MAJOR/VALIDATION_FAILURE issues"
    )

    # Validation integration
    validation_results: List[ValidationResult] = Field(
        default_factory=list, description="Automated validation results"
    )
    validation_passed: bool = Field(
        ..., description="Whether all automated validations passed"
    )

    # Iteration tracking
    previous_review_id: Optional[str] = Field(
        None,
        pattern=r"^review-[a-z0-9-]+-iter\d+-[a-z0-9.-]+$",
        description="Previous review in iteration chain",
    )
    addresses_issues: List[str] = Field(
        default_factory=list,
        description="Issue IDs from previous iteration that this addresses",
    )
    llm_responses: List[LLMResponse] = Field(
        default_factory=list,
        description="LLM responses to issues from previous iteration",
    )

    @field_validator("review_id")
    @classmethod
    def validate_review_id(cls, v: str) -> str:
        """Validator → Lean proof obligation: review ID well-formed."""
        if not v.startswith("review-"):
            raise ValueError(f"Review IDs must start with 'review-': {v}")
        if "-iter" not in v:
            raise ValueError(f"Review IDs must contain '-iterN-': {v}")
        return v

    # Pure function: Get overall status
    def get_status(self) -> ReviewStatus:
        """
        Pure function: Compute overall status from issues.

        Maps to Lean:
            def get_status (r : Review) : ReviewStatus :=
              if r.blocking_issue_count > 0 then ReviewStatus.blocked
              else if (r.issues.filter (fun i => i.severity == ReviewSeverity.major)).length > 0
                then ReviewStatus.needs_major_revision
              ...
        """
        if self.blocking_issue_count > 0:
            return ReviewStatus.BLOCKED

        has_major = any(i.severity == ReviewSeverity.MAJOR for i in self.issues)
        if has_major:
            return ReviewStatus.NEEDS_MAJOR_REVISION

        has_minor = any(
            i.severity in (ReviewSeverity.MINOR, ReviewSeverity.SUGGESTION)
            for i in self.issues
        )
        if has_minor:
            return ReviewStatus.NEEDS_MINOR_REVISION

        return ReviewStatus.READY

    # Pure function: Get issues by severity
    def get_issues_by_severity(self, severity: ReviewSeverity) -> List[ReviewIssue]:
        """
        Pure function: Filter issues by severity.

        Maps to Lean:
            def get_issues_by_severity (r : Review) (s : ReviewSeverity) : List ReviewIssue :=
              r.issues.filter (fun i => i.severity == s)
        """
        return [i for i in self.issues if i.severity == severity]

    # Pure function: Get blocking issues
    def get_blocking_issues(self) -> List[ReviewIssue]:
        """
        Pure function: Get all blocking issues.

        Maps to Lean:
            def get_blocking_issues (r : Review) : List ReviewIssue :=
              r.issues.filter (fun i => i.is_blocking)
        """
        return [i for i in self.issues if i.is_blocking()]

    # Pure function: Compute average score
    def get_average_score(self) -> float:
        """
        Pure function: Compute average of all scores.

        Maps to Lean:
            def get_average_score (r : Review) : Float :=
              (r.rigor_score + r.soundness_score + ... ) / 5.0
        """
        scores = [
            self.rigor_score,
            self.soundness_score,
            self.completeness_score,
            self.clarity_score,
            self.framework_consistency_score,
        ]
        return sum(scores) / len(scores)

    # Pure function: Check if improved from previous
    def is_improvement_over(self, previous: Review) -> bool:
        """
        Pure function: Check if this review shows improvement over previous.

        Improvement if:
        - Fewer blocking issues
        - Higher average score
        - More issues resolved

        Maps to Lean:
            def is_improvement_over (r1 r2 : Review) : Bool :=
              r1.blocking_issue_count < r2.blocking_issue_count ||
              r1.get_average_score > r2.get_average_score
        """
        if self.blocking_issue_count < previous.blocking_issue_count:
            return True
        if self.get_average_score() > previous.get_average_score():
            return True
        return False


# =============================================================================
# REVIEW COMPARISON (Dual Review)
# =============================================================================


class ReviewComparison(BaseModel):
    """
    Comparison of two reviews (for dual review protocol).

    Used to compare Gemini vs Codex reviews, identifying:
    - Consensus issues (both found) → high confidence
    - Discrepancies (contradictory) → requires manual verification
    - Unique issues (only one found) → medium confidence

    Maps to Lean:
        structure ReviewComparison where
          review_a : Review
          review_b : Review
          consensus_issues : List ReviewIssue
          discrepancies : List (ReviewIssue × ReviewIssue)
          ...
    """

    model_config = ConfigDict(frozen=True)

    # Reviews being compared
    review_a: Review = Field(..., description="First review (e.g., Gemini)")
    review_b: Review = Field(..., description="Second review (e.g., Codex)")

    # Comparison results
    consensus_issues: List[ReviewIssue] = Field(
        default_factory=list,
        description="Issues identified by both reviewers (high confidence)",
    )
    discrepancies: List[Tuple[ReviewIssue, ReviewIssue]] = Field(
        default_factory=list,
        description="Contradictory findings requiring manual verification",
    )
    unique_to_a: List[ReviewIssue] = Field(
        default_factory=list, description="Issues only in review A"
    )
    unique_to_b: List[ReviewIssue] = Field(
        default_factory=list, description="Issues only in review B"
    )

    # Confidence weights
    confidence_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence weights (consensus=1.0, unique=0.5, discrepancy=0.0)",
    )

    # Pure function: Get all high-confidence issues
    def get_high_confidence_issues(self) -> List[ReviewIssue]:
        """
        Pure function: Get all high-confidence issues (consensus only).

        Maps to Lean:
            def get_high_confidence_issues (rc : ReviewComparison) : List ReviewIssue :=
              rc.consensus_issues
        """
        return self.consensus_issues

    # Pure function: Get all medium-confidence issues
    def get_medium_confidence_issues(self) -> List[ReviewIssue]:
        """
        Pure function: Get all medium-confidence issues (unique to either reviewer).

        Maps to Lean:
            def get_medium_confidence_issues (rc : ReviewComparison) : List ReviewIssue :=
              rc.unique_to_a ++ rc.unique_to_b
        """
        return self.unique_to_a + self.unique_to_b

    # Pure function: Check if reviewers agree
    def reviewers_agree(self) -> bool:
        """
        Pure function: Check if reviewers agree (no discrepancies, minimal unique issues).

        Maps to Lean:
            def reviewers_agree (rc : ReviewComparison) : Bool :=
              rc.discrepancies.isEmpty &&
              (rc.unique_to_a.length + rc.unique_to_b.length) <= 2
        """
        return (
            len(self.discrepancies) == 0
            and (len(self.unique_to_a) + len(self.unique_to_b)) <= 2
        )
