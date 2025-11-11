#!/usr/bin/env python3
"""DSPy signatures, pydantic models, and review agents for proof sketch validation."""

from __future__ import annotations

import json
from typing import Literal

import dspy
from pydantic import BaseModel, Field

from mathster.proof_sketcher.tools import configure_claude_tool, configure_search_tool


__all__ = [
    "CoTReviewSignature",
    "CompletenessCorrectnessReview",
    "CompletenessCorrectnessSignature",
    "DependencyIssue",
    "DependencyValidationReview",
    "DependencyValidationSignature",
    "LogicalFlowReview",
    "LogicalFlowValidationSignature",
    "OverallAssessment",
    "OverallAssessmentSignature",
    "SketchReviewAgent",
    "SketchValidationReview",
    "SketchValidationReviewSignature",
    "TechnicalDeepDiveCritique",
    "TechnicalDeepDiveValidation",
    "TechnicalDeepDiveValidationSignature",
]


ConfidenceScore = Literal[1, 2, 3, 4, 5]

Recommendation = Literal[
    "Proceed to Expansion",
    "Revise and Resubmit for Validation",
    "Strategy is Flawed - A New Sketch is Recommended",
]

DependencyStatus = Literal["Complete and Correct", "Minor Issues Found", "Major Issues Found"]
DependencyIssueType = Literal[
    "Incorrectly Used", "Preconditions Not Met", "Missing Dependency", "Citation Error"
]

SolutionViability = Literal[
    "Viable and Well-Described",
    "Plausible but Requires More Detail",
    "Potentially Flawed",
    "Flawed",
]

class BaseAgent(dspy.Module):
    data_model: TypeVar[BaseModel]  # type: ignore
    signature_cls: TypeVar[dspy.Signature]  # type: ignore
    agent_cls: TypeVar[dspy.Module]  # type: ignore
    instructions: str

    def __init__(self, data_model=None, signature_cls=None, agent_cls=None, instructions=None) -> None:
        super().__init__()
        self.data_model = data_model or self.data_model
        self.signature_cls = signature_cls or self.signature_cls
        self.agent_cls = agent_cls or self.agent_cls
        self.instructions = instructions or self.instructions
        self.instructions = self.render_instructions(self.instructions)
        self.signature = self.signature_cls.with_instructions(self.instructions)
        self.agent = self.agent_cls(signature=self.signature)

    def render_instructions(self, instructions: str) -> str:
        schema = json.dumps(CompletenessCorrectnessReview.model_json_schema(), indent=2)
        return instructions.format(schema=schema)
    
    def forward(self, **kwargs) -> dspy.Prediction:
        return self.agent(**kwargs)
    
    async def aforward(self, **kwargs) -> dspy.Prediction:
        return await self.agent.aforward(**kwargs)


# ---------------------------------
# Completeness and Correctness Components
# ---------------------------------
class IdentifiedError(BaseModel):
    """A concrete mathematical error or mismatch in the proof sketch."""

    location: str = Field(..., description="Reference to the specific step or section.")
    description: str = Field(..., description="Details of the error or inconsistency.")
    suggestedCorrection: str = Field(..., description="Proposed fix or remedy.")

class CompletenessCorrectnessReview(BaseModel):
    """Assessment of coverage and mathematical correctness."""

    coversAllClaims: bool = Field(..., description="Does the sketch cover every claim?")
    identifiedErrors: list[IdentifiedError] = Field(
        default_factory=list,
        description="List of explicit mathematical errors or typos found.",
    )
    score: int = Field(
        ...,
        description="Numeric quality score from 1-5. 1 means unusable, 5 means publication-quality."
    )
    confidence: int = Field(
        ...,
        description="Numeric confidence score from 1-5 indicating reviewer's confidence in the assessment."
    )

class CompletenessCorrectnessSignature(dspy.Signature):
    """Check if the sketch covers the full theorem and is mathematically correct."""

    theorem_statement = dspy.InputField(desc="Formal theorem statement for reference.")
    proof_sketch_text = dspy.InputField(
        desc="Full proof sketch text or concatenated steps to analyze coverage."
    )

    completenessAndCorrectness: CompletenessCorrectnessReview = dspy.OutputField(
        desc="Coverage/correctness evaluation result."
    )


class CompletenessCorrectnessAgent(BaseAgent):
    """Tool-augmented reviewer that scores proof sketch completeness/correctness."""
    data_model = CompletenessCorrectnessReview
    signature_cls = CompletenessCorrectnessSignature
    agent_cls = dspy.Predict
    instructions = """
You audit a Fragile proof sketch for completeness and correctness.

Workflow:
1. Decompose the theorem_statement into explicit claims/conditions.
2. Cross-check each claim against the proof_sketch_text.
3. Record every mathematical error or missing coverage as an IdentifiedError with
   the exact location, issue description, and a concrete fix.
4. Set coversAllClaims=True only if every claim has adequate reasoning AND no blocking errors.
5. Score rubric (1=worse, 5=publication-ready):
   - 5: complete coverage, no errors.
   - 4: minor gaps/clarity issues but claims supported.
   - 3: partial coverage or moderate mistakes that require revision.
   - 2: major gaps or incorrect arguments invalidate key parts.
   - 1: unusable or contradicts hypotheses.
6. Confidence reflects how certain you are in the assessment (1-5).

Use available tools when needed:
- search_project(query): inspect repository files or locate references.
- ask_claude(prompt): consult Claude with the shared system prompt for deeper analysis.

Return ONLY JSON matching:
{schema}
"""


# ---------------------------------
# Logical Flow Validation Components
# ---------------------------------
class LogicalFlowReview(BaseModel):
    """Assessment of logical structure and step progression."""

    isSound: bool = Field(..., description="True if the argument is coherent and valid.")
    comments: str = Field(..., description="Narrative assessment of clarity and structure.")
    identifiedGaps: list[str] = Field(
        default_factory=list,
        description="Specific logical gaps or leaps of faith that require attention.",
    )
    score: int = Field(
        ...,
        description="Numeric quality score from 1-5. 1 means unusable, 5 means publication-quality."
    )
    confidence: int = Field(
        ...,
        description="Numeric confidence score from 1-5 indicating reviewer's confidence in the assessment."
    )

class LogicalFlowValidationSignature(dspy.Signature):
    """Evaluate the logical flow of the proof sketch."""

    proof_outline = dspy.InputField(
        desc="Top-level outline of the proof steps and transitions between them."
    )
    reasoning_notes = dspy.InputField(
        desc="Additional reviewer notes about logical flow (optional).", optional=True
    )

    logicalFlowValidation: LogicalFlowReview = dspy.OutputField(
        desc="Logical flow validation outcome."
    )

class AgentLogicalFlowValidation(BaseAgent):
    """ Agent using a dspy.ChainOfThought to evaluate the logical flow of a proof sketch. """


# ---------------------------------
# Dependency Validation Components
# ---------------------------------
class DependencyIssue(BaseModel):
    """An issue linked to a specific dependency label."""

    label: str = Field(..., description="Framework label (thm-*, lem-*, def-*, ...).")
    issueType: DependencyIssueType = Field(..., description="Classification of the issue.")
    comment: str = Field(..., description="Explanation of the issue and impact.")


class DependencyValidationReview(BaseModel):
    """Verification of framework references."""

    status: DependencyStatus = Field(..., description="Overall status of dependency usage.")
    issues: list[DependencyIssue] = Field(
        default_factory=list, description="Specific dependency issues that were detected."
    )
class DependencyValidationSignature(dspy.Signature):
    """Inspect framework dependency usage in the proof sketch."""

    dependency_ledger_json = dspy.InputField(
        desc="JSON text describing verified and missing dependencies."
    )
    sketch_references = dspy.InputField(
        desc="Excerpt of the proof sketch referencing dependencies (optional).", optional=True
    )

    dependencyValidation: DependencyValidationReview = dspy.OutputField(
        desc="Structured dependency validation assessment."
    )
    score = dspy.OutputField(
        desc="Numeric quality score from 1-5. 1 means unusable, 5 means publication-quality."
    )
    confidence = dspy.OutputField(
        desc="Numeric confidence score from 1-5 indicating reviewer's confidence in the assessment."
    )

class AgentDependencyValidation(BaseAgent):
    """ Agent using a dspy.ChainOfThought to validate framework dependencies in a proof sketch implementing DependencyValidationSignature. """

# ---------------------------------
# Technical Deep Dive Components
# ---------------------------------
class TechnicalDeepDiveCritique(BaseModel):
    """Review of a single technical challenge and proposed fix."""

    challengeTitle: str = Field(..., description="Matching title from the original sketch.")
    solutionViability: SolutionViability = Field(..., description="Assessment of feasibility.")
    critique: str = Field(
        ..., description="Detailed analysis highlighting strengths and weaknesses."
    )
    suggestedImprovements: str | None = Field(
        default=None,
        description="Optional concrete suggestions for strengthening the argument.",
    )


class TechnicalDeepDiveValidation(BaseModel):
    """Collection of critiques for technical deep dives."""

    critiques: list[TechnicalDeepDiveCritique] = Field(
        default_factory=list,
        description="One entry per challenging component in the sketch.",
    )

class TechnicalDeepDiveValidationSignature(dspy.Signature):
    """Critique the sketch's technical deep dives."""

    deep_dives_json = dspy.InputField(
        desc="JSON text representing the sketch's technical deep dives."
    )
    reviewer_focus = dspy.InputField(
        desc="Optional notes about what to scrutinize (e.g., regularity, bounds).", optional=True
    )

    technicalDeepDiveValidation: TechnicalDeepDiveValidation = dspy.OutputField(
        desc="List of critiques mapped to the original challenges."
    )
    score = dspy.OutputField(
        desc="Numeric quality score from 1-5. 1 means unusable, 5 means publication-quality."
    )
    confidence = dspy.OutputField(
        desc="Numeric confidence score from 1-5 indicating reviewer's confidence in the assessment."
    )

class AgentTechnicalDeepDiveValidation(BaseAgent):
    """ Agent using a dspy.ChainOfThought to critique the technical deep dives of a proof sketch. """

# ---------------------------------
# Overall Assessment Components
# ---------------------------------
class OverallAssessment(BaseModel):
    """High-level summary and recommendation."""

    confidenceScore: ConfidenceScore = Field(
        ...,
        description=(
            "Score from 1-5 (1=Unusable, 2=Critical but fixable, 3=Major revisions, "
            "4=Minor revisions, 5=Publication-quality)."
        ),
    )
    summary: str = Field(..., description="Concise summary of core findings.")
    recommendation: Recommendation = Field(
        ..., description="Actionable recommendation for the next step."
    )
    score: int = Field(
        ...,
        description="Numeric quality score from 1-5. 1 means unusable, 5 means publication-quality."
    )
    confidence: int = Field(
        ...,
        description="Numeric confidence score from 1-5 indicating reviewer's confidence in the assessment."
    )

class OverallAssessmentSignature(dspy.Signature):
    """Generate the high-level assessment summary."""

    extra_instructions = dspy.InputField(desc="Additional instructions for the reviewer.")
    proof_sketch_json = dspy.InputField(desc="JSON representation of the proof sketch.")
    logicalFlowValidation: LogicalFlowReview = dspy.InputField(
        desc="Logic and structure review."
    )
    dependencyValidation: DependencyValidationReview = dspy.InputField(
        desc="Dependency verification results."
    )
    technicalDeepDiveValidation: TechnicalDeepDiveValidation = dspy.InputField(
        desc="Critiques of difficult components."
    )
    completenessAndCorrectness: CompletenessCorrectnessReview = dspy.InputField(
        desc="Coverage and correctness analysis."
    )

    overallAssessment: OverallAssessment = dspy.OutputField(
        desc="Structured overall assessment matching sketch_validation_request schema."
    )

class AgentOverallAssessment(BaseAgent):
    """ Agent using a dspy.ChainOfThought to generate the overall assessment of a proof sketch. """

#
class SketchValidationReview(BaseModel):
    """Top-level review combining assessment and detailed analysis."""

    reviewer: str = Field(..., description="Reviewer identity (model, version, etc.).")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the review.")
    overallAssessment: OverallAssessment = Field(..., description="Summary + recommendation.")
    logicalFlowValidation: LogicalFlowReview = Field(
        ..., description="Logic and structure review."
    )
    dependencyValidation: DependencyValidationReview = Field(
        ..., description="Dependency verification results."
    )
    technicalDeepDiveValidation: TechnicalDeepDiveValidation = Field(
        ..., description="Critiques of difficult components."
    )
    completenessAndCorrectness: CompletenessCorrectnessReview = Field(
        ..., description="Coverage and correctness analysis."
    )


class SketchReviewAgent(dspy.Module):
    """This agent calls all the other agents in order and handles the "glue code" to produce a full sketch validation review. as described by SketchValidationReview.
    It is a composition of modules taht implements no new module on its own as it only does orchestration.
    """
