#!/usr/bin/env python3
"""DSPy signatures and pydantic models for proof sketch validation reviews."""

from __future__ import annotations

from typing import Callable, Literal

import dspy
from pydantic import BaseModel, Field

from mathster.proof_sketcher.tools import configure_claude_tool, configure_search_tool


__all__ = [
    "CompletenessCorrectnessReview",
    "CompletenessCorrectnessSignature",
    "DependencyIssue",
    "DependencyValidationReview",
    "DependencyValidationSignature",
    "LogicalFlowReview",
    "LogicalFlowValidationSignature",
    "OverallAssessment",
    "OverallAssessmentSignature",
    "SketchValidationReview",
    "SketchValidationReviewSignature",
    "TechnicalDeepDiveCritique",
    "TechnicalDeepDiveValidation",
    "TechnicalDeepDiveValidationSignature",
    "SketchReviewAgent",
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


class LogicalFlowReview(BaseModel):
    """Assessment of logical structure and step progression."""

    isSound: bool = Field(..., description="True if the argument is coherent and valid.")
    comments: str = Field(..., description="Narrative assessment of clarity and structure.")
    identifiedGaps: list[str] = Field(
        default_factory=list,
        description="Specific logical gaps or leaps of faith that require attention.",
    )


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


class TechnicalDeepDiveCritique(BaseModel):
    """Review of a single technical challenge and proposed fix."""

    challengeTitle: str = Field(..., description="Matching title from the original sketch.")
    solutionViability: SolutionViability = Field(..., description="Assessment of feasibility.")
    critique: str = Field(..., description="Detailed analysis highlighting strengths and weaknesses.")
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


class SketchValidationReview(BaseModel):
    """Top-level review combining assessment and detailed analysis."""

    reviewer: str = Field(..., description="Reviewer identity (model, version, etc.).")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the review.")
    overallAssessment: OverallAssessment = Field(..., description="Summary + recommendation.")
    logicalFlowValidation: LogicalFlowReview = Field(..., description="Logic and structure review.")
    dependencyValidation: DependencyValidationReview = Field(
        ..., description="Dependency verification results."
    )
    technicalDeepDiveValidation: TechnicalDeepDiveValidation = Field(
        ..., description="Critiques of difficult components."
    )
    completenessAndCorrectness: CompletenessCorrectnessReview = Field(
        ..., description="Coverage and correctness analysis."
    )


class OverallAssessmentSignature(dspy.Signature):
    """Generate the high-level assessment summary."""

    reviewer = dspy.InputField(desc="Name/model of the reviewer.")
    proof_sketch_summary = dspy.InputField(
        desc="Short narrative describing the proof sketch and its status."
    )
    key_findings = dspy.InputField(
        desc="List or paragraph summarizing critical findings from the review.",
    )

    overallAssessment: OverallAssessment = dspy.OutputField(
        desc="Structured overall assessment matching sketch_validation_request schema."
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


class CompletenessCorrectnessSignature(dspy.Signature):
    """Check if the sketch covers the full theorem and is mathematically correct."""

    theorem_statement = dspy.InputField(desc="Formal theorem statement for reference.")
    proof_sketch_text = dspy.InputField(
        desc="Full proof sketch text or concatenated steps to analyze coverage."
    )

    completenessAndCorrectness: CompletenessCorrectnessReview = dspy.OutputField(
        desc="Coverage/correctness evaluation result."
    )


class SketchValidationReviewSignature(dspy.Signature):
    """Assemble the full sketch validation review."""

    reviewer = dspy.InputField(desc="Reviewer name/model.")
    timestamp = dspy.InputField(desc="ISO 8601 timestamp for when the review is produced.")
    overallAssessment = dspy.InputField(desc="Serialized OverallAssessment JSON.")
    logicalFlowValidation = dspy.InputField(desc="Serialized LogicalFlowReview JSON.")
    dependencyValidation = dspy.InputField(desc="Serialized DependencyValidationReview JSON.")
    technicalDeepDiveValidation = dspy.InputField(
        desc="Serialized TechnicalDeepDiveValidation JSON."
    )
    completenessAndCorrectness = dspy.InputField(
        desc="Serialized CompletenessCorrectnessReview JSON."
    )

    review: SketchValidationReview = dspy.OutputField(
        desc="Complete review object complying with sketch_validation_request.json."
    )


class SketchReviewAgent(dspy.Module):
    """ReAct-based reviewer that orchestrates tools to produce a sketch validation review."""

    def __init__(
        self,
        project_root: str | None = None,
        claude_system_prompt: str | None = None,
    ) -> None:
        super().__init__()

        project_root = project_root or "."
        claude_system_prompt = (
            claude_system_prompt
            or ("You are an expert mathematical reviewer auditing proof sketches for publication "
            "that leverages the full access to the fragile "
            "framework documents and python coding capabilities.")
        )

        self.search_tool = configure_search_tool(project_root)
        self.ask_claude_tool = configure_claude_tool(claude_system_prompt)
        self.cot_reviewer = dspy.ChainOfThought(
            dspy.Signature(
                "proof_sketch_json: str, instructions: str -> review_json: str"
            ).with_instructions(
                "Read the proof sketch JSON and reviewer instructions. Produce findings as JSON."
            )
        )

        instructions = """
You are a senior reviewer conducting a structured audit of a proof sketch.

Workflow:
1. Use search_project(query) to retrieve relevant files or context.
2. Use ask_claude(prompt)  for exploring searching the framework documents or running calculations.
3. Use CoT reviewer (run_cot_review tool) to reason carefully for deep mathematical insights and summarizing findings.
4. Populate each review component:
   - Overall assessment (score 1-5, summary, recommendation)
   - Logical flow validation
   - Dependency validation
   - Technical deep dive critiques
   - Completeness & correctness
5. Assemble final SketchValidationReview JSON.

Return data strictly matching sketch_validation_request.json schema.
"""

        self.signature = SketchValidationReviewSignature.with_instructions(instructions)
        self.tools = [
            self.search_tool,
            self.ask_claude_tool,
            self._run_cot_review_tool,
        ]
        self.agent = dspy.ReAct(
            signature=self.signature,
            tools=self.tools,
            max_iters=5,
        )

    def _run_cot_review_tool(self, proof_sketch_json: str, instructions: str) -> str:
        """Chain-of-thought helper for reflective analysis."""
        prediction = self.cot_reviewer(
            proof_sketch_json=proof_sketch_json,
            instructions=instructions,
        )
        return getattr(prediction, "review_json", str(prediction))

    def forward(
        self,
        reviewer: str,
        timestamp: str,
        overallAssessment: str,
        logicalFlowValidation: str,
        dependencyValidation: str,
        technicalDeepDiveValidation: str,
        completenessAndCorrectness: str,
    ) -> dspy.Prediction:
        return self.agent(
            reviewer=reviewer,
            timestamp=timestamp,
            overallAssessment=overallAssessment,
            logicalFlowValidation=logicalFlowValidation,
            dependencyValidation=dependencyValidation,
            technicalDeepDiveValidation=technicalDeepDiveValidation,
            completenessAndCorrectness=completenessAndCorrectness,
        )
