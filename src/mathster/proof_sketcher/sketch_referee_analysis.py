#!/usr/bin/env python3
"""DSPy signatures, pydantic models, and review agents for proof sketch validation."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal

import dspy


logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field


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
    """Helper wrapper that configures a dspy.Predict program with shared instructions.

    Args:
        instructions: Optional instructions to override default
        predict_kwargs: Additional kwargs to pass to agent_cls
        lm: Optional dspy.LM instance to use for this agent. If None, uses dspy.settings.lm.
    """

    data_model: type[BaseModel]
    signature_cls: type[dspy.Signature]
    agent_cls: type[Any] = dspy.Predict
    instructions: str = "{schema}"
    predict_kwargs: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        instructions: str | None = None,
        predict_kwargs: dict[str, Any] | None = None,
        lm: dspy.LM | None = None,
    ) -> None:
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.lm = lm or self.get_default_lm()

        schema = json.dumps(self.data_model.model_json_schema(), indent=2)
        resolved_instructions = (instructions or self.instructions or "{schema}").format(
            schema=schema
        )
        self.signature = self.signature_cls.with_instructions(resolved_instructions)
        self.predict_kwargs = self.predict_kwargs or predict_kwargs or {}

        # Pass lm to agent_cls if provided
        if self.lm:
            self.predict_kwargs["lm"] = self.lm

        self.agent = self.agent_cls(self.signature, **self.predict_kwargs)

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        return self.agent(**kwargs)

    def get_default_lm(self) -> dspy.LM:
        """Return a new ."""
        return dspy.settings.lm


# ---------------------------------
# Completeness and Correctness Components
# ---------------------------------
class IdentifiedError(BaseModel):
    """A concrete mathematical error or mismatch in the proof sketch."""

    location: str = Field(..., description="Reference to the specific step or section. Show here **what** is wrong")
    description: str = Field(..., description="Details of the error or inconsistency. Explain **why** it is wrong.")
    suggested_fix: str = Field(..., description="Proposed fix or remedy to correct the error. Offer **how** to fix it.")


class CompletenessCorrectnessReview(BaseModel):
    """Assessment of coverage and mathematical correctness."""

    covers_all_claims: bool = Field(..., description="Does the sketch cover every claim?")
    identified_errors: list[IdentifiedError] = Field(
        default_factory=list,
        description="List of explicit mathematical errors or typos found.",
    )
    score: int = Field(
        ...,
        description="Numeric quality score from 1-5. 1 means unusable, 5 means publication-quality.",
    )
    confidence: int = Field(
        ...,
        description="Numeric confidence score from 1-5 indicating reviewer's confidence in the assessment.",
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


class AgentCompletenessCorrectness(BaseAgent):
    """Predictor that scores completeness/correctness coverage for a proof sketch."""

    data_model = CompletenessCorrectnessReview
    signature_cls = CompletenessCorrectnessSignature
    instructions = """
Audit the proof sketch for coverage and correctness.

Checklist:
1. Expand theorem_statement into explicit hypotheses and conclusions.
2. Map each claim to supporting passages inside proof_sketch_text.
3. Record every flaw as an IdentifiedError with location, problem statement, and fix.
4. coversAllClaims is True only when every required component is justified.
5. Score ∈ [1,5] reflects mathematical rigor (5=publication-ready, 1=unusable).
6. confidence ∈ [1,5] mirrors certainty in the assessment.

Return ONLY JSON matching:
{schema}
"""

# ----------------------------------
# Logical Flow Validation Components
# ----------------------------------
class IdentifiedGap(BaseModel):
    """A specific logical gap or leap of faith in the proof sketch."""

    location: str = Field(..., description="Reference to the specific step or section with the gap. Explane **where** the gap occurs.")
    description: str = Field(..., description="Details of the logical gap or leap of faith. Explain **what** the gap is.")
    proposed_resolution: str = Field(..., description="Suggested way to address or fill the gap. Offer **how** to resolve it.")


class LogicalFlowReview(BaseModel):
    """Assessment of logical structure and step progression."""

    is_sound: bool = Field(..., description="True if the argument is coherent and valid.")
    comments: str = Field(..., description="Narrative assessment of clarity and structure.")
    identified_gaps: list[IdentifiedGap] = Field(
        default_factory=list,
        description="Specific logical gaps or leaps of faith that require attention.",
    )
    score: int = Field(
        ...,
        description="Numeric quality score from 1-5. 1 means unusable, 5 means publication-quality.",
    )
    confidence: int = Field(
        ...,
        description="Numeric confidence score from 1-5 indicating reviewer's confidence in the assessment.",
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
    """Agent using a dspy.ChainOfThought to evaluate the logical flow of a proof sketch."""

    data_model = LogicalFlowReview
    signature_cls = LogicalFlowValidationSignature
    instructions = """
Audit the proof sketch for logical flow and coherence.

Checklist:
1. Review the proof outline for clear step-by-step progression.
2. Identify any gaps or leaps in reasoning that need to be addressed.
3. Provide a score from 1-5 reflecting the overall logical quality.
4. Include confidence level in the assessment (1-5).

Return ONLY JSON matching:
{schema}
"""


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
    score: int = Field(
        ...,
        description="Numeric quality score from 1-5. 1 means unusable, 5 means publication-quality.",
    )
    confidence: int = Field(
        ...,
        description="Numeric confidence score from 1-5 indicating reviewer's confidence in the assessment.",
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


class AgentDependencyValidation(BaseAgent):
    """Agent using a dspy.ChainOfThought to validate framework dependencies in a proof sketch implementing DependencyValidationSignature."""

    data_model = DependencyValidationReview
    signature_cls = DependencyValidationSignature
    instructions = """
Audit the proof sketch for correct usage of framework dependencies.

Checklist:
1. Verify all framework references are accurate and properly cited.
2. Identify any missing or incorrectly used dependencies.
3. Provide a score from 1-5 reflecting the overall dependency quality.
4. Include confidence level in the assessment (1-5).

Return ONLY JSON matching:
{schema}
"""


# ---------------------------------
# Technical Deep Dive Components
# ---------------------------------
class TechnicalDeepDiveCritique(BaseModel):
    """Review of a single technical challenge and proposed fix."""

    challengeTitle: str = Field(..., description="Matching title from the original sketch.")
    solution_viability: SolutionViability = Field(..., description="Assessment of feasibility.")
    critique: str = Field(
        ..., description="Detailed analysis highlighting strengths and weaknesses."
    )
    suggested_improvements: str | None = Field(
        default=None,
        description="Optional concrete suggestions for strengthening the argument.",
    )


class TechnicalDeepDiveValidation(BaseModel):
    """Collection of critiques for technical deep dives."""

    critiques: list[TechnicalDeepDiveCritique] = Field(
        default_factory=list,
        description="One entry per challenging component in the sketch.",
    )
    score: int = Field(
        ...,
        description="Numeric quality score from 1-5. 1 means unusable, 5 means ready.",
    )
    confidence: int = Field(
        ...,
        description="Numeric confidence score from 1-5 indicating reviewer's confidence in the assessment.",
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


class AgentTechnicalDeepDiveValidation(BaseAgent):
    """Agent using a dspy.ChainOfThought to critique the technical deep dives of a proof sketch."""

    data_model = TechnicalDeepDiveValidation
    signature_cls = TechnicalDeepDiveValidationSignature
    instructions = """
Audit the technical deep dives in the proof sketch.
Checklist:
1. For each technical challenge, evaluate the proposed solution's viability.
2. Provide a detailed critique highlighting strengths and weaknesses.
3. Suggest concrete improvements where applicable.
4. Provide a score from 1-5 reflecting the overall technical quality.
5. Include confidence level in the assessment (1-5).

Return ONLY JSON matching:
{schema}
"""


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
        description="Numeric quality score from 1-5. 1 means unusable, 5 means publication-quality.",
    )
    confidence: int = Field(
        ...,
        description="Numeric confidence score from 1-5 indicating reviewer's confidence in the assessment.",
    )


class OverallAssessmentSignature(dspy.Signature):
    """Generate the high-level assessment summary."""

    extra_instructions = dspy.InputField(desc="Additional instructions for the reviewer.")
    proof_sketch_json = dspy.InputField(desc="JSON representation of the proof sketch.")
    logicalFlowValidation: LogicalFlowReview = dspy.InputField(desc="Logic and structure review.")
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
    """Agent using a dspy.ChainOfThought to generate the overall assessment of a proof sketch."""

    data_model = OverallAssessment
    signature_cls = OverallAssessmentSignature
    instructions = """
Synthesize a high-level assessment of the proof sketch.
Checklist:
1. Integrate findings from all detailed reviews.
2. Summarize core strengths and weaknesses concisely.
3. Provide a clear, actionable recommendation for next steps.
4. Assign a confidenceScore from 1-5 reflecting overall quality.
5. Provide a score from 1-5 reflecting the overall quality.
6. Include confidence level in the assessment (1-5).

Return ONLY JSON matching:
{schema}
"""


# ---------------------------------
# Sketsch Validation Review Components
# ---------------------------------
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


class SketchRefereeAgent(dspy.Module):
    """This agent calls all the other agents in order and handles the "glue code" to produce a full sketch validation review. as described by SketchValidationReview.
    It is a composition of modules taht implements no new module on its own as it only does orchestration.

    Args:
        lm: Optional dspy.LM instance to use for all sub-agents. If None, uses dspy.settings.lm.
        synthesis_lm: Optional dspy.LM instance specifically for the overall assessment synthesis agent.
                      If None, uses the same lm as other agents.
    """

    def __init__(
        self,
        lm: dspy.LM | None = None,
        synthesis_lm: dspy.LM | None = None,
    ) -> None:
        """Initialize all component agents for orchestrated proof sketch review.

        Args:
            lm: Language model for the 4 parallel review components
            synthesis_lm: Optional separate model for synthesis (overall assessment)
        """
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.lm = lm
        self.synthesis_lm = synthesis_lm or lm

        # Initialize 4 parallel review components with main lm
        self.completeness_agent = AgentCompletenessCorrectness(lm=self.lm)
        self.logical_flow_agent = AgentLogicalFlowValidation(lm=self.lm)
        self.dependency_agent = AgentDependencyValidation(lm=self.lm)
        self.technical_dive_agent = AgentTechnicalDeepDiveValidation(lm=self.lm)

        # Initialize synthesis agent with potentially different model
        self.overall_agent = AgentOverallAssessment(lm=self.synthesis_lm)

    def old_forward(
        self,
        proof_sketch_dict: dict[str, Any],
        reviewer: str,
        extra_instructions: str = "",
        reasoning_notes: str | None = None,
        sketch_references: str | None = None,
        reviewer_focus: str | None = None,
    ) -> dspy.Prediction:
        """Orchestrate all review agents to produce a complete sketch validation review.

        Args:
            proof_sketch_dict: Dictionary representation of a ProofSketch instance.
            reviewer: Reviewer identity (model, version, etc.).
            extra_instructions: Additional instructions for the overall assessment.
            reasoning_notes: Optional notes about logical flow (auto-extracted if None).
            sketch_references: Optional excerpt of sketch referencing dependencies.
            reviewer_focus: Optional notes about what to scrutinize in technical dives.

        Returns:
            dspy.Prediction containing the complete SketchValidationReview.
        """
        import datetime

        self.logger.info("SketchRefereeAgent starting review for reviewer=%s", reviewer)
        overall_start = time.perf_counter()

        # Extract fields from proof_sketch_dict
        theorem_statement = proof_sketch_dict.get("statement", {}).get("formal", "")

        # Concatenate all proof steps into a single text
        detailed_proof = proof_sketch_dict.get("detailedProof", {})
        steps = detailed_proof.get("steps", [])
        proof_sketch_text = "\n\n".join([
            f"Step {step.get('stepNumber', '?')}: {step.get('title', 'Untitled')}\n"
            f"Goal: {step.get('goal', '')}\n"
            f"Action: {step.get('action', '')}\n"
            f"Justification: {step.get('justification', '')}\n"
            f"Expected Result: {step.get('expectedResult', '')}"
            for step in steps
        ])

        # Join top-level outline into a single string
        top_level_outline = detailed_proof.get("topLevelOutline", [])
        proof_outline = "\n".join([f"{i + 1}. {item}" for i, item in enumerate(top_level_outline)])

        # Convert dependencies to JSON
        dependencies = proof_sketch_dict.get("dependencies", {})
        dependency_ledger_json = json.dumps(dependencies, indent=2)

        # Convert technical deep dives to JSON
        technical_deep_dives = proof_sketch_dict.get("technicalDeepDives", [])
        deep_dives_json = json.dumps(technical_deep_dives, indent=2)

        # Convert entire sketch to JSON
        proof_sketch_json = json.dumps(proof_sketch_dict, indent=2)

        # Auto-extract reasoning notes if not provided
        if reasoning_notes is None:
            overview = detailed_proof.get("overview", "")
            reasoning_notes = f"Proof overview: {overview}" if overview else None

        # 1. Completeness and Correctness Review
        self.logger.debug("Running completeness & correctness review")
        start = time.perf_counter()
        completeness_result = self.completeness_agent(
            theorem_statement=theorem_statement,
            proof_sketch_text=proof_sketch_text,
        )
        completeness_review = completeness_result.completenessAndCorrectness
        self.logger.debug(
            "Completeness review completed in %.2fs | Score: %d/5 | Errors: %d",
            time.perf_counter() - start,
            completeness_review.score,
            len(completeness_review.identifiedErrors),
        )

        # 2. Logical Flow Validation
        self.logger.debug("Running logical flow validation")
        start = time.perf_counter()
        logical_flow_result = self.logical_flow_agent(
            proof_outline=proof_outline,
            reasoning_notes=reasoning_notes,
        )
        logical_flow_review = logical_flow_result.logicalFlowValidation
        self.logger.debug(
            "Logical flow review completed in %.2fs | Score: %d/5 | Sound: %s | Gaps: %d",
            time.perf_counter() - start,
            logical_flow_review.score,
            logical_flow_review.isSound,
            len(logical_flow_review.identifiedGaps),
        )

        # 3. Dependency Validation
        self.logger.debug("Running dependency validation")
        start = time.perf_counter()
        dependency_result = self.dependency_agent(
            dependency_ledger_json=dependency_ledger_json,
            sketch_references=sketch_references,
        )
        dependency_review = dependency_result.dependencyValidation
        self.logger.debug(
            "Dependency review completed in %.2fs | Status: %s | Issues: %d",
            time.perf_counter() - start,
            dependency_review.status,
            len(dependency_review.issues),
        )

        # 4. Technical Deep Dive Validation
        self.logger.debug("Running technical deep dive validation")
        start = time.perf_counter()
        technical_dive_result = self.technical_dive_agent(
            deep_dives_json=deep_dives_json,
            reviewer_focus=reviewer_focus,
        )
        technical_dive_review = technical_dive_result.technicalDeepDiveValidation
        self.logger.debug(
            "Technical dive review completed in %.2fs | Critiques: %d",
            time.perf_counter() - start,
            len(technical_dive_review.critiques),
        )

        # 5. Overall Assessment (synthesizes all previous reviews)
        self.logger.debug("Generating overall assessment")
        start = time.perf_counter()
        overall_result = self.overall_agent(
            extra_instructions=extra_instructions,
            proof_sketch_json=proof_sketch_json,
            logicalFlowValidation=logical_flow_review,
            dependencyValidation=dependency_review,
            technicalDeepDiveValidation=technical_dive_review,
            completenessAndCorrectness=completeness_review,
        )
        overall_assessment = overall_result.overallAssessment
        self.logger.debug(
            "Overall assessment completed in %.2fs | Confidence: %d/5 | Recommendation: %s",
            time.perf_counter() - start,
            overall_assessment.confidenceScore,
            overall_assessment.recommendation,
        )

        # 6. Assemble final review with metadata
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        review = SketchValidationReview(
            reviewer=reviewer,
            timestamp=timestamp,
            overallAssessment=overall_assessment,
            logicalFlowValidation=logical_flow_review,
            dependencyValidation=dependency_review,
            technicalDeepDiveValidation=technical_dive_review,
            completenessAndCorrectness=completeness_review,
        )

        overall_elapsed = time.perf_counter() - overall_start
        self.logger.info(
            "SketchRefereeAgent completed in %.2fs | Reviewer: %s | Overall score: %d/5 | Recommendation: %s",
            overall_elapsed,
            reviewer,
            overall_assessment.score,
            overall_assessment.recommendation,
        )

        return dspy.Prediction(review=review)

    def forward(
        self,
        proof_sketch_dict: dict[str, Any],
        reviewer: str,
        extra_instructions: str = "",
        reasoning_notes: str | None = None,
        sketch_references: str | None = None,
        reviewer_focus: str | None = None,
    ) -> dspy.Prediction:
        # Extract data
        overall_start = time.perf_counter()

        # Extract fields from proof_sketch_dict
        theorem_statement = proof_sketch_dict.get("statement", {}).get("formal", "")

        # Concatenate all proof steps into a single text
        detailed_proof = proof_sketch_dict.get("detailedProof", {})
        steps = detailed_proof.get("steps", [])
        proof_sketch_text = "\n\n".join([
            f"Step {step.get('stepNumber', '?')}: {step.get('title', 'Untitled')}\n"
            f"Goal: {step.get('goal', '')}\n"
            f"Action: {step.get('action', '')}\n"
            f"Justification: {step.get('justification', '')}\n"
            f"Expected Result: {step.get('expectedResult', '')}"
            for step in steps
        ])

        # Join top-level outline into a single string
        top_level_outline = detailed_proof.get("topLevelOutline", [])
        proof_outline = "\n".join([f"{i + 1}. {item}" for i, item in enumerate(top_level_outline)])

        # Convert dependencies to JSON
        dependencies = proof_sketch_dict.get("dependencies", {})
        dependency_ledger_json = json.dumps(dependencies, indent=2)

        # Convert technical deep dives to JSON
        technical_deep_dives = proof_sketch_dict.get("technicalDeepDives", [])
        deep_dives_json = json.dumps(technical_deep_dives, indent=2)

        # Convert entire sketch to JSON
        proof_sketch_json = json.dumps(proof_sketch_dict, indent=2)

        # Auto-extract reasoning notes if not provided
        if reasoning_notes is None:
            overview = detailed_proof.get("overview", "")
            reasoning_notes = f"Proof overview: {overview}" if overview else None

        # Create dspy.Example objects for parallel execution
        self.logger.info("SketchRefereeAgent starting parallel review for reviewer=%s", reviewer)

        completeness_example = dspy.Example(
            theorem_statement=theorem_statement,
            proof_sketch_text=proof_sketch_text,
        ).with_inputs("theorem_statement", "proof_sketch_text")

        logical_flow_example = dspy.Example(
            proof_outline=proof_outline,
            reasoning_notes=reasoning_notes,
        ).with_inputs("proof_outline", "reasoning_notes")

        dependency_example = dspy.Example(
            dependency_ledger_json=dependency_ledger_json,
            sketch_references=sketch_references,
        ).with_inputs("dependency_ledger_json", "sketch_references")

        technical_dive_example = dspy.Example(
            deep_dives_json=deep_dives_json,
            reviewer_focus=reviewer_focus,
        ).with_inputs("deep_dives_json", "reviewer_focus")

        # Create execution pairs for parallel processing
        exec_pairs = [
            (self.completeness_agent, completeness_example),
            (self.logical_flow_agent, logical_flow_example),
            (self.dependency_agent, dependency_example),
            (self.technical_dive_agent, technical_dive_example),
        ]

        # Execute all 4 agents in parallel
        self.logger.debug(
            "Running 4 review agents in parallel (completeness, logical flow, dependency, technical)"
        )
        parallel = dspy.Parallel(num_threads=4)
        parallel_start = time.perf_counter()
        results = parallel(exec_pairs)
        parallel_elapsed = time.perf_counter() - parallel_start

        # Extract results from parallel execution
        completeness_result = results[0]
        logical_flow_result = results[1]
        dependency_result = results[2]
        technical_dive_result = results[3]

        # Extract review objects from predictions
        completeness_review = completeness_result.completenessAndCorrectness
        logical_flow_review = logical_flow_result.logicalFlowValidation
        dependency_review = dependency_result.dependencyValidation
        technical_dive_review = technical_dive_result.technicalDeepDiveValidation

        # Log aggregated parallel results
        self.logger.debug(
            "Parallel review completed in %.2fs | Completeness: %d/5 (%d errors) | Flow: %d/5 (sound=%s, %d gaps) | Deps: %s (%d issues) | Tech: %d critiques",
            parallel_elapsed,
            completeness_review.score,
            len(completeness_review.identifiedErrors),
            logical_flow_review.score,
            logical_flow_review.isSound,
            len(logical_flow_review.identifiedGaps),
            dependency_review.status,
            len(dependency_review.issues),
            len(technical_dive_review.critiques),
        )

        # 5. Overall Assessment (synthesizes all previous reviews)
        self.logger.debug("Generating overall assessment")
        start = time.perf_counter()
        overall_result = self.overall_agent(
            extra_instructions=extra_instructions,
            proof_sketch_json=proof_sketch_json,
            logicalFlowValidation=logical_flow_review,
            dependencyValidation=dependency_review,
            technicalDeepDiveValidation=technical_dive_review,
            completenessAndCorrectness=completeness_review,
        )
        overall_assessment = overall_result.overallAssessment
        self.logger.debug(
            "Overall assessment completed in %.2fs | Confidence: %d/5 | Recommendation: %s",
            time.perf_counter() - start,
            overall_assessment.confidenceScore,
            overall_assessment.recommendation,
        )

        # 6. Assemble final review with metadata
        import datetime

        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        review = SketchValidationReview(
            reviewer=reviewer,
            timestamp=timestamp,
            overallAssessment=overall_assessment,
            logicalFlowValidation=logical_flow_review,
            dependencyValidation=dependency_review,
            technicalDeepDiveValidation=technical_dive_review,
            completenessAndCorrectness=completeness_review,
        )

        overall_elapsed = time.perf_counter() - overall_start
        self.logger.info(
            "SketchRefereeAgent parallel review completed in %.2fs | Reviewer: %s | Overall score: %d/5 | Recommendation: %s",
            overall_elapsed,
            reviewer,
            overall_assessment.score,
            overall_assessment.recommendation,
        )

        return dspy.Prediction(review=review)
