#!/usr/bin/env python3
"""Signatures and models for full proof sketch validation cycle reports."""

from __future__ import annotations

import json
import logging
import time
from typing import Literal

import dspy


logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field

from mathster.proof_sketcher.sketch_referee_analysis import SketchRefereeAgent


__all__ = [
    "ActionItem",
    "ActionItemPriority",
    "ActionableItemsSignature",
    "ConsensusAnalysis",
    "ConsensusAnalysisSignature",
    "DisagreementEntry",
    "ReportMetadata",
    "ReportMetadataSignature",
    "SketchValidationReport",
    "SketchValidationReportSignature",
    "SketchValidator",
    "SynthesisAndActionPlan",
    "SynthesisAndActionPlanSignature",
]


ActionItemPriority = Literal["Critical", "High", "Medium", "Low"]
FinalDecision = Literal[
    "Approved for Expansion",
    "Requires Minor Revisions",
    "Requires Major Revisions",
    "Rejected - New Strategy Needed",
]


class ReportMetadata(BaseModel):
    """Metadata about the validation cycle."""

    sketchLabel: str = Field(..., description="Label of the proof sketch under review.")
    validationCycleId: str = Field(
        ..., description="UUID identifying this validation cycle.", pattern=r"^[0-9a-fA-F-]{36}$"
    )
    validationTimestamp: str = Field(
        ...,
        description="ISO 8601 timestamp marking completion of the validation cycle.",
    )


class DisagreementEntry(BaseModel):
    """Comparison of divergent reviewer feedback."""

    topic: str = Field(..., description="Issue/topic under discussion.")
    geminiView: str = Field(..., description="View expressed by Gemini reviewer.")
    codexView: str = Field(..., description="View expressed by Codex reviewer.")
    resolution: str = Field(
        ..., description="Summary of how disagreements should be resolved or investigated."
    )


class ConsensusAnalysis(BaseModel):
    """Aggregated analysis of reviewer agreements and disagreements."""

    pointsOfAgreement: list[str] = Field(
        default_factory=list,
        description="Issues or strengths where reviewers independently agree.",
    )
    pointsOfDisagreement: list[DisagreementEntry] = Field(
        default_factory=list,
        description="Conflicting viewpoints with proposed resolutions.",
    )
    summaryOfFindings: str = Field(
        ..., description="Narrative synthesize of reviewer feedback and rationale for decisions."
    )


class ActionItem(BaseModel):
    """Concrete tasks needed before the sketch can progress."""

    itemId: str = Field(..., description="Unique identifier for the action item.")
    description: str = Field(..., description="Detailed description of the revision/task.")
    priority: ActionItemPriority = Field(..., description="Priority classification.")
    references: list[str] = Field(
        default_factory=list,
        description="Pointers to sketch/review segments related to this item.",
    )


class SynthesisAndActionPlan(BaseModel):
    """Final decision plus tasks required for follow-up."""

    finalDecision: FinalDecision = Field(
        ...,
        description="Consolidated status for the sketch after considering all reviews.",
    )
    consensusAnalysis: ConsensusAnalysis = Field(..., description="Reviewer agreement analysis.")
    actionableItems: list[ActionItem] = Field(
        default_factory=list,
        description="List of actionable tasks required before next stage.",
    )
    confidenceStatement: str = Field(
        ..., description="Forward-looking statement about viability after action items."
    )


class SketchValidationReport(BaseModel):
    """Top-level proof sketch validation cycle report."""

    reportMetadata: ReportMetadata = Field(..., description="Metadata describing the cycle.")
    originalProofSketch: dict = Field(
        ...,
        description="Proof sketch JSON that conforms to the proof sketch schema.",
    )
    reviews: list[dict] = Field(
        ...,
        description="List of validation reviews (each conforming to sketch_validation_request schema).",
    )
    synthesisAndActionPlan: SynthesisAndActionPlan = Field(
        ..., description="Final decision, consensus analysis, and tasks."
    )


class ReportMetadataSignature(dspy.Signature):
    """Generate report metadata for a validation cycle."""

    sketch_label = dspy.InputField(desc="Label of the proof sketch under validation.")
    cycle_uuid = dspy.InputField(desc="UUID for this validation cycle.")
    timestamp = dspy.InputField(desc="ISO 8601 timestamp for this report.")

    reportMetadata: ReportMetadata = dspy.OutputField(
        desc="Structured metadata matching sketch_validation.json schema."
    )


class ConsensusAnalysisSignature(dspy.Signature):
    """Summarize reviewer agreement and disagreements."""

    reviewer_notes = dspy.InputField(
        desc="Combined notes referencing key points of agreement/disagreement."
    )
    gemini_review = dspy.InputField(desc="Serialized review JSON from Gemini reviewer.")
    codex_review = dspy.InputField(desc="Serialized review JSON from Codex reviewer.")

    consensusAnalysis: ConsensusAnalysis = dspy.OutputField(
        desc="Consensus analysis summary object."
    )


class ActionableItemsSignature(dspy.Signature):
    """Generate the list of action items for the action plan."""

    consolidated_feedback = dspy.InputField(desc="Text summarizing required revisions or tasks.")
    references_json = dspy.InputField(
        desc="JSON array mapping action items to review references.", optional=True
    )

    actionableItems: list[ActionItem] = dspy.OutputField(
        desc="List of action items with priority and references."
    )


class SynthesisAndActionPlanSignature(dspy.Signature):
    """Produce final decision, consensus, actionables, and confidence statement."""

    consensus_analysis_json = dspy.InputField(
        desc="JSON string representing the consensusAnalysis block."
    )
    actionable_items_json = dspy.InputField(
        desc="JSON array string for action items.", optional=True
    )
    final_decision_context = dspy.InputField(
        desc="Description of the factors influencing the final decision."
    )
    confidence_statement_context = dspy.InputField(
        desc="Narrative guidance for the confidence statement."
    )

    synthesisAndActionPlan: SynthesisAndActionPlan = dspy.OutputField(
        desc="Structured synthesis/action plan conforming to schema."
    )


class ActionableItemsListSignature(dspy.Signature):
    """Signature alias for compatibility."""

    consolidated_feedback = dspy.InputField(desc="Same as ActionableItemsSignature.")
    references_json = dspy.InputField(desc="Same as ActionableItemsSignature.", optional=True)

    items: list[ActionItem] = dspy.OutputField(desc="Action items list.")


class SketchValidationReportSignature(dspy.Signature):
    """Assemble the full sketch validation report."""

    reportMetadata = dspy.InputField(desc="Serialized ReportMetadata JSON.")
    originalProofSketch = dspy.InputField(desc="Serialized proof sketch JSON.")
    reviews = dspy.InputField(desc="JSON array of two review objects.")
    synthesisAndActionPlan = dspy.InputField(desc="Serialized SynthesisAndActionPlan JSON.")

    report: SketchValidationReport = dspy.OutputField(
        desc="Complete validation cycles report object."
    )


class Scores(BaseModel):
    """Comprehensive numerical metrics extracted from dual review validation cycle.

    Tracks all score-related data from Gemini and Codex reviews, aggregate statistics,
    synthesis metrics, and derived quality indicators. All scores follow 1-5 scale:
    - 1 = Unusable
    - 2 = Critical issues but fixable
    - 3 = Major revisions needed
    - 4 = Minor revisions needed
    - 5 = Publication-quality
    """

    # === GEMINI REVIEW SCORES ===
    gemini_overall_score: int = Field(..., description="Gemini's overall quality score (1-5)")
    gemini_overall_confidence: int = Field(..., description="Gemini's overall confidence (1-5)")
    gemini_overall_confidence_score: int = Field(
        ..., description="Gemini's OverallAssessment.confidenceScore field (1-5)"
    )

    gemini_completeness_score: int = Field(
        ..., description="Gemini's completeness/correctness quality score (1-5)"
    )
    gemini_completeness_confidence: int = Field(
        ..., description="Gemini's completeness confidence (1-5)"
    )
    gemini_covers_all_claims: bool = Field(
        ..., description="Whether Gemini confirms all theorem claims are covered"
    )
    gemini_error_count: int = Field(
        ..., description="Number of mathematical errors identified by Gemini"
    )

    gemini_logical_flow_score: int = Field(
        ..., description="Gemini's logical flow quality score (1-5)"
    )
    gemini_logical_flow_confidence: int = Field(
        ..., description="Gemini's logical flow confidence (1-5)"
    )
    gemini_logical_flow_sound: bool = Field(
        ..., description="Whether Gemini deems the logical flow sound"
    )
    gemini_logical_gap_count: int = Field(
        ..., description="Number of logical gaps identified by Gemini"
    )

    gemini_dependency_score: int | None = Field(
        None, description="Gemini's dependency validation score (1-5) if available"
    )
    gemini_dependency_confidence: int | None = Field(
        None, description="Gemini's dependency validation confidence (1-5) if available"
    )
    gemini_dependency_issue_count: int = Field(
        ..., description="Number of dependency issues identified by Gemini"
    )

    gemini_technical_dive_score: int | None = Field(
        None, description="Gemini's technical deep dive score (1-5) if available"
    )
    gemini_technical_dive_confidence: int | None = Field(
        None, description="Gemini's technical deep dive confidence (1-5) if available"
    )
    gemini_technical_critique_count: int = Field(
        ..., description="Number of technical critiques provided by Gemini"
    )

    # === CODEX REVIEW SCORES ===
    codex_overall_score: int = Field(..., description="Codex's overall quality score (1-5)")
    codex_overall_confidence: int = Field(..., description="Codex's overall confidence (1-5)")
    codex_overall_confidence_score: int = Field(
        ..., description="Codex's OverallAssessment.confidenceScore field (1-5)"
    )

    codex_completeness_score: int = Field(
        ..., description="Codex's completeness/correctness quality score (1-5)"
    )
    codex_completeness_confidence: int = Field(
        ..., description="Codex's completeness confidence (1-5)"
    )
    codex_covers_all_claims: bool = Field(
        ..., description="Whether Codex confirms all theorem claims are covered"
    )
    codex_error_count: int = Field(
        ..., description="Number of mathematical errors identified by Codex"
    )

    codex_logical_flow_score: int = Field(
        ..., description="Codex's logical flow quality score (1-5)"
    )
    codex_logical_flow_confidence: int = Field(
        ..., description="Codex's logical flow confidence (1-5)"
    )
    codex_logical_flow_sound: bool = Field(
        ..., description="Whether Codex deems the logical flow sound"
    )
    codex_logical_gap_count: int = Field(
        ..., description="Number of logical gaps identified by Codex"
    )

    codex_dependency_score: int | None = Field(
        None, description="Codex's dependency validation score (1-5) if available"
    )
    codex_dependency_confidence: int | None = Field(
        None, description="Codex's dependency validation confidence (1-5) if available"
    )
    codex_dependency_issue_count: int = Field(
        ..., description="Number of dependency issues identified by Codex"
    )

    codex_technical_dive_score: int | None = Field(
        None, description="Codex's technical deep dive score (1-5) if available"
    )
    codex_technical_dive_confidence: int | None = Field(
        None, description="Codex's technical deep dive confidence (1-5) if available"
    )
    codex_technical_critique_count: int = Field(
        ..., description="Number of technical critiques provided by Codex"
    )

    # === AGGREGATE SCORES (across both reviewers) ===
    average_overall_score: float = Field(
        ..., description="Mean of Gemini and Codex overall scores"
    )
    average_overall_confidence: float = Field(
        ..., description="Mean of Gemini and Codex overall confidences"
    )
    average_completeness_score: float = Field(
        ..., description="Mean of Gemini and Codex completeness scores"
    )
    average_logical_flow_score: float = Field(
        ..., description="Mean of Gemini and Codex logical flow scores"
    )

    total_error_count: int = Field(
        ..., description="Total mathematical errors identified by both reviewers"
    )
    total_logical_gap_count: int = Field(
        ..., description="Total logical gaps identified by both reviewers"
    )
    total_dependency_issue_count: int = Field(
        ..., description="Total dependency issues identified by both reviewers"
    )
    total_technical_critique_count: int = Field(
        ..., description="Total technical critiques from both reviewers"
    )

    both_reviewers_sound: bool = Field(
        ..., description="True if both Gemini and Codex deem logical flow sound"
    )
    both_reviewers_cover_claims: bool = Field(
        ..., description="True if both Gemini and Codex confirm all claims covered"
    )

    overall_score_variance: float = Field(
        ..., description="Squared difference between Gemini and Codex overall scores"
    )
    completeness_score_variance: float = Field(
        ..., description="Squared difference between Gemini and Codex completeness scores"
    )
    logical_flow_score_variance: float = Field(
        ..., description="Squared difference between Gemini and Codex logical flow scores"
    )

    # === SYNTHESIS-LEVEL METRICS ===
    final_decision_numeric: int = Field(
        ...,
        description="Numeric mapping of finalDecision (1=Rejected, 2=Major, 3=Minor, 4=Approved)",
    )

    action_item_count: int = Field(..., description="Total number of action items")
    critical_action_count: int = Field(..., description="Number of critical priority actions")
    high_action_count: int = Field(..., description="Number of high priority actions")
    medium_action_count: int = Field(..., description="Number of medium priority actions")
    low_action_count: int = Field(..., description="Number of low priority actions")

    points_of_agreement_count: int = Field(
        ..., description="Number of issues where reviewers independently agree"
    )
    points_of_disagreement_count: int = Field(
        ..., description="Number of conflicting viewpoints between reviewers"
    )

    # === DERIVED METRICS ===
    overall_quality_index: float = Field(
        ..., description="Composite quality index combining scores, weighted by confidence"
    )
    risk_score: float = Field(
        ..., description="Risk metric based on error counts, gaps, and score variance"
    )


class SketchValidator(dspy.Module):
    """Run dual sketch reviews and synthesize a complete validation report."""

    def __init__(
        self,
        project_root: str | None = None,
        gemini_prompt: str | None = None,
        codex_prompt: str | None = None,
    ) -> None:
        super().__init__()
        project_root = project_root or "."
        gemini_prompt = gemini_prompt or (
            "You are Gemini 2.5 Flash serving as a rigorous mathematical proof reviewer. "
            "Provide meticulous, publication-ready analysis."
        )
        codex_prompt = codex_prompt or (
            "You are Codex (GPT-5) acting as an adversarial proof reviewer. "
            "Focus on technical rigor and computational correctness."
        )

        self.metadata_generator = dspy.Predict(ReportMetadataSignature)
        self.consensus_generator = dspy.ChainOfThought(
            ConsensusAnalysisSignature.with_instructions(
                "Compare Gemini and Codex reviews to extract agreements, disagreements, "
                "and summarize key findings."
            )
        )
        self.action_items_generator = dspy.ChainOfThought(
            ActionableItemsSignature.with_instructions(
                "Turn consolidated reviewer feedback into actionable TODO items with priorities."
            )
        )
        self.synthesis_generator = dspy.ChainOfThought(
            SynthesisAndActionPlanSignature.with_instructions(
                "Merge consensus analysis and action items into a final decision and confidence statement."
            )
        )
        self.report_builder = dspy.Predict(SketchValidationReportSignature)

        self.review_agent_gemini = SketchRefereeAgent()
        self.review_agent_codex = SketchRefereeAgent()

    def _run_review(
        self,
        agent: SketchRefereeAgent,
        reviewer_name: str,
        proof_sketch_dict: dict,
        extra_instructions: str,
    ) -> dict:
        """Invoke a SketchRefereeAgent with shared proof sketch context."""
        logger.info("Running review with %s", reviewer_name)
        start_time = time.perf_counter()

        prediction = agent(
            proof_sketch_dict=proof_sketch_dict,
            reviewer=reviewer_name,
            extra_instructions=extra_instructions,
        )

        elapsed = time.perf_counter() - start_time
        logger.info("Review with %s completed in %.2fs", reviewer_name, elapsed)

        review = prediction.review
        return review.model_dump() if hasattr(review, "model_dump") else review

    def forward(
        self,
        proof_sketch: dict,
        sketch_label: str,
        validation_cycle_id: str,
        validation_timestamp: str,
        reviewer_context: str = "",
        final_decision_context: str = "",
        confidence_context: str = "",
    ) -> dspy.Prediction:
        logger.info(
            "Starting validation cycle for sketch %s (cycle_id=%s)",
            sketch_label,
            validation_cycle_id,
        )
        overall_start = time.perf_counter()

        reviewer_context = (
            reviewer_context or "Perform a thorough dual-review of the provided proof sketch."
        )
        logger.debug("Generating report metadata")
        metadata = self.metadata_generator(
            sketch_label=sketch_label,
            cycle_uuid=validation_cycle_id,
            timestamp=validation_timestamp,
        ).reportMetadata

        proof_sketch_json = json.dumps(proof_sketch)
        gemini_review = self._run_review(
            self.review_agent_gemini,
            reviewer_name="Gemini 2.5 Flash",
            proof_sketch_dict=proof_sketch,
            extra_instructions=reviewer_context,
        )
        codex_review = self._run_review(
            self.review_agent_codex,
            reviewer_name="GPT-5 Codex",
            proof_sketch_dict=proof_sketch,
            extra_instructions=reviewer_context,
        )

        logger.debug("Analyzing consensus between reviewers")
        consensus_start = time.perf_counter()
        consensus = self.consensus_generator(
            reviewer_notes=reviewer_context,
            gemini_review=json.dumps(gemini_review),
            codex_review=json.dumps(codex_review),
        ).consensusAnalysis
        logger.debug(
            "Consensus analysis completed in %.2fs", time.perf_counter() - consensus_start
        )

        consolidated_feedback = (
            consensus.summaryOfFindings
            + "\nPoints of agreement:\n"
            + "\n".join(f"- {p}" for p in consensus.pointsOfAgreement)
        )
        logger.debug("Generating action items")
        action_items_prediction = self.action_items_generator(
            consolidated_feedback=consolidated_feedback,
            references_json=json.dumps({
                "gemini": gemini_review.get("reviewer", "Gemini 2.5 Flash"),
                "codex": codex_review.get("reviewer", "GPT-5 Codex"),
            }),
        )
        action_items = action_items_prediction.actionableItems
        action_items_serialized = [
            item.model_dump() if hasattr(item, "model_dump") else item for item in action_items
        ]

        final_decision_context = final_decision_context or consensus.summaryOfFindings
        confidence_context = confidence_context or reviewer_context

        logger.debug("Synthesizing final decision and action plan")
        synthesis = self.synthesis_generator(
            consensus_analysis_json=json.dumps(consensus.model_dump()),
            actionable_items_json=json.dumps(action_items_serialized),
            final_decision_context=final_decision_context,
            confidence_statement_context=confidence_context,
        ).synthesisAndActionPlan

        report = self.report_builder(
            reportMetadata=metadata.model_dump_json(),
            originalProofSketch=proof_sketch_json,
            reviews=json.dumps([gemini_review, codex_review]),
            synthesisAndActionPlan=synthesis.model_dump_json(),
        ).report
        logger.debug("Building quantitative scores from reviews")
        scores = self.build_scores(gemini_review, codex_review, report)

        overall_elapsed = time.perf_counter() - overall_start
        logger.info(
            "Validation cycle completed in %.2fs | Final decision: %s | Action items: %d | Overall score: %.1f",
            overall_elapsed,
            report.synthesisAndActionPlan.finalDecision,
            len(report.synthesisAndActionPlan.actionableItems),
            scores.average_overall_score,
        )

        return dspy.Prediction(
            report=report, scores=scores, review_1=gemini_review, review_2=codex_review
        )

    def build_scores(
        self, gemini_review: dict, codex_review: dict, report: SketchValidationReport
    ) -> Scores:
        """Extract numerical scores from reviews and report for scoring purposes.

        Args:
            gemini_review: Dictionary from Gemini's SketchValidationReview
            codex_review: Dictionary from Codex's SketchValidationReview
            report: Complete validation report with synthesis

        Returns:
            Scores object with all extracted metrics
        """
        # === EXTRACT GEMINI SCORES ===
        gemini_overall = gemini_review.get("overallAssessment", {})
        gemini_completeness = gemini_review.get("completenessAndCorrectness", {})
        gemini_logical = gemini_review.get("logicalFlowValidation", {})
        gemini_dependency = gemini_review.get("dependencyValidation", {})
        gemini_technical = gemini_review.get("technicalDeepDiveValidation", {})

        # Overall assessment
        gemini_overall_score = gemini_overall.get("score", 0)
        gemini_overall_confidence = gemini_overall.get("confidence", 0)
        gemini_overall_confidence_score = gemini_overall.get("confidenceScore", 0)

        # Completeness
        gemini_completeness_score = gemini_completeness.get("score", 0)
        gemini_completeness_confidence = gemini_completeness.get("confidence", 0)
        gemini_covers_all_claims = gemini_completeness.get("coversAllClaims", False)
        gemini_error_count = len(gemini_completeness.get("identifiedErrors", []))

        # Logical flow
        gemini_logical_flow_score = gemini_logical.get("score", 0)
        gemini_logical_flow_confidence = gemini_logical.get("confidence", 0)
        gemini_logical_flow_sound = gemini_logical.get("isSound", False)
        gemini_logical_gap_count = len(gemini_logical.get("identifiedGaps", []))

        # Dependency (scores may be missing due to model inconsistency)
        gemini_dependency_score = gemini_dependency.get("score")
        gemini_dependency_confidence = gemini_dependency.get("confidence")
        gemini_dependency_issue_count = len(gemini_dependency.get("issues", []))

        # Technical deep dive (scores may be missing)
        gemini_technical_dive_score = gemini_technical.get("score")
        gemini_technical_dive_confidence = gemini_technical.get("confidence")
        gemini_technical_critique_count = len(gemini_technical.get("critiques", []))

        # === EXTRACT CODEX SCORES ===
        codex_overall = codex_review.get("overallAssessment", {})
        codex_completeness = codex_review.get("completenessAndCorrectness", {})
        codex_logical = codex_review.get("logicalFlowValidation", {})
        codex_dependency = codex_review.get("dependencyValidation", {})
        codex_technical = codex_review.get("technicalDeepDiveValidation", {})

        codex_overall_score = codex_overall.get("score", 0)
        codex_overall_confidence = codex_overall.get("confidence", 0)
        codex_overall_confidence_score = codex_overall.get("confidenceScore", 0)

        codex_completeness_score = codex_completeness.get("score", 0)
        codex_completeness_confidence = codex_completeness.get("confidence", 0)
        codex_covers_all_claims = codex_completeness.get("coversAllClaims", False)
        codex_error_count = len(codex_completeness.get("identifiedErrors", []))

        codex_logical_flow_score = codex_logical.get("score", 0)
        codex_logical_flow_confidence = codex_logical.get("confidence", 0)
        codex_logical_flow_sound = codex_logical.get("isSound", False)
        codex_logical_gap_count = len(codex_logical.get("identifiedGaps", []))

        codex_dependency_score = codex_dependency.get("score")
        codex_dependency_confidence = codex_dependency.get("confidence")
        codex_dependency_issue_count = len(codex_dependency.get("issues", []))

        codex_technical_dive_score = codex_technical.get("score")
        codex_technical_dive_confidence = codex_technical.get("confidence")
        codex_technical_critique_count = len(codex_technical.get("critiques", []))

        # === COMPUTE AGGREGATES ===
        average_overall_score = (gemini_overall_score + codex_overall_score) / 2.0
        average_overall_confidence = (gemini_overall_confidence + codex_overall_confidence) / 2.0
        average_completeness_score = (gemini_completeness_score + codex_completeness_score) / 2.0
        average_logical_flow_score = (gemini_logical_flow_score + codex_logical_flow_score) / 2.0

        total_error_count = gemini_error_count + codex_error_count
        total_logical_gap_count = gemini_logical_gap_count + codex_logical_gap_count
        total_dependency_issue_count = gemini_dependency_issue_count + codex_dependency_issue_count
        total_technical_critique_count = (
            gemini_technical_critique_count + codex_technical_critique_count
        )

        both_reviewers_sound = gemini_logical_flow_sound and codex_logical_flow_sound
        both_reviewers_cover_claims = gemini_covers_all_claims and codex_covers_all_claims

        overall_score_variance = (gemini_overall_score - codex_overall_score) ** 2
        completeness_score_variance = (gemini_completeness_score - codex_completeness_score) ** 2
        logical_flow_score_variance = (gemini_logical_flow_score - codex_logical_flow_score) ** 2

        # === SYNTHESIS METRICS ===
        final_decision_map = {
            "Approved for Expansion": 4,
            "Requires Minor Revisions": 3,
            "Requires Major Revisions": 2,
            "Rejected - New Strategy Needed": 1,
        }
        final_decision_numeric = final_decision_map.get(
            report.synthesisAndActionPlan.finalDecision, 0
        )

        action_items = report.synthesisAndActionPlan.actionableItems
        action_item_count = len(action_items)
        critical_action_count = sum(1 for item in action_items if item.priority == "Critical")
        high_action_count = sum(1 for item in action_items if item.priority == "High")
        medium_action_count = sum(1 for item in action_items if item.priority == "Medium")
        low_action_count = sum(1 for item in action_items if item.priority == "Low")

        consensus = report.synthesisAndActionPlan.consensusAnalysis
        points_of_agreement_count = len(consensus.pointsOfAgreement)
        points_of_disagreement_count = len(consensus.pointsOfDisagreement)

        # === DERIVED METRICS ===
        # Quality index: weighted average of scores by confidence
        total_confidence = gemini_overall_confidence + codex_overall_confidence
        if total_confidence > 0:
            overall_quality_index = (
                gemini_overall_score * gemini_overall_confidence
                + codex_overall_score * codex_overall_confidence
            ) / total_confidence
        else:
            overall_quality_index = average_overall_score

        # Risk score: combines issue counts and variance (higher = more risky)
        risk_score = (
            total_error_count * 2.0  # Errors weighted heavily
            + total_logical_gap_count * 1.5  # Gaps weighted moderately
            + total_dependency_issue_count * 1.0
            + overall_score_variance * 0.5  # Disagreement between reviewers
            + completeness_score_variance * 0.5
            + logical_flow_score_variance * 0.5
            + critical_action_count * 3.0  # Critical actions weighted heavily
        )

        return Scores(
            # Gemini scores
            gemini_overall_score=gemini_overall_score,
            gemini_overall_confidence=gemini_overall_confidence,
            gemini_overall_confidence_score=gemini_overall_confidence_score,
            gemini_completeness_score=gemini_completeness_score,
            gemini_completeness_confidence=gemini_completeness_confidence,
            gemini_covers_all_claims=gemini_covers_all_claims,
            gemini_error_count=gemini_error_count,
            gemini_logical_flow_score=gemini_logical_flow_score,
            gemini_logical_flow_confidence=gemini_logical_flow_confidence,
            gemini_logical_flow_sound=gemini_logical_flow_sound,
            gemini_logical_gap_count=gemini_logical_gap_count,
            gemini_dependency_score=gemini_dependency_score,
            gemini_dependency_confidence=gemini_dependency_confidence,
            gemini_dependency_issue_count=gemini_dependency_issue_count,
            gemini_technical_dive_score=gemini_technical_dive_score,
            gemini_technical_dive_confidence=gemini_technical_dive_confidence,
            gemini_technical_critique_count=gemini_technical_critique_count,
            # Codex scores
            codex_overall_score=codex_overall_score,
            codex_overall_confidence=codex_overall_confidence,
            codex_overall_confidence_score=codex_overall_confidence_score,
            codex_completeness_score=codex_completeness_score,
            codex_completeness_confidence=codex_completeness_confidence,
            codex_covers_all_claims=codex_covers_all_claims,
            codex_error_count=codex_error_count,
            codex_logical_flow_score=codex_logical_flow_score,
            codex_logical_flow_confidence=codex_logical_flow_confidence,
            codex_logical_flow_sound=codex_logical_flow_sound,
            codex_logical_gap_count=codex_logical_gap_count,
            codex_dependency_score=codex_dependency_score,
            codex_dependency_confidence=codex_dependency_confidence,
            codex_dependency_issue_count=codex_dependency_issue_count,
            codex_technical_dive_score=codex_technical_dive_score,
            codex_technical_dive_confidence=codex_technical_dive_confidence,
            codex_technical_critique_count=codex_technical_critique_count,
            # Aggregates
            average_overall_score=average_overall_score,
            average_overall_confidence=average_overall_confidence,
            average_completeness_score=average_completeness_score,
            average_logical_flow_score=average_logical_flow_score,
            total_error_count=total_error_count,
            total_logical_gap_count=total_logical_gap_count,
            total_dependency_issue_count=total_dependency_issue_count,
            total_technical_critique_count=total_technical_critique_count,
            both_reviewers_sound=both_reviewers_sound,
            both_reviewers_cover_claims=both_reviewers_cover_claims,
            overall_score_variance=overall_score_variance,
            completeness_score_variance=completeness_score_variance,
            logical_flow_score_variance=logical_flow_score_variance,
            # Synthesis metrics
            final_decision_numeric=final_decision_numeric,
            action_item_count=action_item_count,
            critical_action_count=critical_action_count,
            high_action_count=high_action_count,
            medium_action_count=medium_action_count,
            low_action_count=low_action_count,
            points_of_agreement_count=points_of_agreement_count,
            points_of_disagreement_count=points_of_disagreement_count,
            # Derived metrics
            overall_quality_index=overall_quality_index,
            risk_score=risk_score,
        )
