#!/usr/bin/env python3
"""Signatures and models for full proof sketch validation cycle reports."""

from __future__ import annotations

import json
from typing import Literal

import dspy
from pydantic import BaseModel, Field

from mathster.proof_sketcher.sketch_referee_analysis import SketchReviewAgent


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

        self.review_agent_gemini = SketchReviewAgent(
            project_root=project_root,
            claude_system_prompt=gemini_prompt,
        )
        self.review_agent_codex = SketchReviewAgent(
            project_root=project_root,
            claude_system_prompt=codex_prompt,
        )

    def _run_review(
        self,
        agent: SketchReviewAgent,
        reviewer_name: str,
        timestamp: str,
        proof_sketch_json: str,
        extra_instructions: str,
    ) -> dict:
        """Invoke a SketchReviewAgent with shared proof sketch context."""
        json_payload = json.loads(proof_sketch_json)
        payload = json.dumps({
            "proof_sketch": json_payload,
            "instructions": extra_instructions,
        })
        print("Running review with", reviewer_name)
        print(json_payload)
        prediction = agent(
            reviewer=reviewer_name,
            timestamp=timestamp,
            overallAssessment=payload,
            logicalFlowValidation=payload,
            dependencyValidation=payload,
            technicalDeepDiveValidation=payload,
            completenessAndCorrectness=payload,
        )
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
        reviewer_context = (
            reviewer_context or "Perform a thorough dual-review of the provided proof sketch."
        )
        metadata = self.metadata_generator(
            sketch_label=sketch_label,
            cycle_uuid=validation_cycle_id,
            timestamp=validation_timestamp,
        ).reportMetadata

        proof_sketch_json = json.dumps(proof_sketch)
        gemini_review = self._run_review(
            self.review_agent_gemini,
            reviewer_name="Gemini 2.5 Flash",
            timestamp=validation_timestamp,
            proof_sketch_json=proof_sketch_json,
            extra_instructions=reviewer_context,
        )
        codex_review = self._run_review(
            self.review_agent_codex,
            reviewer_name="GPT-5 Codex",
            timestamp=validation_timestamp,
            proof_sketch_json=proof_sketch_json,
            extra_instructions=reviewer_context,
        )

        consensus = self.consensus_generator(
            reviewer_notes=reviewer_context,
            gemini_review=json.dumps(gemini_review),
            codex_review=json.dumps(codex_review),
        ).consensusAnalysis

        consolidated_feedback = (
            consensus.summaryOfFindings
            + "\nPoints of agreement:\n"
            + "\n".join(f"- {p}" for p in consensus.pointsOfAgreement)
        )
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

        return dspy.Prediction(report=report)
