#!/usr/bin/env python3
"""Proof sketch pipeline orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import TYPE_CHECKING
from uuid import uuid4

import dspy

from mathster.proof_sketcher.sketch_referee_analysis import SketchValidationReview
from mathster.proof_sketcher.sketch_validator import (
    Scores,
    SketchValidationReport,
    SketchValidator,
)
from mathster.proof_sketcher.sketcher import (
    ProofSketch,
    ProofSketchAgent as DraftingAgent,
)

if TYPE_CHECKING:
    from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig


__all__ = ["ProofSketchWorkflowResult", "AgentSketchPipeline"]

logger = logging.getLogger(__name__)


def _iso_timestamp() -> str:
    """Return a timezone-aware ISO 8601 timestamp without microseconds."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ProofSketchWorkflowResult:
    """Container for the draft sketch and its validation report."""

    sketch: ProofSketch
    validation_report: SketchValidationReport
    scores: Scores
    strategy_1: ProofSketch
    strategy_2: ProofSketch
    review_1: SketchValidationReview
    review_2: SketchValidationReview


class AgentSketchPipeline(dspy.Module):
    """Run the full proof sketch workflow, from drafting to validation.

    Creates agents from ProofSketcherLMConfig for automatic model selection:

    ```python
    from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig

    config = ProofSketcherLMConfig.default()
    pipeline = AgentSketchPipeline(lm_config=config)
    ```

    Args:
        lm_config: ProofSketcherLMConfig specifying models for all pipeline stages
        project_root: Optional path to project root for prompt loading
        gemini_prompt: Optional custom prompt for Gemini reviewer
        codex_prompt: Optional custom prompt for Codex reviewer
    """

    def __init__(
        self,
        *,
        lm_config: ProofSketcherLMConfig,
        project_root: str | None = None,
        gemini_prompt: str | None = None,
        codex_prompt: str | None = None,
    ) -> None:
        super().__init__()

        lms = lm_config.to_dspy_lms()

        # Create drafting agent with perspective models and synthesis/fast models
        self._drafting_agent = DraftingAgent(
            strategist_1=lms["perspective_1"],  # Primary strategist
            strategist_2=lms["perspective_2"],  # Secondary strategist
            model=lms["fast"],  # Fast model for simple tasks
            stronger_model=lms["synthesis"],  # Synthesis model for complex reasoning
        )

        # Create validator with perspective and synthesis models
        self._validator = SketchValidator(
            project_root=project_root,
            gemini_prompt=gemini_prompt,
            codex_prompt=codex_prompt,
            perspective_1_lm=lms["perspective_1"],  # Gemini reviewer
            perspective_2_lm=lms["perspective_2"],  # Codex reviewer
            synthesis_lm=lms["synthesis"],  # Consensus/synthesis
            fast_lm=lms["fast"],  # Metadata generation
        )

    def forward(
        self,
        *,
        title_hint: str,
        theorem_label: str,
        theorem_type: str,
        theorem_statement: str,
        document_source: str,
        creation_date: str,
        proof_status: str,
        framework_context: str | None = None,
        operator_notes: str | None = None,
        validation_cycle_id: str | None = None,
        validation_timestamp: str | None = None,
        reviewer_context: str | None = None,
        final_decision_context: str | None = None,
        confidence_context: str | None = None,
    ) -> dspy.Prediction:
        framework_context = framework_context or ""
        operator_notes = operator_notes or ""
        reviewer_context = reviewer_context or (
            "Run dual-math reviewer analysis emphasizing rigor and computational checks."
        )

        # Draft the proof sketch using the modular generation pipeline.
        sketch_prediction = self._drafting_agent(
            title_hint=title_hint,
            theorem_label=theorem_label,
            theorem_type=theorem_type,
            theorem_statement=theorem_statement,
            document_source=document_source,
            creation_date=creation_date,
            proof_status=proof_status,
            framework_context=framework_context,
            operator_notes=operator_notes,
        )
        proof_sketch = sketch_prediction.sketch

        # Validate the sketch to produce the review artifacts.
        validation_cycle_id = validation_cycle_id or str(uuid4())
        validation_timestamp = validation_timestamp or _iso_timestamp()
        validation_prediction = self._validator(
            proof_sketch=proof_sketch.model_dump(),
            sketch_label=theorem_label,
            validation_cycle_id=validation_cycle_id,
            validation_timestamp=validation_timestamp,
            reviewer_context=reviewer_context,
            final_decision_context=final_decision_context or operator_notes,
            confidence_context=confidence_context or framework_context,
        )
        validation_report = validation_prediction.report
        scores = validation_prediction.scores

        # Log overall score for terminal tracking
        overall_score = scores.get_score()
        logger.info(
            "=" * 80 + "\n"
            "PROOF SKETCH VALIDATION COMPLETE\n"
            "Theorem: %s\n"
            "Overall Score: %.2f/100\n"
            "Gemini Score: %d/5 | Codex Score: %d/5\n"
            "Decision: %s\n"+
            "=" * 80,
            theorem_label,
            overall_score,
            scores.gemini_overall_score,
            scores.codex_overall_score,
            validation_report.synthesisAndActionPlan.finalDecision,
        )

        return dspy.Prediction(
            result=ProofSketchWorkflowResult(
                sketch=proof_sketch,
                validation_report=validation_report,
                scores=scores,
                strategy_1=sketch_prediction.strategy_1,
                strategy_2=sketch_prediction.strategy_2,
                review_1=validation_prediction.review_1,
                review_2=validation_prediction.review_2,
            ),
        )
