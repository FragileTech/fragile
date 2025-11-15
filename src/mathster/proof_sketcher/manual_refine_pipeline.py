#!/usr/bin/env python3
"""Manual refinement pipeline for proof sketch generation.

This module provides a custom dspy.Module that replicates dspy.Refine behavior
but allows full access to intermediate results and avoids parallelization conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import time
from typing import Any, Literal, TYPE_CHECKING

import dspy

from mathster.proof_sketcher.feedback_formatter import FeedbackConfig, IterationFeedbackFormatter
from mathster.proof_sketcher.sketch_pipeline import AgentSketchPipeline, ProofSketchWorkflowResult
from mathster.proof_sketcher.sketch_validator import Scores

if TYPE_CHECKING:
    from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig


__all__ = [
    "RefinementIteration",
    "RefinementResult",
    "ManualRefineSketchPipeline",
    "LogVerbosity",
    "FeedbackConfig",
]

logger = logging.getLogger(__name__)


class LogVerbosity(str, Enum):
    """Logging verbosity levels for refinement pipeline."""

    MINIMAL = "minimal"  # Only start/end and best result
    STANDARD = "standard"  # Include per-iteration summaries (default)
    DETAILED = "detailed"  # Add score breakdowns and metrics
    VERBOSE = "verbose"  # Full diagnostic output with all metrics
    DEBUG = "debug"  # Everything including raw data dumps


@dataclass
class RefinementIteration:
    """Store results from a single refinement iteration."""

    iteration_num: int
    """Iteration number (1-indexed)."""

    result: ProofSketchWorkflowResult
    """Complete workflow result from this iteration."""

    score: float
    """Quality score (0-100) from this iteration."""

    prediction: dspy.Prediction
    """Raw DSPy prediction object."""

    elapsed_time: float = 0.0
    """Time elapsed for this iteration in seconds."""

    is_best: bool = False
    """Whether this iteration produced the best result."""

    improvement: float = 0.0
    """Score improvement over previous best (can be negative)."""


@dataclass
class RefinementResult:
    """Store complete refinement results with all intermediate iterations.

    This class provides comprehensive tracking of the refinement process,
    including all iterations, scores, and metadata about convergence.
    """

    best_result: ProofSketchWorkflowResult
    """The highest-scoring workflow result."""

    best_iteration_num: int
    """Iteration number (1-indexed) that produced the best result."""

    best_score: float
    """Quality score (0-100) of the best result."""

    all_iterations: list[RefinementIteration]
    """All refinement iterations with results and scores."""

    scores: list[float]
    """Score progression across all iterations."""

    total_iterations: int
    """Total number of iterations performed."""

    early_stopped: bool
    """Whether refinement stopped early due to threshold."""

    threshold_met: bool
    """Whether the best score met or exceeded the threshold."""

    consecutive_fails: int
    """Number of consecutive iterations without improvement at termination."""

    stopped_reason: str
    """Human-readable explanation of why refinement stopped."""

    total_time: float = 0.0
    """Total time elapsed for all iterations in seconds."""

    average_time_per_iteration: float = 0.0
    """Average time per iteration in seconds."""

    score_improvement: float = 0.0
    """Total improvement from first to best score."""

    score_variance: float = 0.0
    """Variance in scores across iterations."""


class ManualRefineSketchPipeline(dspy.Module):
    """Manual refinement wrapper for AgentSketchPipeline.

    Replicates dspy.Refine behavior with sequential iteration loop,
    allowing access to all intermediate results and avoiding nested
    parallelization conflicts.

    Refinement Logic:
    - Run up to N iterations
    - Track best result by score
    - Early stop if score ≥ threshold
    - Track consecutive failures (score not improving)
    - Stop if fail_count consecutive failures reached

    Creates pipeline from ProofSketcherLMConfig:

    ```python
    from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig

    config = ProofSketcherLMConfig.default()
    refiner = ManualRefineSketchPipeline(lm_config=config, N=5)
    ```

    Example:
        >>> from mathster.proof_sketcher.llm_config import ProofSketcherLMConfig
        >>> config = ProofSketcherLMConfig.cost_optimized()
        >>> refiner = ManualRefineSketchPipeline(
        ...     lm_config=config,
        ...     N=5,
        ...     threshold=60,
        ...     fail_count=5
        ... )
        >>> result = refiner(
        ...     title_hint="KL Convergence",
        ...     theorem_label="thm-kl-conv",
        ...     ...
        ... )
        >>> print(f"Best score: {result.best_score}")
        >>> print(f"Iterations: {result.total_iterations}")
        >>> print(f"Reason: {result.stopped_reason}")
    """

    def __init__(
        self,
        *,
        lm_config: ProofSketcherLMConfig,
        N: int = 5,
        threshold: float = 60.0,
        fail_count: int = 5,
        verbosity: LogVerbosity | str = LogVerbosity.STANDARD,
        log_json_path: str | None = None,
        enable_iteration_feedback: bool = True,
        feedback_config: FeedbackConfig | None = None,
        project_root: str | None = None,
        gemini_prompt: str | None = None,
        codex_prompt: str | None = None,
    ) -> None:
        """Initialize manual refinement pipeline.

        Args:
            lm_config: ProofSketcherLMConfig for pipeline creation
            N: Maximum number of refinement iterations
            threshold: Quality score threshold for early stopping (0-100)
            fail_count: Maximum consecutive failures before stopping
            verbosity: Logging verbosity level (minimal/standard/detailed/verbose/debug)
            log_json_path: Optional path to export refinement data as JSON
            enable_iteration_feedback: Enable feedback injection from best iteration (default: True)
            feedback_config: Optional feedback configuration. If None, uses default config.
            project_root: Optional path to project root
            gemini_prompt: Optional custom prompt for Gemini reviewer
            codex_prompt: Optional custom prompt for Codex reviewer
        """
        super().__init__()

        # Create pipeline from LM config
        self.pipeline = AgentSketchPipeline(
            lm_config=lm_config,
            project_root=project_root,
            gemini_prompt=gemini_prompt,
            codex_prompt=codex_prompt,
        )

        self.N = N
        self.threshold = threshold
        self.fail_count = fail_count
        self.verbosity = LogVerbosity(verbosity) if isinstance(verbosity, str) else verbosity
        self.log_json_path = log_json_path
        self.enable_iteration_feedback = enable_iteration_feedback
        self.feedback_formatter = (
            IterationFeedbackFormatter(feedback_config)
            if enable_iteration_feedback
            else None
        )

    def _log_score_breakdown(self, scores: Scores, iteration: int) -> None:
        """Log detailed score component breakdown."""
        if self.verbosity.value not in ["detailed", "verbose", "debug"]:
            return

        logger.info(f"\n--- Score Breakdown (Iteration {iteration}) ---")

        # Reviewer scores
        logger.info("Reviewer Scores:")
        logger.info(
            f"  Gemini: Overall={scores.gemini_overall_score}/5, "
            f"Completeness={scores.gemini_completeness_score}/5, "
            f"Logical Flow={scores.gemini_logical_flow_score}/5, "
            f"Confidence={scores.gemini_overall_confidence}/5"
        )
        logger.info(
            f"  Codex:  Overall={scores.codex_overall_score}/5, "
            f"Completeness={scores.codex_completeness_score}/5, "
            f"Logical Flow={scores.codex_logical_flow_score}/5, "
            f"Confidence={scores.codex_overall_confidence}/5"
        )

        # Issue counts
        logger.info("Issue Counts:")
        logger.info(
            f"  Errors: Gemini={scores.gemini_error_count}, "
            f"Codex={scores.codex_error_count}, Total={scores.total_error_count}"
        )
        logger.info(
            f"  Gaps: Gemini={scores.gemini_logical_gap_count}, "
            f"Codex={scores.codex_logical_gap_count}, Total={scores.total_logical_gap_count}"
        )
        logger.info(
            f"  Dependencies: Gemini={scores.gemini_dependency_issue_count}, "
            f"Codex={scores.codex_dependency_issue_count}, "
            f"Total={scores.total_dependency_issue_count}"
        )
        logger.info(
            f"  Technical Critiques: Gemini={scores.gemini_technical_critique_count}, "
            f"Codex={scores.codex_technical_critique_count}, "
            f"Total={scores.total_technical_critique_count}"
        )

        # Agreement/disagreement
        logger.info("Reviewer Agreement:")
        logger.info(f"  Both sound: {scores.both_reviewers_sound}")
        logger.info(f"  Both cover all claims: {scores.both_reviewers_cover_claims}")
        logger.info(
            f"  Score variance: Overall={scores.overall_score_variance:.2f}, "
            f"Completeness={scores.completeness_score_variance:.2f}, "
            f"LogicalFlow={scores.logical_flow_score_variance:.2f}"
        )

        # Synthesis metrics
        logger.info("Synthesis Metrics:")
        logger.info(
            f"  Action items: Total={scores.action_item_count}, "
            f"Critical={scores.critical_action_count}, High={scores.high_action_count}, "
            f"Medium={scores.medium_action_count}, Low={scores.low_action_count}"
        )
        logger.info(
            f"  Consensus: Agreements={scores.points_of_agreement_count}, "
            f"Disagreements={scores.points_of_disagreement_count}"
        )
        logger.info(
            f"  Quality Index: {scores.overall_quality_index:.2f}, "
            f"Risk Score: {scores.risk_score:.2f}"
        )

    def _log_iteration_comparison(
        self,
        current_iteration: RefinementIteration,
        best_iteration: RefinementIteration | None,
    ) -> None:
        """Log comparison between current and best iteration."""
        if self.verbosity.value not in ["verbose", "debug"]:
            return

        if best_iteration is None:
            logger.info("--- First Iteration (Baseline) ---")
            return

        logger.info(f"\n--- Iteration Comparison ---")
        logger.info(
            f"Current (#{current_iteration.iteration_num}) vs "
            f"Best (#{best_iteration.iteration_num})"
        )

        # Score comparison
        curr_scores = current_iteration.result.scores
        best_scores = best_iteration.result.scores

        delta = current_iteration.score - best_iteration.score
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        logger.info(
            f"  Overall Score: {current_iteration.score:.2f} vs "
            f"{best_iteration.score:.2f} ({delta_str})"
        )

        # Component deltas
        logger.info("  Component Deltas:")
        avg_overall_delta = curr_scores.average_overall_score - best_scores.average_overall_score
        avg_complete_delta = (
            curr_scores.average_completeness_score - best_scores.average_completeness_score
        )
        avg_logic_delta = (
            curr_scores.average_logical_flow_score - best_scores.average_logical_flow_score
        )

        logger.info(f"    Avg Overall: {avg_overall_delta:+.2f}")
        logger.info(f"    Avg Completeness: {avg_complete_delta:+.2f}")
        logger.info(f"    Avg Logical Flow: {avg_logic_delta:+.2f}")

        # Issue count deltas
        error_delta = curr_scores.total_error_count - best_scores.total_error_count
        gap_delta = curr_scores.total_logical_gap_count - best_scores.total_logical_gap_count
        action_delta = curr_scores.action_item_count - best_scores.action_item_count

        logger.info(f"  Issue Deltas:")
        logger.info(f"    Errors: {error_delta:+d}")
        logger.info(f"    Logical Gaps: {gap_delta:+d}")
        logger.info(f"    Action Items: {action_delta:+d}")

        # Decision comparison
        curr_decision = current_iteration.result.validation_report.synthesisAndActionPlan.finalDecision
        best_decision = best_iteration.result.validation_report.synthesisAndActionPlan.finalDecision
        logger.info(f"  Decision: {curr_decision} vs {best_decision}")

    def _log_convergence_indicators(self, iterations: list[RefinementIteration]) -> None:
        """Log convergence analysis across iterations."""
        if self.verbosity.value not in ["verbose", "debug"] or len(iterations) < 2:
            return

        logger.info(f"\n--- Convergence Analysis ---")

        scores = [it.score for it in iterations]

        # Score trend
        if len(scores) >= 2:
            trend_direction = "improving" if scores[-1] > scores[0] else "declining"
            logger.info(f"  Overall Trend: {trend_direction}")

        # Moving average (if enough iterations)
        if len(scores) >= 3:
            recent_avg = sum(scores[-3:]) / min(3, len(scores))
            logger.info(f"  Recent Avg (last 3): {recent_avg:.2f}")

        # Best iteration timing
        best_idx = max(range(len(iterations)), key=lambda i: iterations[i].score)
        logger.info(f"  Best found at: Iteration {best_idx + 1}/{len(iterations)}")

        # Improvement rate
        improvements = [it.improvement for it in iterations if it.improvement > 0]
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            logger.info(f"  Avg Improvement (when positive): {avg_improvement:.2f}")

    def _export_json_log(self, result: RefinementResult) -> None:
        """Export refinement results to JSON file."""
        if not self.log_json_path:
            return

        try:
            export_data = {
                "best_score": result.best_score,
                "best_iteration": result.best_iteration_num,
                "total_iterations": result.total_iterations,
                "scores": result.scores,
                "stopped_reason": result.stopped_reason,
                "threshold_met": result.threshold_met,
                "early_stopped": result.early_stopped,
                "total_time": result.total_time,
                "average_time_per_iteration": result.average_time_per_iteration,
                "score_improvement": result.score_improvement,
                "score_variance": result.score_variance,
                "iterations": [
                    {
                        "iteration_num": it.iteration_num,
                        "score": it.score,
                        "elapsed_time": it.elapsed_time,
                        "is_best": it.is_best,
                        "improvement": it.improvement,
                        "decision": it.result.validation_report.synthesisAndActionPlan.finalDecision,
                        "gemini_score": it.result.scores.gemini_overall_score,
                        "codex_score": it.result.scores.codex_overall_score,
                        "total_errors": it.result.scores.total_error_count,
                        "total_gaps": it.result.scores.total_logical_gap_count,
                        "action_items": it.result.scores.action_item_count,
                    }
                    for it in result.all_iterations
                ],
            }

            with open(self.log_json_path, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Refinement log exported to: {self.log_json_path}")
        except Exception as e:
            logger.error(f"Failed to export JSON log: {e}", exc_info=True)

    def _extract_feedback_summary(self, result: ProofSketchWorkflowResult) -> str:
        """Extract brief summary for operator_notes.

        Creates a concise 2-3 sentence summary with:
        - Final decision
        - Top 2-3 critical issues
        - Overall guidance

        Args:
            result: Workflow result from previous iteration

        Returns:
            Brief feedback summary (2-3 sentences)
        """
        report = result.validation_report
        scores = result.scores

        decision = report.synthesisAndActionPlan.finalDecision
        score = scores.get_score()

        # Get top critical/high priority actions
        critical_actions = [
            item
            for item in report.synthesisAndActionPlan.actionableItems
            if item.priority in ["Critical", "High"]
        ][:3]  # Top 3

        summary_parts = []

        # Decision and score
        summary_parts.append(
            f"Previous iteration scored {score:.1f}/100 with decision: {decision}."
        )

        # Top issues
        if critical_actions:
            issues = ", ".join(
                action.description.split(".")[0][:80] for action in critical_actions
            )
            summary_parts.append(f"Address these issues: {issues}.")

        # Overall guidance
        if score < 40:
            summary_parts.append("Major revisions needed - reconsider proof strategy.")
        elif score < 60:
            summary_parts.append("Focus on fixing critical gaps and errors.")
        else:
            summary_parts.append("Minor refinements needed to meet threshold.")

        return " ".join(summary_parts)

    def _inject_iteration_feedback(
        self,
        base_kwargs: dict[str, Any],
        best_iteration: RefinementIteration,
    ) -> dict[str, Any]:
        """Inject feedback from best iteration into kwargs.

        Injects feedback into both parameters:
        - Summary → operator_notes (brief actionable guidance)
        - Full feedback → framework_context (detailed analysis + suggestions)

        Args:
            base_kwargs: Original kwargs from user
            best_iteration: Best iteration result so far

        Returns:
            Updated kwargs with feedback injected
        """
        if not self.feedback_formatter:
            return base_kwargs

        # Generate combined feedback (detailed + suggestions)
        full_feedback = self.feedback_formatter.format_combined(
            best_iteration.result,
            best_iteration.iteration_num,
        )

        # Extract summary for operator_notes
        summary = self._extract_feedback_summary(best_iteration.result)

        # Inject into kwargs
        updated_kwargs = base_kwargs.copy()

        # Append to operator_notes (brief actionable summary)
        updated_kwargs["operator_notes"] = (
            base_kwargs.get("operator_notes", "")
            + f"\n\n## Previous Iteration Guidance:\n{summary}"
        )

        # Append to framework_context (full detailed feedback)
        updated_kwargs["framework_context"] = (
            base_kwargs.get("framework_context", "")
            + f"\n\n## Iteration {best_iteration.iteration_num} Feedback:\n{full_feedback}"
        )

        return updated_kwargs

    def forward(self, **kwargs: Any) -> RefinementResult:
        """Run refinement loop with full intermediate result tracking.

        Args:
            **kwargs: All arguments to pass to AgentSketchPipeline.forward()

        Returns:
            RefinementResult with best result, all iterations, and metadata
        """
        start_time = time.time()

        if self.verbosity != LogVerbosity.MINIMAL:
            logger.info("=" * 80)
            logger.info("MANUAL REFINEMENT PIPELINE STARTED")
            logger.info(f"Max iterations: {self.N}")
            logger.info(f"Score threshold: {self.threshold}")
            logger.info(f"Fail count: {self.fail_count}")
            logger.info(f"Verbosity: {self.verbosity.value}")
            logger.info(f"Iteration feedback: {'enabled' if self.enable_iteration_feedback else 'disabled'}")
            logger.info("=" * 80)

        # Track all iterations and best result
        iterations: list[RefinementIteration] = []
        best_iteration: RefinementIteration | None = None
        consecutive_fails = 0
        stopped_reason = ""

        for i in range(1, self.N + 1):
            iteration_start = time.time()

            if self.verbosity != LogVerbosity.MINIMAL:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"ITERATION {i}/{self.N}")
                logger.info(f"{'=' * 80}")

            # Inject feedback from best iteration (if available)
            if best_iteration is not None and self.enable_iteration_feedback:
                current_kwargs = self._inject_iteration_feedback(kwargs, best_iteration)
                if self.verbosity in [
                    LogVerbosity.DETAILED,
                    LogVerbosity.VERBOSE,
                    LogVerbosity.DEBUG,
                ]:
                    logger.info(
                        f"Injecting feedback from best iteration "
                        f"(#{best_iteration.iteration_num}, score: {best_iteration.score:.2f})"
                    )
            else:
                current_kwargs = kwargs

            # Run pipeline
            try:
                prediction = self.pipeline(**current_kwargs)
            except Exception as e:
                logger.error(f"Iteration {i} failed with exception: {e}", exc_info=True)
                consecutive_fails += 1
                if consecutive_fails >= self.fail_count:
                    stopped_reason = f"Failed {consecutive_fails} consecutive times (exception)"
                    logger.warning(f"Stopping: {stopped_reason}")
                    break
                logger.warning(f"Consecutive failures: {consecutive_fails}/{self.fail_count}")
                continue

            iteration_elapsed = time.time() - iteration_start

            # Extract result and score
            result: ProofSketchWorkflowResult = prediction.result
            score = result.scores.get_score()

            # Calculate improvement
            improvement = score - best_iteration.score if best_iteration else score
            is_best = best_iteration is None or score > best_iteration.score

            # Store iteration
            iteration = RefinementIteration(
                iteration_num=i,
                result=result,
                score=score,
                prediction=prediction,
                elapsed_time=iteration_elapsed,
                is_best=is_best,
                improvement=improvement,
            )
            iterations.append(iteration)

            # Log iteration result (standard verbosity and above)
            if self.verbosity != LogVerbosity.MINIMAL:
                logger.info(f"Iteration {i} score: {score:.2f}/100 (took {iteration_elapsed:.1f}s)")
                logger.info(
                    f"Gemini: {result.scores.gemini_overall_score}/5 | "
                    f"Codex: {result.scores.codex_overall_score}/5"
                )
                logger.info(
                    f"Decision: {result.validation_report.synthesisAndActionPlan.finalDecision}"
                )

            # Log detailed score breakdown (detailed verbosity and above)
            self._log_score_breakdown(result.scores, i)

            # Update best result
            if is_best:
                best_iteration = iteration
                consecutive_fails = 0
                if self.verbosity != LogVerbosity.MINIMAL:
                    logger.info(
                        f"✓ NEW BEST SCORE: {score:.2f}/100 "
                        f"(+{improvement:.2f} improvement)"
                    )
            else:
                consecutive_fails += 1
                if self.verbosity != LogVerbosity.MINIMAL:
                    logger.info(
                        f"✗ No improvement (best: {best_iteration.score:.2f}/100, "
                        f"consecutive fails: {consecutive_fails}/{self.fail_count})"
                    )

            # Log iteration comparison (verbose and above)
            self._log_iteration_comparison(iteration, best_iteration if not is_best else None)

            # Log convergence indicators (verbose and above)
            self._log_convergence_indicators(iterations)

            # Check early stop conditions
            if score >= self.threshold:
                stopped_reason = f"Threshold {self.threshold} met (score: {score:.2f})"
                if self.verbosity != LogVerbosity.MINIMAL:
                    logger.info(f"\n{'=' * 80}")
                    logger.info(f"EARLY STOP: {stopped_reason}")
                    logger.info(f"{'=' * 80}")
                break

            if consecutive_fails >= self.fail_count:
                stopped_reason = (
                    f"Failed to improve for {consecutive_fails} consecutive iterations"
                )
                logger.warning(f"\n{'=' * 80}")
                logger.warning(f"STOPPING: {stopped_reason}")
                logger.warning(f"{'=' * 80}")
                break

        # Determine final stopping reason
        if not stopped_reason:
            stopped_reason = f"Reached maximum iterations ({self.N})"

        # Handle case where all iterations failed
        if best_iteration is None:
            raise RuntimeError(
                f"All {len(iterations)} refinement iterations failed. "
                f"No valid results produced."
            )

        # Calculate timing and statistics
        total_time = time.time() - start_time
        avg_time = total_time / len(iterations) if iterations else 0.0

        # Calculate score statistics
        scores_list = [it.score for it in iterations]
        score_improvement = best_iteration.score - scores_list[0] if scores_list else 0.0
        score_variance = (
            sum((s - sum(scores_list) / len(scores_list)) ** 2 for s in scores_list)
            / len(scores_list)
            if len(scores_list) > 1
            else 0.0
        )

        # Build final result
        result = RefinementResult(
            best_result=best_iteration.result,
            best_iteration_num=best_iteration.iteration_num,
            best_score=best_iteration.score,
            all_iterations=iterations,
            scores=scores_list,
            total_iterations=len(iterations),
            early_stopped=(best_iteration.score >= self.threshold),
            threshold_met=(best_iteration.score >= self.threshold),
            consecutive_fails=consecutive_fails,
            stopped_reason=stopped_reason,
            total_time=total_time,
            average_time_per_iteration=avg_time,
            score_improvement=score_improvement,
            score_variance=score_variance,
        )

        # Log final summary (all verbosity levels)
        logger.info(f"\n{'=' * 80}")
        logger.info("MANUAL REFINEMENT PIPELINE COMPLETE")
        logger.info(f"{'=' * 80}")
        logger.info(
            f"Best score: {result.best_score:.2f}/100 (iteration {result.best_iteration_num})"
        )
        logger.info(f"Total iterations: {result.total_iterations}")
        logger.info(f"Total time: {result.total_time:.1f}s (avg: {result.average_time_per_iteration:.1f}s/iter)")

        if self.verbosity != LogVerbosity.MINIMAL:
            logger.info(f"Score progression: {[f'{s:.2f}' for s in result.scores]}")
            logger.info(f"Score improvement: {result.score_improvement:+.2f}")
            logger.info(f"Score variance: {result.score_variance:.2f}")
            logger.info(f"Threshold met: {result.threshold_met}")
            logger.info(f"Early stopped: {result.early_stopped}")
            logger.info(f"Stopped reason: {result.stopped_reason}")

        logger.info(f"{'=' * 80}\n")

        # Export JSON log if requested
        self._export_json_log(result)

        return result
