"""
Review Helpers: Building, Analyzing, and Dual Review Protocol.

This module provides utilities for:
1. ReviewBuilder: Create Review objects from LLM responses and validation results
2. ReviewAnalyzer: Compute metrics, actionability, and suggest next actions
3. DualReviewProtocol: Submit to Gemini + Codex, compare results, synthesize

All functions are pure or total functions following Lean-compatible patterns.

Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from mathster.core.reviews import (
    Review,
    ReviewComparison,
    ReviewIssue,
    ReviewSeverity,
    ReviewSource,
    ValidationResult,
)
from mathster.registry.review_registry import get_review_registry


# =============================================================================
# REVIEW BUILDER
# =============================================================================


class ReviewBuilder:
    """
    Builder for creating Review objects from various sources.

    Provides factory methods for:
    - LLM responses (Gemini, Codex)
    - Validation failures (dataflow, SymPy, framework checker)
    - Manual reviews
    """

    @staticmethod
    def from_llm_response(
        response_text: str,
        object_id: str,
        object_type: str,
        iteration: int,
        source: ReviewSource,
    ) -> Review:
        """
        Create Review from LLM response text.

        Expected format (flexible, uses heuristics):
        ```
        Overall Assessment: ...

        Issues:
        1. [CRITICAL] Title
           Location: ...
           Problem: ...
           Fix: ...

        2. [MAJOR] ...

        Strengths:
        - ...

        Scores:
        Rigor: 7/10
        Soundness: 6/10
        ...
        ```

        Pure function: parsing only, no side effects.

        Args:
            response_text: LLM response text
            object_id: ID of object being reviewed
            object_type: Type of object (ProofBox, etc.)
            iteration: Iteration number
            source: Review source (GEMINI_2_5_PRO, CODEX, etc.)

        Returns:
            Review object
        """
        # Parse issues
        issues = ReviewBuilder._parse_issues(response_text, object_id)

        # Parse strengths
        strengths = ReviewBuilder._parse_strengths(response_text)

        # Parse overall assessment
        overall = ReviewBuilder._parse_overall_assessment(response_text)

        # Parse scores
        scores = ReviewBuilder._parse_scores(response_text)

        # Compute metrics
        mean_actionability = (
            sum(i.actionability_score for i in issues) / len(issues) if issues else 1.0
        )
        blocking_count = sum(1 for i in issues if i.is_blocking())

        # Generate review ID
        review_id = f"review-{object_id}-iter{iteration}-{source.value}"

        return Review(
            review_id=review_id,
            object_id=object_id,
            object_type=object_type,
            iteration=iteration,
            timestamp=datetime.now(),
            source=source,
            issues=issues,
            strengths=strengths,
            overall_assessment=overall or "No overall assessment provided.",
            rigor_score=scores.get("rigor", 5.0),
            soundness_score=scores.get("soundness", 5.0),
            completeness_score=scores.get("completeness", 5.0),
            clarity_score=scores.get("clarity", 5.0),
            framework_consistency_score=scores.get("framework_consistency", 5.0),
            mean_actionability=mean_actionability,
            blocking_issue_count=blocking_count,
            validation_results=[],
            validation_passed=True,  # LLM review doesn't affect validation
            previous_review_id=None,  # Will be set by caller
            addresses_issues=[],
            llm_responses=[],
        )

    @staticmethod
    def _parse_issues(response_text: str, object_id: str) -> list[ReviewIssue]:
        """Parse issues from LLM response text."""
        issues = []

        # Find issues section
        issues_section = ReviewBuilder._extract_section(response_text, "issues")
        if not issues_section:
            return []

        # Parse individual issues (numbered list)
        issue_pattern = r"(\d+)\.\s*\[([A-Z_]+)\]\s*(.+?)(?=\n\d+\.\s*\[|$)"
        matches = re.finditer(issue_pattern, issues_section, re.DOTALL | re.IGNORECASE)

        for i, match in enumerate(matches):
            severity_str = match.group(2).upper()
            issue_text = match.group(3).strip()

            # Parse severity
            severity = ReviewBuilder._parse_severity(severity_str)

            # Extract components
            location = ReviewBuilder._extract_field(issue_text, "location") or "unknown"
            title = ReviewBuilder._extract_field(issue_text, "title") or f"Issue {i + 1}"
            problem = ReviewBuilder._extract_field(issue_text, "problem") or issue_text[:200]
            mechanism = ReviewBuilder._extract_field(issue_text, "mechanism")
            evidence = ReviewBuilder._extract_field(issue_text, "evidence")
            suggested_fix = ReviewBuilder._extract_field(issue_text, "fix")
            impact = ReviewBuilder._extract_list(issue_text, "impact")

            # Compute actionability
            actionability = ReviewBuilder._compute_actionability(suggested_fix, mechanism)

            issues.append(
                ReviewIssue(
                    id=f"issue-{object_id}-iter-{i + 1}",
                    severity=severity,
                    location=location,
                    title=title,
                    problem=problem,
                    mechanism=mechanism,
                    evidence=evidence,
                    impact=impact,
                    suggested_fix=suggested_fix,
                    actionability_score=actionability,
                    framework_references=[],  # TODO: extract from text
                    validation_failure=None,
                )
            )

        return issues

    @staticmethod
    def _parse_severity(severity_str: str) -> ReviewSeverity:
        """Parse severity from string."""
        severity_map = {
            "CRITICAL": ReviewSeverity.CRITICAL,
            "MAJOR": ReviewSeverity.MAJOR,
            "MINOR": ReviewSeverity.MINOR,
            "SUGGESTION": ReviewSeverity.SUGGESTION,
            "VALIDATION_FAILURE": ReviewSeverity.VALIDATION_FAILURE,
            "FRAMEWORK_INCONSISTENCY": ReviewSeverity.FRAMEWORK_INCONSISTENCY,
        }
        return severity_map.get(severity_str, ReviewSeverity.MINOR)

    @staticmethod
    def _extract_section(text: str, section_name: str) -> str | None:
        """Extract section from markdown-like text."""
        pattern = rf"(?:^|\n)#+\s*{section_name}\s*:?\s*\n(.*?)(?=\n#+\s|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_field(text: str, field_name: str) -> str | None:
        """Extract field from text (e.g., 'Location: step-2')."""
        pattern = rf"{field_name}\s*:\s*(.+?)(?=\n[A-Z][a-z]+\s*:|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_list(text: str, field_name: str) -> list[str]:
        """Extract list from text (e.g., 'Impact: - item1 - item2')."""
        section = ReviewBuilder._extract_field(text, field_name)
        if not section:
            return []
        items = re.findall(r"[-*]\s*(.+)", section)
        return [item.strip() for item in items]

    @staticmethod
    def _parse_strengths(response_text: str) -> list[str]:
        """Parse strengths from LLM response."""
        strengths_section = ReviewBuilder._extract_section(response_text, "strengths")
        if not strengths_section:
            return []
        items = re.findall(r"[-*]\s*(.+)", strengths_section)
        return [item.strip() for item in items]

    @staticmethod
    def _parse_overall_assessment(response_text: str) -> str | None:
        """Parse overall assessment from LLM response."""
        return ReviewBuilder._extract_section(response_text, "overall assessment")

    @staticmethod
    def _parse_scores(response_text: str) -> dict[str, float]:
        """Parse scores from LLM response."""
        scores = {}
        scores_section = ReviewBuilder._extract_section(response_text, "scores")
        if not scores_section:
            return scores

        score_names = ["rigor", "soundness", "completeness", "clarity", "framework_consistency"]
        for name in score_names:
            pattern = rf"{name}\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10"
            match = re.search(pattern, scores_section, re.IGNORECASE)
            if match:
                scores[name] = float(match.group(1))

        return scores

    @staticmethod
    def _compute_actionability(suggested_fix: str | None, mechanism: str | None) -> float:
        """
        Compute actionability score based on fix clarity.

        Actionability heuristic:
        - Has specific fix + mechanism: 1.0
        - Has specific fix, no mechanism: 0.8
        - Has vague fix: 0.5
        - No fix: 0.2
        """
        if not suggested_fix:
            return 0.2

        # Check if fix is specific (has code, equations, or step-by-step)
        specific_indicators = ["step", "replace", "add", "change", "$$", "```", "equation"]
        is_specific = any(ind in suggested_fix.lower() for ind in specific_indicators)

        if is_specific and mechanism:
            return 1.0
        if is_specific:
            return 0.8
        if mechanism:
            return 0.6
        return 0.5

    @staticmethod
    def from_validation_failure(
        validator: str,
        errors: list[str],
        warnings: list[str],
        object_id: str,
        object_type: str,
        iteration: int,
        metadata: dict[str, Any] | None = None,
    ) -> Review:
        """
        Create Review from validation failure.

        Pure function: creates Review with VALIDATION_FAILURE issues.

        Args:
            validator: Validator name (e.g., 'dataflow')
            errors: Validation error messages
            warnings: Validation warnings
            object_id: ID of object
            object_type: Type of object
            iteration: Iteration number
            metadata: Optional validation metadata

        Returns:
            Review object with validation issues
        """
        issues = []

        # Create issue for each error
        for i, error in enumerate(errors):
            issues.append(
                ReviewIssue(
                    id=f"issue-{object_id}-val-{i + 1}",
                    severity=ReviewSeverity.VALIDATION_FAILURE,
                    location="automated-validation",
                    title=f"{validator} validation failure",
                    problem=error,
                    mechanism=None,
                    evidence=None,
                    impact=[],
                    suggested_fix=None,
                    actionability_score=0.5,  # Medium actionability
                    framework_references=[],
                    validation_failure={"validator": validator, "error": error},
                )
            )

        # Create issue for warnings (if any)
        if warnings:
            issues.append(
                ReviewIssue(
                    id=f"issue-{object_id}-val-warnings",
                    severity=ReviewSeverity.SUGGESTION,
                    location="automated-validation",
                    title=f"{validator} warnings",
                    problem="\n".join(warnings),
                    mechanism=None,
                    evidence=None,
                    impact=[],
                    suggested_fix=None,
                    actionability_score=0.5,
                    framework_references=[],
                    validation_failure=None,
                )
            )

        validation_result = ValidationResult(
            validator=validator,
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        review_id = f"review-{object_id}-iter{iteration}-{validator}"

        return Review(
            review_id=review_id,
            object_id=object_id,
            object_type=object_type,
            iteration=iteration,
            timestamp=datetime.now(),
            source=ReviewSource.DATAFLOW_VALIDATOR
            if validator == "dataflow"
            else ReviewSource.SYMPY_VALIDATOR
            if validator == "sympy"
            else ReviewSource.FRAMEWORK_CHECKER,
            issues=issues,
            strengths=[],
            overall_assessment=f"{validator} validation {'passed' if not errors else 'failed'}",
            rigor_score=10.0 if not errors else 3.0,
            soundness_score=10.0 if not errors else 3.0,
            completeness_score=10.0 if not errors else 5.0,
            clarity_score=10.0,
            framework_consistency_score=10.0 if not errors else 5.0,
            mean_actionability=0.5,
            blocking_issue_count=len(errors),
            validation_results=[validation_result],
            validation_passed=len(errors) == 0,
            previous_review_id=None,
            addresses_issues=[],
            llm_responses=[],
        )


# =============================================================================
# REVIEW ANALYZER
# =============================================================================


class ReviewAnalyzer:
    """
    Analyzer for computing review metrics and suggesting next actions.

    All methods are pure functions.
    """

    @staticmethod
    def compute_review_metrics(review: Review) -> dict[str, float]:
        """
        Pure function: Compute comprehensive metrics for a review.

        Returns dict with:
        - avg_score: Overall average score
        - issue_severity_distribution: Dict[severity, count]
        - actionability_mean: Mean actionability
        - actionability_std: Std deviation of actionability
        - blocking_ratio: Fraction of issues that are blocking
        """
        avg_score = review.get_average_score()

        severity_dist = {}
        for severity in ReviewSeverity:
            count = len(review.get_issues_by_severity(severity))
            if count > 0:
                severity_dist[severity.value] = count

        actionabilities = [i.actionability_score for i in review.issues]
        actionability_mean = (
            sum(actionabilities) / len(actionabilities) if actionabilities else 1.0
        )
        actionability_std = (
            (sum((a - actionability_mean) ** 2 for a in actionabilities) / len(actionabilities))
            ** 0.5
            if actionabilities
            else 0.0
        )

        blocking_ratio = review.blocking_issue_count / len(review.issues) if review.issues else 0.0

        return {
            "avg_score": avg_score,
            "severity_distribution": severity_dist,
            "actionability_mean": actionability_mean,
            "actionability_std": actionability_std,
            "blocking_ratio": blocking_ratio,
        }

    @staticmethod
    def identify_blocking_issues(review: Review) -> list[ReviewIssue]:
        """
        Pure function: Get all blocking issues.

        Convenience wrapper for review.get_blocking_issues().
        """
        return review.get_blocking_issues()

    @staticmethod
    def suggest_next_action(review: Review) -> str:
        """
        Pure function: Suggest next action based on review status.

        Returns one of:
        - "fix-critical": Has CRITICAL issues, must fix immediately
        - "fix-major": Has MAJOR issues, significant rework needed
        - "fix-validation": Validation failed, fix dataflow/SymPy errors
        - "fix-minor": Only MINOR issues, small fixes
        - "review-suggestions": Only SUGGESTION issues, optional improvements
        - "ready": No issues, ready for publication
        """
        review.get_status()

        critical = review.get_issues_by_severity(ReviewSeverity.CRITICAL)
        if critical:
            return "fix-critical"

        validation = review.get_issues_by_severity(ReviewSeverity.VALIDATION_FAILURE)
        if validation:
            return "fix-validation"

        major = review.get_issues_by_severity(ReviewSeverity.MAJOR)
        if major:
            return "fix-major"

        minor = review.get_issues_by_severity(ReviewSeverity.MINOR)
        if minor:
            return "fix-minor"

        suggestions = review.get_issues_by_severity(ReviewSeverity.SUGGESTION)
        if suggestions:
            return "review-suggestions"

        return "ready"

    @staticmethod
    def compute_actionability(issue: ReviewIssue) -> float:
        """
        Pure function: Re-compute actionability for an issue.

        Uses heuristics from ReviewBuilder._compute_actionability.
        """
        return ReviewBuilder._compute_actionability(issue.suggested_fix, issue.mechanism)


# =============================================================================
# DUAL REVIEW PROTOCOL
# =============================================================================


class DualReviewProtocol:
    """
    Protocol for dual review (Gemini 2.5 Pro + Codex).

    IMPORTANT: This class provides methods to PREPARE prompts and PARSE responses,
    but does NOT directly call MCP tools. The actual MCP calls must be done by
    the calling code (Claude).

    Why: Claude (the orchestrator) needs to make the MCP calls with identical prompts
    to ensure independent reviews.
    """

    @staticmethod
    def prepare_review_prompt(
        object_type: str,
        object_content: str,
        review_focus: str | None = None,
    ) -> str:
        """
        Pure function: Prepare review prompt for LLMs.

        This prompt will be sent IDENTICALLY to both Gemini and Codex
        to ensure independent reviews.

        Args:
            object_type: Type of object (ProofBox, ProofStep, etc.)
            object_content: Serialized object content (markdown, JSON, etc.)
            review_focus: Optional focus area (rigor, completeness, etc.)

        Returns:
            Prompt string for LLMs
        """
        focus_instruction = f"\nFocus particularly on: {review_focus}" if review_focus else ""

        return f"""Review this {object_type} for mathematical rigor and correctness.{focus_instruction}

{object_content}

Provide your review in the following format:

## Overall Assessment
[1-3 paragraphs summarizing your assessment]

## Issues
[List each issue in the following format:]

1. [SEVERITY] Title
   Location: [where in the object]
   Problem: [what is wrong]
   Mechanism: [why this fails - precise mechanism]
   Evidence: [counterexample, quote, or calculation]
   Impact: [what is affected]
   Fix: [specific suggestion for how to fix]

[Severity levels: CRITICAL, MAJOR, MINOR, SUGGESTION, VALIDATION_FAILURE, FRAMEWORK_INCONSISTENCY]

## Strengths
- [positive aspect 1]
- [positive aspect 2]
...

## Scores
Rigor: X/10
Soundness: X/10
Completeness: X/10
Clarity: X/10
Framework Consistency: X/10

---

**IMPORTANT**: Be specific about locations (step IDs, line numbers, equations). Explain WHY each issue is a problem, not just WHAT is wrong.
"""

    @staticmethod
    def parse_dual_review_responses(
        gemini_response: str,
        codex_response: str,
        object_id: str,
        object_type: str,
        iteration: int,
    ) -> ReviewComparison:
        """
        Pure function: Parse responses from both reviewers and create comparison.

        This should be called AFTER getting responses from both Gemini and Codex
        via MCP calls.

        Args:
            gemini_response: Response from Gemini 2.5 Pro
            codex_response: Response from Codex
            object_id: ID of object reviewed
            object_type: Type of object
            iteration: Iteration number

        Returns:
            ReviewComparison with consensus, discrepancies, and unique issues
        """
        # Build reviews from responses
        gemini_review = ReviewBuilder.from_llm_response(
            gemini_response,
            object_id,
            object_type,
            iteration,
            ReviewSource.GEMINI_2_5_PRO,
        )

        codex_review = ReviewBuilder.from_llm_response(
            codex_response,
            object_id,
            object_type,
            iteration,
            ReviewSource.CODEX,
        )

        # Add reviews to registry
        registry = get_review_registry()
        registry.add_review(gemini_review)
        registry.add_review(codex_review)

        # Create comparison
        comparison = registry.compare_reviews(gemini_review.review_id, codex_review.review_id)

        if comparison is None:
            msg = "Failed to create review comparison"
            raise ValueError(msg)

        return comparison

    @staticmethod
    def synthesize_consensus(comparison: ReviewComparison, iteration: int) -> Review:
        """
        Pure function: Synthesize unified review from dual review comparison.

        Creates a consensus review that:
        - Includes all high-confidence issues (consensus)
        - Includes medium-confidence issues (unique) with lower weight
        - Excludes discrepancies (requires manual verification)
        - Averages scores

        Args:
            comparison: ReviewComparison from dual review
            iteration: Iteration number for synthesized review

        Returns:
            Synthesized Review combining both reviews
        """
        gemini = comparison.review_a
        codex = comparison.review_b

        # Combine issues with confidence weighting
        issues = []

        # High confidence: consensus issues
        issues.extend(comparison.consensus_issues)

        # Medium confidence: unique issues (mark as SUGGESTION if not blocking)
        for issue in comparison.unique_to_a + comparison.unique_to_b:
            # Downgrade severity for unique issues (unless CRITICAL)
            if issue.severity == ReviewSeverity.CRITICAL:
                issues.append(issue)
            elif issue.severity == ReviewSeverity.MAJOR:
                # Downgrade to MINOR
                issues.append(
                    ReviewIssue(
                        id=issue.id,
                        severity=ReviewSeverity.MINOR,
                        location=issue.location,
                        title=issue.title + " [unique finding]",
                        problem=issue.problem,
                        mechanism=issue.mechanism,
                        evidence=issue.evidence,
                        impact=issue.impact,
                        suggested_fix=issue.suggested_fix,
                        actionability_score=issue.actionability_score * 0.5,  # Lower confidence
                        framework_references=issue.framework_references,
                        validation_failure=issue.validation_failure,
                    )
                )
            else:
                issues.append(issue)

        # Combine strengths
        strengths = list(set(gemini.strengths + codex.strengths))

        # Average scores
        avg_scores = {
            "rigor": (gemini.rigor_score + codex.rigor_score) / 2,
            "soundness": (gemini.soundness_score + codex.soundness_score) / 2,
            "completeness": (gemini.completeness_score + codex.completeness_score) / 2,
            "clarity": (gemini.clarity_score + codex.clarity_score) / 2,
            "framework_consistency": (
                gemini.framework_consistency_score + codex.framework_consistency_score
            )
            / 2,
        }

        # Compute metrics
        mean_actionability = (
            sum(i.actionability_score for i in issues) / len(issues) if issues else 1.0
        )
        blocking_count = sum(1 for i in issues if i.is_blocking())

        # Overall assessment
        agreement_note = (
            "Reviewers agree on major points."
            if comparison.reviewers_agree()
            else f"Reviewers have {len(comparison.discrepancies)} discrepancies requiring manual verification."
        )
        overall = f"Synthesized from dual review (Gemini + Codex). {agreement_note}\n\nGemini: {gemini.overall_assessment}\n\nCodex: {codex.overall_assessment}"

        review_id = f"review-{gemini.object_id}-iter{iteration}-synthesis"

        return Review(
            review_id=review_id,
            object_id=gemini.object_id,
            object_type=gemini.object_type,
            iteration=iteration,
            timestamp=datetime.now(),
            source=ReviewSource.CLAUDE_SYNTHESIS,
            issues=issues,
            strengths=strengths,
            overall_assessment=overall,
            rigor_score=avg_scores["rigor"],
            soundness_score=avg_scores["soundness"],
            completeness_score=avg_scores["completeness"],
            clarity_score=avg_scores["clarity"],
            framework_consistency_score=avg_scores["framework_consistency"],
            mean_actionability=mean_actionability,
            blocking_issue_count=blocking_count,
            validation_results=[],
            validation_passed=True,
            previous_review_id=None,
            addresses_issues=[],
            llm_responses=[],
        )
