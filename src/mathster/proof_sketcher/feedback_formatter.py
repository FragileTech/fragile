#!/usr/bin/env python3
"""Format validation feedback for iterative proof sketch refinement.

This module provides utilities to extract, format, and structure feedback from
validation reports for injection into subsequent refinement iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mathster.proof_sketcher.sketch_pipeline import ProofSketchWorkflowResult
from mathster.proof_sketcher.sketch_referee_analysis import SketchValidationReview
from mathster.proof_sketcher.sketch_validator import ActionItem, Scores, SketchValidationReport


__all__ = ["FeedbackConfig", "IterationFeedbackFormatter"]


@dataclass
class FeedbackConfig:
    """Configuration for iteration feedback formatting.

    Controls which sections to include in formatted feedback and limits
    on the amount of detail to provide.
    """

    # Section toggles
    include_score_breakdown: bool = True
    """Include detailed score component breakdown."""

    include_decision: bool = True
    """Include final validation decision."""

    include_action_items: bool = True
    """Include actionable TODO items."""

    include_consensus: bool = True
    """Include points of agreement/disagreement."""

    include_errors: bool = True
    """Include specific mathematical errors."""

    include_gaps: bool = True
    """Include logical gaps."""

    include_dependency_issues: bool = True
    """Include dependency validation issues."""

    include_technical_critiques: bool = True
    """Include technical deep dive critiques."""

    include_summary: bool = True
    """Include narrative summary of findings."""

    include_fix_suggestions: bool = True
    """Include specific suggestions section."""

    # Limits
    max_actions: int = 10
    """Maximum number of action items to include (prioritized)."""

    max_errors: int = 8
    """Maximum number of errors to include."""

    max_gaps: int = 6
    """Maximum number of logical gaps to include."""

    max_dependency_issues: int = 5
    """Maximum number of dependency issues to include."""

    max_technical_critiques: int = 5
    """Maximum number of technical critiques to include."""

    # Priority filters
    action_priorities: list[str] = field(
        default_factory=lambda: ["Critical", "High", "Medium", "Low"]
    )
    """Action item priorities to include (in order)."""


class IterationFeedbackFormatter:
    """Format validation feedback for injection into next iteration.

    This class extracts relevant feedback from validation reports and formats
    it for consumption by the proof sketching pipeline in subsequent iterations.

    Example:
        >>> formatter = IterationFeedbackFormatter()
        >>> feedback = formatter.format_combined(result, iteration_num=1)
        >>> # Inject feedback into next iteration kwargs
        >>> next_kwargs["framework_context"] += f"\\n\\n{feedback}"
    """

    def __init__(self, config: FeedbackConfig | None = None) -> None:
        """Initialize formatter with optional configuration.

        Args:
            config: Feedback configuration. If None, uses default config.
        """
        self.config = config or FeedbackConfig()

    def format_combined(
        self,
        result: ProofSketchWorkflowResult,
        iteration_num: int,
    ) -> str:
        """Format combined detailed feedback + specific fix suggestions.

        This is the main entry point for generating feedback. It combines
        comprehensive context (detailed format) with actionable guidance
        (suggestions format).

        Args:
            result: Workflow result from previous iteration
            iteration_num: Iteration number (1-indexed)

        Returns:
            Formatted feedback string (markdown)
        """
        sections = []

        # Header
        sections.append(f"## Iteration {iteration_num} Detailed Feedback\n")

        # Section 1: Detailed feedback (context)
        detailed = self.format_detailed(result, iteration_num)
        sections.append(detailed)

        # Section 2: Specific fix suggestions (actionable)
        if self.config.include_fix_suggestions:
            sections.append("\n---\n")
            suggestions = self.format_suggestions(result, iteration_num)
            sections.append(suggestions)

        return "\n".join(sections)

    def format_detailed(
        self,
        result: ProofSketchWorkflowResult,
        iteration_num: int,
    ) -> str:
        """Format detailed comprehensive feedback (500-1000 words).

        Includes all major feedback components: scores, decision, actions,
        consensus, issues, and summary.

        Args:
            result: Workflow result from previous iteration
            iteration_num: Iteration number (1-indexed)

        Returns:
            Formatted detailed feedback (markdown)
        """
        sections = []

        report = result.validation_report
        scores = result.scores

        # Overall Assessment
        if self.config.include_score_breakdown or self.config.include_decision:
            sections.append("### Overall Assessment\n")

            if self.config.include_score_breakdown:
                score_summary = self._format_score_summary(scores)
                sections.append(score_summary)

            if self.config.include_decision:
                decision = report.synthesisAndActionPlan.finalDecision
                sections.append(f"- **Decision:** {decision}")

            sections.append("")  # Blank line

        # Action Items (Prioritized)
        if self.config.include_action_items:
            actions_section = self._format_action_items(report)
            if actions_section:
                sections.append(actions_section)
                sections.append("")

        # Reviewer Consensus
        if self.config.include_consensus:
            consensus_section = self._format_consensus(report)
            if consensus_section:
                sections.append(consensus_section)
                sections.append("")

        # Specific Issues
        issues_section = self._format_specific_issues(result)
        if issues_section:
            sections.append(issues_section)
            sections.append("")

        # Summary of Findings
        if self.config.include_summary:
            summary = report.synthesisAndActionPlan.consensusAnalysis.summaryOfFindings
            if summary:
                sections.append("### Summary of Findings\n")
                sections.append(summary)
                sections.append("")

        return "\n".join(sections)

    def format_suggestions(
        self,
        result: ProofSketchWorkflowResult,
        iteration_num: int,
    ) -> str:
        """Format specific actionable fix suggestions.

        Extracts concrete remediation steps from reviews and action items,
        organizing them by priority and providing detailed "how to fix" guidance.

        Args:
            result: Workflow result from previous iteration
            iteration_num: Iteration number (1-indexed)

        Returns:
            Formatted suggestions section (markdown)
        """
        sections = []
        sections.append("## SPECIFIC SUGGESTIONS TO FIX ISSUES\n")

        report = result.validation_report
        actions = self._prioritize_actions(report.synthesisAndActionPlan.actionableItems)

        # Extract fix suggestions from action items and reviews
        suggestions = self._extract_fix_suggestions(actions, result)

        if not suggestions:
            sections.append("*No specific fix suggestions could be extracted from the reviews.*\n")
            return "\n".join(sections)

        # Format each suggestion
        for i, suggestion in enumerate(suggestions, 1):
            sections.append(f"### Fix #{i}: {suggestion['title']} ({suggestion['priority']})\n")
            sections.append(f"**Problem:** {suggestion['problem']}\n")
            sections.append(f"**How to Fix:**\n{suggestion['fix_steps']}\n")

            if suggestion.get("references"):
                sections.append(f"**References:** {suggestion['references']}\n")

            sections.append("")  # Blank line between suggestions

        return "\n".join(sections)

    def _format_score_summary(self, scores: Scores) -> str:
        """Format score breakdown summary."""
        lines = []
        lines.append(f"- **Overall Score:** {scores.get_score():.2f}/100")
        lines.append(
            f"- **Gemini Score:** {scores.gemini_overall_score}/5 "
            f"(Confidence: {scores.gemini_overall_confidence}/5)"
        )
        lines.append(
            f"- **Codex Score:** {scores.codex_overall_score}/5 "
            f"(Confidence: {scores.codex_overall_confidence}/5)"
        )

        # Component scores
        lines.append("\n**Score Breakdown:**")
        lines.append(
            f"- Completeness: Gemini={scores.gemini_completeness_score}/5, "
            f"Codex={scores.codex_completeness_score}/5, "
            f"Avg={scores.average_completeness_score:.1f}/5"
        )
        lines.append(
            f"- Logical Flow: Gemini={scores.gemini_logical_flow_score}/5, "
            f"Codex={scores.codex_logical_flow_score}/5, "
            f"Avg={scores.average_logical_flow_score:.1f}/5"
        )
        lines.append(
            f"- Confidence: Gemini={scores.gemini_overall_confidence}/5, "
            f"Codex={scores.codex_overall_confidence}/5, "
            f"Avg={scores.average_overall_confidence:.1f}/5"
        )

        return "\n".join(lines)

    def _format_action_items(self, report: SketchValidationReport) -> str:
        """Format prioritized action items."""
        actions = self._prioritize_actions(report.synthesisAndActionPlan.actionableItems)

        if not actions:
            return ""

        lines = []
        lines.append("### Required Actions\n")

        # Group by priority
        for priority in self.config.action_priorities:
            priority_actions = [a for a in actions if a.priority == priority]
            if not priority_actions:
                continue

            lines.append(f"#### {priority} Priority ({len(priority_actions)} items)\n")

            for i, action in enumerate(priority_actions, 1):
                refs = f" [REF: {', '.join(action.references)}]" if action.references else ""
                lines.append(f"{i}. **{refs}** {action.description}")

            lines.append("")  # Blank line between priorities

        return "\n".join(lines)

    def _format_consensus(self, report: SketchValidationReport) -> str:
        """Format reviewer consensus analysis."""
        consensus = report.synthesisAndActionPlan.consensusAnalysis

        if not consensus.pointsOfAgreement and not consensus.pointsOfDisagreement:
            return ""

        lines = []
        lines.append("### Reviewer Consensus\n")

        # Points of Agreement
        if consensus.pointsOfAgreement:
            lines.append("**Points of Agreement (Both Reviewers):**")
            for point in consensus.pointsOfAgreement:
                lines.append(f"- {point}")
            lines.append("")

        # Points of Disagreement
        if consensus.pointsOfDisagreement:
            lines.append("**Points of Disagreement:**")
            for disagreement in consensus.pointsOfDisagreement:
                lines.append(f"- **Topic:** {disagreement.topic}")
                lines.append(f"  - Gemini: {disagreement.geminiView}")
                lines.append(f"  - Codex: {disagreement.codexView}")
                lines.append(f"  - Resolution: {disagreement.resolution}")
            lines.append("")

        return "\n".join(lines)

    def _format_specific_issues(self, result: ProofSketchWorkflowResult) -> str:
        """Format specific errors, gaps, and issues."""
        lines = []
        has_content = False

        # Errors
        if self.config.include_errors:
            errors = self._extract_errors(result)
            if errors:
                has_content = True
                lines.append(f"### Mathematical Errors ({len(errors)} total)\n")
                for error in errors[: self.config.max_errors]:
                    lines.append(f"- [{error['reviewer']}] {error['description']}")
                if len(errors) > self.config.max_errors:
                    lines.append(f"- *... and {len(errors) - self.config.max_errors} more*")
                lines.append("")

        # Logical Gaps
        if self.config.include_gaps:
            gaps = self._extract_gaps(result)
            if gaps:
                has_content = True
                lines.append(f"### Logical Gaps ({len(gaps)} total)\n")
                for gap in gaps[: self.config.max_gaps]:
                    lines.append(f"- [{gap['reviewer']}] {gap['description']}")
                if len(gaps) > self.config.max_gaps:
                    lines.append(f"- *... and {len(gaps) - self.config.max_gaps} more*")
                lines.append("")

        # Dependency Issues
        if self.config.include_dependency_issues:
            dep_issues = self._extract_dependency_issues(result)
            if dep_issues:
                has_content = True
                lines.append(f"### Dependency Issues ({len(dep_issues)} total)\n")
                for issue in dep_issues[: self.config.max_dependency_issues]:
                    lines.append(f"- [{issue['reviewer']}] {issue['description']}")
                if len(dep_issues) > self.config.max_dependency_issues:
                    lines.append(
                        f"- *... and {len(dep_issues) - self.config.max_dependency_issues} more*"
                    )
                lines.append("")

        # Technical Critiques
        if self.config.include_technical_critiques:
            critiques = self._extract_technical_critiques(result)
            if critiques:
                has_content = True
                lines.append(f"### Technical Critiques ({len(critiques)} total)\n")
                for critique in critiques[: self.config.max_technical_critiques]:
                    lines.append(f"- [{critique['reviewer']}] {critique['description']}")
                if len(critiques) > self.config.max_technical_critiques:
                    lines.append(
                        f"- *... and {len(critiques) - self.config.max_technical_critiques} more*"
                    )
                lines.append("")

        if not has_content:
            return ""

        return "### Specific Issues\n\n" + "\n".join(lines)

    def _prioritize_actions(self, actions: list[ActionItem]) -> list[ActionItem]:
        """Sort actions by priority and apply limit."""
        # Define priority order
        priority_order = {p: i for i, p in enumerate(self.config.action_priorities)}

        # Sort by priority
        sorted_actions = sorted(
            actions,
            key=lambda a: priority_order.get(a.priority, 999),
        )

        # Apply limit
        return sorted_actions[: self.config.max_actions]

    def _extract_errors(self, result: ProofSketchWorkflowResult) -> list[dict[str, str]]:
        """Extract mathematical errors from both reviews."""
        errors = []

        # Gemini errors
        gemini_review = result.review_1
        if isinstance(gemini_review, dict):
            cc = gemini_review.get("completenessAndCorrectness", {})
            for error in cc.get("identifiedErrors", []):
                errors.append({"reviewer": "Gemini", "description": error})
        else:
            for error in gemini_review.completenessAndCorrectness.identifiedErrors:
                errors.append({"reviewer": "Gemini", "description": error})

        # Codex errors
        codex_review = result.review_2
        if isinstance(codex_review, dict):
            cc = codex_review.get("completenessAndCorrectness", {})
            for error in cc.get("identifiedErrors", []):
                errors.append({"reviewer": "Codex", "description": error})
        else:
            for error in codex_review.completenessAndCorrectness.identifiedErrors:
                errors.append({"reviewer": "Codex", "description": error})

        return errors

    def _extract_gaps(self, result: ProofSketchWorkflowResult) -> list[dict[str, str]]:
        """Extract logical gaps from both reviews."""
        gaps = []

        # Gemini gaps
        gemini_review = result.review_1
        if isinstance(gemini_review, dict):
            lf = gemini_review.get("logicalFlowValidation", {})
            for gap in lf.get("identifiedGaps", []):
                gaps.append({"reviewer": "Gemini", "description": gap})
        else:
            for gap in gemini_review.logicalFlowValidation.identifiedGaps:
                gaps.append({"reviewer": "Gemini", "description": gap})

        # Codex gaps
        codex_review = result.review_2
        if isinstance(codex_review, dict):
            lf = codex_review.get("logicalFlowValidation", {})
            for gap in lf.get("identifiedGaps", []):
                gaps.append({"reviewer": "Codex", "description": gap})
        else:
            for gap in codex_review.logicalFlowValidation.identifiedGaps:
                gaps.append({"reviewer": "Codex", "description": gap})

        return gaps

    def _extract_dependency_issues(self, result: ProofSketchWorkflowResult) -> list[dict[str, str]]:
        """Extract dependency validation issues from both reviews."""
        issues = []

        # Gemini issues
        gemini_review = result.review_1
        if isinstance(gemini_review, dict):
            dv = gemini_review.get("dependencyValidation", {})
            for issue in dv.get("issues", []):
                issues.append({"reviewer": "Gemini", "description": issue})
        else:
            for issue in gemini_review.dependencyValidation.issues:
                issues.append({"reviewer": "Gemini", "description": issue})

        # Codex issues
        codex_review = result.review_2
        if isinstance(codex_review, dict):
            dv = codex_review.get("dependencyValidation", {})
            for issue in dv.get("issues", []):
                issues.append({"reviewer": "Codex", "description": issue})
        else:
            for issue in codex_review.dependencyValidation.issues:
                issues.append({"reviewer": "Codex", "description": issue})

        return issues

    def _extract_technical_critiques(
        self, result: ProofSketchWorkflowResult
    ) -> list[dict[str, str]]:
        """Extract technical critiques from both reviews."""
        critiques = []

        # Gemini critiques
        gemini_review = result.review_1
        if isinstance(gemini_review, dict):
            td = gemini_review.get("technicalDeepDiveValidation", {})
            for critique in td.get("critiques", []):
                critiques.append({"reviewer": "Gemini", "description": critique})
        else:
            for critique in gemini_review.technicalDeepDiveValidation.critiques:
                critiques.append({"reviewer": "Gemini", "description": critique})

        # Codex critiques
        codex_review = result.review_2
        if isinstance(codex_review, dict):
            td = codex_review.get("technicalDeepDiveValidation", {})
            for critique in td.get("critiques", []):
                critiques.append({"reviewer": "Codex", "description": critique})
        else:
            for critique in codex_review.technicalDeepDiveValidation.critiques:
                critiques.append({"reviewer": "Codex", "description": critique})

        return critiques

    def _extract_fix_suggestions(
        self,
        actions: list[ActionItem],
        result: ProofSketchWorkflowResult,
    ) -> list[dict[str, Any]]:
        """Extract and format specific fix suggestions from actions and reviews.

        Args:
            actions: Prioritized action items
            result: Workflow result with full reviews

        Returns:
            List of formatted suggestions with problem/fix/references
        """
        suggestions = []

        # Focus on Critical and High priority actions
        critical_actions = [a for a in actions if a.priority in ["Critical", "High"]]

        for action in critical_actions[: 5]:  # Limit to top 5
            suggestion = self._format_single_suggestion(action, result)
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    def _format_single_suggestion(
        self,
        action: ActionItem,
        result: ProofSketchWorkflowResult,
    ) -> dict[str, Any] | None:
        """Format a single action item into a detailed suggestion.

        Args:
            action: Action item to format
            result: Workflow result for context

        Returns:
            Formatted suggestion dict or None if cannot be formatted
        """
        # Extract title from action description (first sentence or up to 80 chars)
        title = action.description.split(".")[0]
        if len(title) > 80:
            title = title[:77] + "..."

        # Build fix steps (look for detailed guidance in action description)
        fix_steps = self._generate_fix_steps(action, result)

        return {
            "title": title,
            "priority": action.priority,
            "problem": action.description,
            "fix_steps": fix_steps,
            "references": ", ".join(action.references) if action.references else None,
        }

    def _generate_fix_steps(
        self,
        action: ActionItem,
        result: ProofSketchWorkflowResult,
    ) -> str:
        """Generate concrete fix steps for an action item.

        Args:
            action: Action item
            result: Workflow result for additional context

        Returns:
            Formatted fix steps (markdown list)
        """
        # This is a heuristic approach - in practice, the action description
        # should contain enough detail to generate steps. We can enhance this
        # with LLM-based expansion if needed.

        steps = []

        # Generic remediation based on common patterns
        if "dependency" in action.description.lower() or "thm-" in action.description:
            steps.append("1. Review the referenced theorem/definition statement")
            steps.append("2. Identify assumptions required for its application")
            steps.append(
                "3. Either: (A) Add missing assumptions to hypotheses, "
                "(B) Prove them as preliminary lemma, or (C) Use alternative result"
            )
            steps.append("4. Add explicit verification in proof text")

        elif "circular" in action.description.lower() or "loop" in action.description.lower():
            steps.append("1. Identify the dependency cycle in proof steps")
            steps.append("2. Restructure proof order to break the cycle")
            steps.append(
                "3. Consider: (A) Proving components independently, "
                "(B) Adding intermediate lemma, or (C) Using different approach"
            )

        elif (
            "regularity" in action.description.lower() or "assumption" in action.description.lower()
        ):
            steps.append("1. Identify where smoothness/regularity is implicitly used")
            steps.append("2. Add explicit assumption to theorem hypotheses")
            steps.append("3. Verify assumption is used correctly in all proof steps")

        elif "error" in action.description.lower() or "mistake" in action.description.lower():
            steps.append("1. Locate the specific error in the proof")
            steps.append("2. Verify the correct formula/statement")
            steps.append("3. Update the proof step with corrected version")
            steps.append("4. Check downstream steps for propagation")

        elif "gap" in action.description.lower() or "missing" in action.description.lower():
            steps.append("1. Identify what reasoning is missing between steps")
            steps.append("2. Add intermediate calculation or bound")
            steps.append("3. Provide explicit justification or reference")

        else:
            # Fallback: generic guidance
            steps.append("1. Carefully review the issue described in the problem statement")
            steps.append("2. Consult relevant framework documents for correct approach")
            steps.append("3. Revise the proof step(s) to address the issue")
            steps.append("4. Verify the fix doesn't introduce new problems")

        return "\n".join(steps)
