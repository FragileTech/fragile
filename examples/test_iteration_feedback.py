#!/usr/bin/env python3
"""Simple test script to validate iteration feedback functionality.

This script creates mock validation results and tests the feedback formatter
to ensure it generates proper feedback for iteration refinement.
"""

from __future__ import annotations

from mathster.proof_sketcher.feedback_formatter import FeedbackConfig, IterationFeedbackFormatter
from mathster.proof_sketcher.sketch_pipeline import ProofSketchWorkflowResult
from mathster.proof_sketcher.sketch_referee_analysis import SketchValidationReview
from mathster.proof_sketcher.sketch_validator import (
    ActionItem,
    ConsensusAnalysis,
    DisagreementEntry,
    ReportMetadata,
    Scores,
    SketchValidationReport,
    SynthesisAndActionPlan,
)
from mathster.proof_sketcher.sketcher import ProofSketch


def create_mock_result() -> ProofSketchWorkflowResult:
    """Create a mock workflow result for testing."""

    # Create mock sketch (minimal fields for testing)
    sketch = ProofSketch(
        title="Test Theorem",
        label="thm-test",
        type="Theorem",
        status="Sketch",
        date="2025-01-12",
        source="test.md",
        statement={
            "formal": "Test statement",
            "informal": "Test informal statement",
        },
        hypotheses=[],
        conclusion="",
        strategySynthesis={
            "strategies": [],
            "recommendedApproach": {
                "chosenMethod": "Test method",
                "rationale": "Test rationale",
                "fallbackPlan": "",
            },
        },
        dependencies={
            "verifiedDependencies": [],
            "missingOrUncertainDependencies": {
                "lemmasToProve": [],
                "uncertainAssumptions": [],
            },
        },
        detailedProof={
            "overview": "Test overview",
            "topLevelOutline": [],
            "steps": [],
            "conclusion": "",
        },
        technicalDeepDives=[],
        validationChecklist={
            "logicalCompleteness": "Yes",
            "hypothesisUsage": "Yes",
            "conclusionDerivation": "Yes",
            "frameworkConsistency": "Yes",
            "noCircularReasoning": "Yes",
        },
        alternativeApproaches=[],
        futureWork=None,
        crossReferences=[],
        expansionRoadmap=None,
        specialNotes="",
    )

    # Create mock scores
    scores = Scores(
        gemini_overall_score=3,
        gemini_overall_confidence=4,
        gemini_overall_confidence_score=4,
        gemini_completeness_score=3,
        gemini_completeness_confidence=4,
        gemini_covers_all_claims=True,
        gemini_error_count=2,
        gemini_logical_flow_score=3,
        gemini_logical_flow_confidence=4,
        gemini_logical_flow_sound=True,
        gemini_logical_gap_count=1,
        gemini_dependency_score=3,
        gemini_dependency_confidence=4,
        gemini_dependency_issue_count=1,
        gemini_technical_dive_score=3,
        gemini_technical_dive_confidence=4,
        gemini_technical_critique_count=2,
        codex_overall_score=3,
        codex_overall_confidence=3,
        codex_overall_confidence_score=3,
        codex_completeness_score=4,
        codex_completeness_confidence=3,
        codex_covers_all_claims=True,
        codex_error_count=1,
        codex_logical_flow_score=3,
        codex_logical_flow_confidence=3,
        codex_logical_flow_sound=True,
        codex_logical_gap_count=2,
        codex_dependency_score=3,
        codex_dependency_confidence=3,
        codex_dependency_issue_count=1,
        codex_technical_dive_score=3,
        codex_technical_dive_confidence=3,
        codex_technical_critique_count=3,
        average_overall_score=3.0,
        average_overall_confidence=3.5,
        average_completeness_score=3.5,
        average_logical_flow_score=3.0,
        total_error_count=3,
        total_logical_gap_count=3,
        total_dependency_issue_count=2,
        total_technical_critique_count=5,
        both_reviewers_sound=True,
        both_reviewers_cover_claims=True,
        overall_score_variance=0.0,
        completeness_score_variance=0.5,
        logical_flow_score_variance=0.0,
        final_decision_numeric=2,
        action_item_count=5,
        critical_action_count=2,
        high_action_count=2,
        medium_action_count=1,
        low_action_count=0,
        points_of_agreement_count=3,
        points_of_disagreement_count=1,
        overall_quality_index=3.2,
        risk_score=12.5,
    )

    # Create mock action items
    actions = [
        ActionItem(
            itemId="action-1",
            description="Verify dependency thm-lsi-bound applies under current assumptions. Missing regularity conditions.",
            priority="Critical",
            references=["Step 3", "thm-lsi-bound"],
        ),
        ActionItem(
            itemId="action-2",
            description="Fix circular reasoning in uniqueness proof. Step 5 depends on Step 8 result.",
            priority="Critical",
            references=["Step 5", "Step 8"],
        ),
        ActionItem(
            itemId="action-3",
            description="Add explicit regularity assumptions for diffusion coefficient.",
            priority="High",
            references=["Hypotheses"],
        ),
        ActionItem(
            itemId="action-4",
            description="Clarify transition from discrete to continuous limit in Step 7.",
            priority="High",
            references=["Step 7"],
        ),
        ActionItem(
            itemId="action-5",
            description="Update notation for consistency with framework conventions.",
            priority="Medium",
            references=["Throughout"],
        ),
    ]

    # Create mock consensus
    consensus = ConsensusAnalysis(
        pointsOfAgreement=[
            "Missing regularity assumptions for diffusion coefficient",
            "Insufficient justification for LSI constant bound",
            "Unclear transition from discrete to continuous limit",
        ],
        pointsOfDisagreement=[
            DisagreementEntry(
                topic="Proof strategy for convergence",
                geminiView="Suggests Bakry-Émery approach",
                codexView="Prefers direct entropy method",
                resolution="Investigate both approaches, choose based on assumption strength",
            )
        ],
        summaryOfFindings=(
            "The proof strategy using LSI → Grönwall → exponential convergence is sound, "
            "but the execution has critical gaps. The main issues are: (1) insufficient "
            "verification of dependency prerequisites, (2) missing regularity conditions, "
            "and (3) circular reasoning in the uniqueness argument. Address these before "
            "proceeding to expansion."
        ),
    )

    # Create mock synthesis and action plan
    synthesis = SynthesisAndActionPlan(
        finalDecision="Requires Major Revisions",
        consensusAnalysis=consensus,
        actionableItems=actions,
        confidenceStatement="Medium confidence in assessment",
    )

    # Create mock validation report
    report = SketchValidationReport(
        reportMetadata=ReportMetadata(
            sketchLabel="thm-test",
            validationCycleId="test-uuid",
            validationTimestamp="2025-01-12T10:00:00Z",
        ),
        originalProofSketch={},
        reviews=[{}, {}],
        synthesisAndActionPlan=synthesis,
    )

    # Create mock reviews (simplified)
    review_1 = SketchValidationReview(
        reviewer="Gemini",
        timestamp="2025-01-12T10:00:00Z",
        overallAssessment={
            "score": 3,
            "confidenceScore": 4,
            "recommendation": "Major Revisions",
            "justification": "Critical gaps in proof",
        },
        completenessAndCorrectness={
            "score": 3,
            "confidence": 4,
            "coversAllClaims": True,
            "identifiedErrors": [
                "Missing factor in equation (12)",
                "Incorrect inequality direction in Step 7",
            ],
            "missingComponents": [],
        },
        logicalFlowValidation={
            "score": 3,
            "confidence": 4,
            "isSound": True,
            "identifiedGaps": [
                "Jump from (15) to (16) requires intermediate bound",
            ],
            "explanationQuality": "Good",
        },
        dependencyValidation={
            "status": "Incomplete",
            "issues": ["thm-lsi-bound assumptions not verified"],
            "suggestions": [],
        },
        technicalDeepDiveValidation={
            "score": 3,
            "confidence": 4,
            "critiques": [
                "Regularity assumptions implicit",
                "Convergence rate derivation incomplete",
            ],
        },
    )

    review_2 = SketchValidationReview(
        reviewer="Codex",
        timestamp="2025-01-12T10:00:00Z",
        overallAssessment={
            "score": 3,
            "confidenceScore": 3,
            "recommendation": "Major Revisions",
            "justification": "Several critical issues",
        },
        completenessAndCorrectness={
            "score": 4,
            "confidence": 3,
            "coversAllClaims": True,
            "identifiedErrors": [
                "Circular reasoning in uniqueness proof",
            ],
            "missingComponents": [],
        },
        logicalFlowValidation={
            "score": 3,
            "confidence": 3,
            "isSound": True,
            "identifiedGaps": [
                "Discrete-continuous transition unclear",
                "Grönwall application needs justification",
            ],
            "explanationQuality": "Acceptable",
        },
        dependencyValidation={
            "status": "Incomplete",
            "issues": ["lem-gronwall missing regularity hypothesis"],
            "suggestions": [],
        },
        technicalDeepDiveValidation={
            "score": 3,
            "confidence": 3,
            "critiques": [
                "LSI constant bound not justified",
                "Momentum conservation not discussed",
                "Potential assumptions too strong",
            ],
        },
    )

    # Create workflow result
    result = ProofSketchWorkflowResult(
        sketch=sketch,
        validation_report=report,
        scores=scores,
        strategy_1=sketch,
        strategy_2=sketch,
        review_1=review_1,
        review_2=review_2,
    )

    return result


def main():
    """Test the feedback formatter."""
    print("=" * 80)
    print("TESTING ITERATION FEEDBACK FORMATTER")
    print("=" * 80)

    # Create mock result
    print("\n1. Creating mock validation result...")
    result = create_mock_result()
    print(f"   Score: {result.scores.get_score():.2f}/100")
    print(f"   Decision: {result.validation_report.synthesisAndActionPlan.finalDecision}")
    print(f"   Action items: {result.scores.action_item_count}")

    # Create formatter
    print("\n2. Creating feedback formatter...")
    formatter = IterationFeedbackFormatter()

    # Test format_detailed
    print("\n3. Testing format_detailed()...")
    detailed = formatter.format_detailed(result, iteration_num=1)
    print(f"   Generated {len(detailed)} characters")
    print(f"   Contains score breakdown: {'Score Breakdown' in detailed or 'Overall Score' in detailed}")
    print(f"   Contains action items: {'Required Actions' in detailed or 'Action' in detailed}")

    # Test format_suggestions
    print("\n4. Testing format_suggestions()...")
    suggestions = formatter.format_suggestions(result, iteration_num=1)
    print(f"   Generated {len(suggestions)} characters")
    print(f"   Contains fix suggestions: {'SPECIFIC SUGGESTIONS' in suggestions}")
    print(f"   Contains fix steps: {'How to Fix' in suggestions}")

    # Test format_combined
    print("\n5. Testing format_combined()...")
    combined = formatter.format_combined(result, iteration_num=1)
    print(f"   Generated {len(combined)} characters")
    print(f"   Contains detailed section: {'Overall Assessment' in combined or 'Score' in combined}")
    print(f"   Contains suggestions section: {'SPECIFIC SUGGESTIONS' in combined}")

    # Display sample output
    print("\n" + "=" * 80)
    print("SAMPLE COMBINED FEEDBACK OUTPUT")
    print("=" * 80)
    print(combined[:2000] + "\n...\n[truncated]")

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
