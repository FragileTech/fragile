"""Tests for validation request and validation report utilities."""

import json
import pytest
from pathlib import Path
from datetime import datetime

# Import validation functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mathster.agent_schemas.validate_sketch import (
    load_validation_request_schema,
    load_validation_report_schema,
    validate_validation_request,
    validate_validation_report,
    load_sketch_for_validation,
)


def test_load_validation_request_schema():
    """Test that validation request schema can be loaded."""
    schema = load_validation_request_schema()
    assert isinstance(schema, dict)
    assert "$schema" in schema
    assert "title" in schema
    assert schema["title"] == "Proof Sketch Validation Review"
    assert "required" in schema


def test_load_validation_report_schema():
    """Test that validation report schema can be loaded."""
    schema = load_validation_report_schema()
    assert isinstance(schema, dict)
    assert "$schema" in schema
    assert "title" in schema
    assert schema["title"] == "Proof Sketch Validation Cycle Report"
    assert "required" in schema


def test_valid_validation_request():
    """Test validation of a complete valid validation request."""
    review = {
        "reviewer": "Gemini 2.5 Pro",
        "timestamp": "2025-11-10T17:30:00Z",
        "overallAssessment": {
            "confidenceScore": "High (Ready for Expansion)",
            "summary": "The proof sketch is mathematically sound and ready for expansion.",
            "recommendation": "Proceed to Expansion"
        },
        "detailedAnalysis": {
            "logicalFlowValidation": {
                "isSound": True,
                "comments": "Logical progression is clear and rigorous",
                "identifiedGaps": []
            },
            "dependencyValidation": {
                "status": "Complete and Correct",
                "issues": []
            },
            "technicalDeepDiveValidation": {
                "critiques": [
                    {
                        "challengeTitle": "LSI Constant Uniformity",
                        "solutionViability": "Viable and Well-Described",
                        "critique": "Framework provides N-uniform LSI constant",
                        "suggestedImprovements": "Could add explicit theorem reference"
                    }
                ]
            },
            "completenessAndCorrectness": {
                "coversAllClaims": True,
                "identifiedErrors": []
            }
        }
    }

    is_valid, errors = validate_validation_request(review)
    assert is_valid, f"Validation failed: {errors}"
    assert errors == []


def test_validation_request_missing_required_fields():
    """Test that missing required fields are detected."""
    review = {
        "reviewer": "Gemini 2.5 Pro",
        "timestamp": "2025-11-10T17:30:00Z"
        # Missing: overallAssessment, detailedAnalysis
    }

    is_valid, errors = validate_validation_request(review)
    assert not is_valid
    assert len(errors) > 0


def test_validation_request_invalid_confidence_score():
    """Test that invalid confidence score enum values are rejected."""
    review = {
        "reviewer": "Gemini 2.5 Pro",
        "timestamp": "2025-11-10T17:30:00Z",
        "overallAssessment": {
            "confidenceScore": "VeryHigh",  # Invalid - not in enum
            "summary": "Good",
            "recommendation": "Proceed to Expansion"
        },
        "detailedAnalysis": {
            "logicalFlowValidation": {
                "isSound": True,
                "comments": "Good"
            },
            "dependencyValidation": {
                "status": "Complete and Correct"
            },
            "completenessAndCorrectness": {
                "coversAllClaims": True
            }
        }
    }

    is_valid, errors = validate_validation_request(review)
    assert not is_valid


def test_validation_request_with_dependency_issues():
    """Test validation request with dependency validation issues."""
    review = {
        "reviewer": "GPT-5 via Codex",
        "timestamp": "2025-11-10T17:30:00Z",
        "overallAssessment": {
            "confidenceScore": "Medium (Sound, but requires minor revisions)",
            "summary": "Strategy sound but LSI preconditions need verification",
            "recommendation": "Revise and Resubmit for Validation"
        },
        "detailedAnalysis": {
            "logicalFlowValidation": {
                "isSound": True,
                "comments": "Clear logical structure",
                "identifiedGaps": ["Step 2→3 needs explicit bound"]
            },
            "dependencyValidation": {
                "status": "Minor Issues Found",
                "issues": [
                    {
                        "label": "thm-lsi-main",
                        "issueType": "Preconditions Not Met",
                        "comment": "LSI constant N-uniformity not verified"
                    },
                    {
                        "label": "def-kl-divergence",
                        "issueType": "Missing Dependency",
                        "comment": "Should cite def-relative-entropy as well"
                    }
                ]
            },
            "technicalDeepDiveValidation": {},
            "completenessAndCorrectness": {
                "coversAllClaims": True,
                "identifiedErrors": [
                    {
                        "location": "Step 3.2",
                        "description": "Sign error in dissipation term",
                        "suggestedCorrection": "Change + to - in equation (3.2)"
                    }
                ]
            }
        }
    }

    is_valid, errors = validate_validation_request(review)
    assert is_valid, f"Validation failed: {errors}"


def test_valid_validation_report():
    """Test validation of a complete valid validation report."""
    report = {
        "reportMetadata": {
            "sketchLabel": "thm-example-convergence",
            "validationCycleId": "550e8400-e29b-41d4-a716-446655440000",
            "validationTimestamp": "2025-11-10T17:35:00Z"
        },
        "originalProofSketch": {
            "label": "thm-example-convergence",
            "entity_type": "theorem",
            "strategy": {
                "strategist": "Gemini 2.5 Pro",
                "method": "Lyapunov Method",
                "summary": "...",
                "keySteps": ["Step 1"],
                "strengths": ["Direct"],
                "weaknesses": ["Complex"],
                "frameworkDependencies": {
                    "theorems": [],
                    "lemmas": [],
                    "axioms": [],
                    "definitions": []
                },
                "confidenceScore": "High"
            }
        },
        "reviews": [
            {
                "reviewer": "Gemini 2.5 Pro",
                "timestamp": "2025-11-10T17:30:00Z",
                "overallAssessment": {
                    "confidenceScore": "High (Ready for Expansion)",
                    "summary": "Sound",
                    "recommendation": "Proceed to Expansion"
                },
                "detailedAnalysis": {
                    "logicalFlowValidation": {
                        "isSound": True,
                        "comments": "Good"
                    },
                    "dependencyValidation": {
                        "status": "Complete and Correct"
                    },
                    "completenessAndCorrectness": {
                        "coversAllClaims": True
                    }
                }
            },
            {
                "reviewer": "GPT-5 via Codex",
                "timestamp": "2025-11-10T17:30:05Z",
                "overallAssessment": {
                    "confidenceScore": "High (Ready for Expansion)",
                    "summary": "Sound",
                    "recommendation": "Proceed to Expansion"
                },
                "detailedAnalysis": {
                    "logicalFlowValidation": {
                        "isSound": True,
                        "comments": "Good"
                    },
                    "dependencyValidation": {
                        "status": "Complete and Correct"
                    },
                    "completenessAndCorrectness": {
                        "coversAllClaims": True
                    }
                }
            }
        ],
        "synthesisAndActionPlan": {
            "finalDecision": "Approved for Expansion",
            "consensusAnalysis": {
                "pointsOfAgreement": [
                    "Both reviewers confirm logical soundness",
                    "Both reviewers confirm completeness"
                ],
                "summaryOfFindings": "The proof sketch is rock-solid..."
            },
            "actionableItems": [],
            "confidenceStatement": "Ready for expansion"
        }
    }

    is_valid, errors = validate_validation_report(report)
    assert is_valid, f"Validation failed: {errors}"


def test_validation_report_missing_reviews():
    """Test that validation report requires exactly 2 reviews."""
    report = {
        "reportMetadata": {
            "sketchLabel": "thm-example",
            "validationCycleId": "550e8400-e29b-41d4-a716-446655440000",
            "validationTimestamp": "2025-11-10T17:35:00Z"
        },
        "originalProofSketch": {},
        "reviews": [  # Only 1 review - should fail minItems: 2
            {
                "reviewer": "Gemini 2.5 Pro",
                "timestamp": "2025-11-10T17:30:00Z",
                "overallAssessment": {
                    "confidenceScore": "High (Ready for Expansion)",
                    "summary": "Good",
                    "recommendation": "Proceed to Expansion"
                },
                "detailedAnalysis": {
                    "logicalFlowValidation": {
                        "isSound": True,
                        "comments": "Good"
                    },
                    "dependencyValidation": {
                        "status": "Complete and Correct"
                    },
                    "completenessAndCorrectness": {
                        "coversAllClaims": True
                    }
                }
            }
        ],
        "synthesisAndActionPlan": {
            "finalDecision": "Approved for Expansion",
            "consensusAnalysis": {
                "pointsOfAgreement": [],
                "summaryOfFindings": "..."
            },
            "actionableItems": [],
            "confidenceStatement": "..."
        }
    }

    is_valid, errors = validate_validation_report(report)
    assert not is_valid  # Should fail because reviews array has only 1 item


def test_validation_report_with_actionable_items():
    """Test validation report with actionable items (minor revisions)."""
    report = {
        "reportMetadata": {
            "sketchLabel": "thm-example",
            "validationCycleId": "550e8400-e29b-41d4-a716-446655440000",
            "validationTimestamp": "2025-11-10T17:35:00Z"
        },
        "originalProofSketch": {},
        "reviews": [
            {
                "reviewer": "Gemini 2.5 Pro",
                "timestamp": "2025-11-10T17:30:00Z",
                "overallAssessment": {
                    "confidenceScore": "Medium (Sound, but requires minor revisions)",
                    "summary": "Needs clarification",
                    "recommendation": "Revise and Resubmit for Validation"
                },
                "detailedAnalysis": {
                    "logicalFlowValidation": {
                        "isSound": True,
                        "comments": "Good"
                    },
                    "dependencyValidation": {
                        "status": "Minor Issues Found"
                    },
                    "completenessAndCorrectness": {
                        "coversAllClaims": True
                    }
                }
            },
            {
                "reviewer": "GPT-5 via Codex",
                "timestamp": "2025-11-10T17:30:05Z",
                "overallAssessment": {
                    "confidenceScore": "High (Ready for Expansion)",
                    "summary": "Good",
                    "recommendation": "Proceed to Expansion"
                },
                "detailedAnalysis": {
                    "logicalFlowValidation": {
                        "isSound": True,
                        "comments": "Good"
                    },
                    "dependencyValidation": {
                        "status": "Complete and Correct"
                    },
                    "completenessAndCorrectness": {
                        "coversAllClaims": True
                    }
                }
            }
        ],
        "synthesisAndActionPlan": {
            "finalDecision": "Requires Minor Revisions",
            "consensusAnalysis": {
                "pointsOfAgreement": ["Logical soundness confirmed"],
                "pointsOfDisagreement": [
                    {
                        "topic": "Dependency validation",
                        "geminiView": "Minor issues found",
                        "codexView": "Complete and correct",
                        "resolution": "Resolved via framework verification"
                    }
                ],
                "summaryOfFindings": "Overall sound but needs minor fixes"
            },
            "actionableItems": [
                {
                    "itemId": "AI-001",
                    "description": "Verify LSI constant is N-uniform",
                    "priority": "Critical",
                    "references": ["Gemini Review: dependencyValidation"]
                },
                {
                    "itemId": "AI-002",
                    "description": "Add explicit bound derivation for Step 3",
                    "priority": "High",
                    "references": ["Gemini Review: logicalFlowValidation.identifiedGaps[0]"]
                }
            ],
            "confidenceStatement": "Once 2 issues addressed, ready for expansion"
        }
    }

    is_valid, errors = validate_validation_report(report)
    assert is_valid, f"Validation failed: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
