"""Tests for sketch strategy validation utilities."""

import json
import pytest
from pathlib import Path

# Import validation functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mathster.agent_schemas.validate_sketch import (
    validate_sketch_strategy,
    get_missing_required_fields,
    fill_missing_fields,
    load_schema,
    validate_full_sketch,
    fill_missing_sketch_fields,
)


def test_load_schema():
    """Test that schema file can be loaded."""
    schema = load_schema()
    assert isinstance(schema, dict)
    assert "$schema" in schema
    assert "properties" in schema
    assert "required" in schema


def test_valid_strategy():
    """Test validation of a complete valid strategy."""
    strategy = {
        "strategist": "Gemini 2.5 Pro",
        "method": "Lyapunov Method via KL-Divergence",
        "summary": "Construct KL-divergence as Lyapunov function and show exponential decay",
        "keySteps": [
            "Step 1: Define KL-divergence as Lyapunov function",
            "Step 2: Compute dissipation via LSI",
            "Step 3: Apply Grönwall inequality"
        ],
        "strengths": [
            "Direct approach using framework LSI",
            "Explicit exponential rate"
        ],
        "weaknesses": [
            "Requires LSI constant to be N-uniform"
        ],
        "frameworkDependencies": {
            "theorems": [
                {
                    "label": "thm-lsi-main",
                    "document": "09_kl_convergence",
                    "purpose": "Provides LSI constant for entropy dissipation"
                }
            ],
            "lemmas": [],
            "axioms": [
                {
                    "label": "axiom-bounded-displacement",
                    "document": "01_fragile_gas_framework",
                    "purpose": "Ensures finite displacement bounds"
                }
            ],
            "definitions": []
        },
        "technicalDeepDives": [
            {
                "challengeTitle": "LSI Constant Uniformity",
                "difficultyDescription": "Need to verify LSI constant doesn't depend on N",
                "proposedSolution": "Use framework's LSI theorem with explicit bounds"
            }
        ],
        "confidenceScore": "High"
    }

    is_valid, errors = validate_sketch_strategy(strategy)
    assert is_valid, f"Validation failed: {errors}"
    assert errors == []


def test_missing_required_fields():
    """Test detection of missing required fields."""
    strategy = {
        "strategist": "Gemini 2.5 Pro",
        "method": "Lyapunov Method"
        # Missing: summary, keySteps, strengths, weaknesses, frameworkDependencies, confidenceScore
    }

    missing = get_missing_required_fields(strategy)
    assert "summary" in missing
    assert "keySteps" in missing
    assert "strengths" in missing
    assert "weaknesses" in missing
    assert "frameworkDependencies" in missing
    assert "confidenceScore" in missing


def test_fill_missing_fields():
    """Test automatic filling of missing required fields."""
    strategy = {
        "strategist": "Gemini 2.5 Pro",
        "method": "Lyapunov Method"
    }

    filled, filled_fields = fill_missing_fields(strategy)

    # Check that missing fields were filled
    assert "summary" in filled_fields
    assert "keySteps" in filled_fields
    assert "frameworkDependencies" in filled_fields

    # Check that filled values are present
    assert filled["summary"] == "[Missing from AI output]"
    assert filled["keySteps"] == ["[Missing from AI output]"]
    assert filled["confidenceScore"] == "Low"
    assert filled["frameworkDependencies"] == {
        "theorems": [],
        "lemmas": [],
        "axioms": [],
        "definitions": []
    }

    # Check that original fields are preserved
    assert filled["strategist"] == "Gemini 2.5 Pro"
    assert filled["method"] == "Lyapunov Method"


def test_invalid_strategy_validation():
    """Test that invalid strategies fail validation."""
    # Strategy with invalid confidenceScore
    strategy = {
        "strategist": "Gemini 2.5 Pro",
        "method": "Lyapunov Method",
        "summary": "Test summary",
        "keySteps": ["Step 1"],
        "strengths": ["Strength 1"],
        "weaknesses": ["Weakness 1"],
        "frameworkDependencies": {
            "theorems": [],
            "lemmas": [],
            "axioms": [],
            "definitions": []
        },
        "confidenceScore": "VeryHigh"  # Invalid - not in enum
    }

    is_valid, errors = validate_sketch_strategy(strategy)
    assert not is_valid
    assert len(errors) > 0


def test_empty_key_steps():
    """Test that empty keySteps array fails validation."""
    strategy = {
        "strategist": "Gemini 2.5 Pro",
        "method": "Lyapunov Method",
        "summary": "Test summary",
        "keySteps": [],  # Empty - should fail minItems: 1
        "strengths": ["Strength 1"],
        "weaknesses": ["Weakness 1"],
        "frameworkDependencies": {
            "theorems": [],
            "lemmas": [],
            "axioms": [],
            "definitions": []
        },
        "confidenceScore": "High"
    }

    is_valid, errors = validate_sketch_strategy(strategy)
    assert not is_valid
    assert len(errors) > 0


# ============================================================================
# Full Sketch Validation Tests (sketch.json format)
# ============================================================================

def test_validate_full_sketch_valid():
    """Test validation of complete valid full sketch (sketch.json format)"""
    full_sketch = {
        "title": "Example Convergence Theorem",
        "label": "thm-example-convergence",
        "type": "Theorem",
        "source": "docs/source/1_euclidean_gas/09_kl_convergence.md#L245",
        "date": "2025-11-10",
        "status": "Ready for Expansion",
        "statement": {
            "formal": "Under the stated assumptions, the system converges exponentially fast.",
            "informal": "The algorithm reaches equilibrium quickly with exponential rate."
        },
        "strategySynthesis": {
            "strategies": [
                {
                    "strategist": "Gemini 2.5 Pro",
                    "method": "Lyapunov via KL-Divergence",
                    "keySteps": ["Step 1", "Step 2"],
                    "strengths": ["Direct approach"],
                    "weaknesses": ["Requires LSI"]
                },
                {
                    "strategist": "GPT-5 via Codex",
                    "method": "Coupling Method",
                    "keySteps": ["Step 1", "Step 2"],
                    "strengths": ["Explicit rate"],
                    "weaknesses": ["Complex construction"]
                }
            ],
            "recommendedApproach": {
                "chosenMethod": "Lyapunov via KL-Divergence",
                "rationale": "More direct and framework-aligned",
                "verificationStatus": {
                    "frameworkDependencies": "Verified",
                    "circularReasoning": "No circularity detected",
                    "keyAssumptions": "All assumptions are standard",
                    "crossValidation": "Consensus between strategists"
                }
            }
        },
        "dependencies": {
            "verifiedDependencies": [
                {
                    "label": "thm-lsi-main",
                    "type": "Theorem",
                    "sourceDocument": "09_kl_convergence.md",
                    "purpose": "Provides LSI constant",
                    "usedInSteps": ["Step 1"]
                }
            ],
            "missingOrUncertainDependencies": {
                "lemmasToProve": [],
                "uncertainAssumptions": []
            }
        },
        "detailedProof": {
            "overview": "Use KL divergence as Lyapunov function...",
            "topLevelOutline": [
                "1. Construct KL divergence",
                "2. Derive dissipation rate",
                "3. Apply Grönwall"
            ],
            "steps": [
                {
                    "stepNumber": 1,
                    "title": "Step 1",
                    "goal": "Construct KL divergence",
                    "action": "Define V(t) = KL(μ_t || π)",
                    "justification": "def-kl-divergence",
                    "expectedResult": "V(t) ≥ 0"
                }
            ],
            "conclusion": "Q.E.D."
        },
        "validationChecklist": {
            "logicalCompleteness": True,
            "hypothesisUsage": True,
            "conclusionDerivation": True,
            "frameworkConsistency": True,
            "noCircularReasoning": True,
            "constantTracking": False,
            "edgeCases": False,
            "regularityAssumptions": True
        }
    }

    is_valid, errors = validate_full_sketch(full_sketch)
    assert is_valid, f"Validation failed: {errors}"
    assert errors == []


def test_validate_full_sketch_missing_required_fields():
    """Test that missing required fields are detected in full sketch"""
    incomplete_sketch = {
        "label": "thm-example",
        "type": "Theorem",
        # Missing: title, source, date, status, statement, strategySynthesis, dependencies, etc.
    }

    is_valid, errors = validate_full_sketch(incomplete_sketch)
    assert not is_valid
    assert len(errors) > 0


def test_fill_missing_sketch_fields():
    """Test auto-filling missing fields in full sketch"""
    partial_sketch = {
        "label": "thm-test",
        "type": "Theorem",
        "statement": {
            "formal": "Test statement",
            "informal": "Test informal"
        }
    }

    filled_sketch, filled_fields = fill_missing_sketch_fields(partial_sketch)

    # Check that missing fields were filled
    assert "title" in filled_sketch
    assert "source" in filled_sketch
    assert "date" in filled_sketch
    assert "status" in filled_sketch
    assert "strategySynthesis" in filled_sketch
    assert "dependencies" in filled_sketch
    assert "detailedProof" in filled_sketch
    assert "validationChecklist" in filled_sketch

    # Check that filled fields were tracked
    assert "title" in filled_fields
    assert "source" in filled_fields
    assert "strategySynthesis" in filled_fields

    # Verify filled sketch now validates
    is_valid, errors = validate_full_sketch(filled_sketch)
    assert is_valid, f"Filled sketch should validate: {errors}"


def test_fill_missing_sketch_fields_preserves_existing():
    """Test that auto-fill preserves existing fields"""
    partial_sketch = {
        "label": "thm-test",
        "title": "Custom Title",
        "type": "Lemma",
        "statement": {
            "formal": "Custom statement",
            "informal": "Custom informal"
        },
        "strategySynthesis": {
            "strategies": [
                {
                    "strategist": "Gemini 2.5 Pro",
                    "method": "Custom Method",
                    "keySteps": ["Step 1", "Step 2"],
                    "strengths": ["Good"],
                    "weaknesses": ["None"]
                }
            ],
            "recommendedApproach": {
                "chosenMethod": "Custom Method",
                "rationale": "Custom rationale",
                "verificationStatus": {
                    "frameworkDependencies": "Verified",
                    "circularReasoning": "None",
                    "keyAssumptions": "Standard",
                    "crossValidation": "Single"
                }
            }
        }
    }

    filled_sketch, filled_fields = fill_missing_sketch_fields(partial_sketch)

    # Verify existing fields were preserved
    assert filled_sketch["label"] == "thm-test"
    assert filled_sketch["title"] == "Custom Title"
    assert filled_sketch["type"] == "Lemma"
    assert filled_sketch["statement"]["formal"] == "Custom statement"
    assert filled_sketch["strategySynthesis"]["strategies"][0]["strategist"] == "Gemini 2.5 Pro"
    assert filled_sketch["strategySynthesis"]["recommendedApproach"]["chosenMethod"] == "Custom Method"

    # Verify only missing fields were tracked
    assert "label" not in filled_fields
    assert "title" not in filled_fields
    assert "statement" not in filled_fields
    assert "strategySynthesis" not in filled_fields

    # But missing fields should be in tracked list
    assert "source" in filled_fields
    assert "date" in filled_fields
    assert "dependencies" in filled_fields


def test_full_sketch_with_both_strategies():
    """Test full sketch containing both Gemini and Codex strategies"""
    sketch_with_dual = {
        "title": "Dual Strategy Example",
        "label": "thm-dual-example",
        "type": "Theorem",
        "source": "test.md#L1",
        "date": "2025-11-10",
        "status": "Draft",
        "statement": {
            "formal": "Test",
            "informal": "Test"
        },
        "strategySynthesis": {
            "strategies": [
                {
                    "strategist": "Gemini 2.5 Pro",
                    "method": "Method A",
                    "keySteps": ["A1", "A2"],
                    "strengths": ["Direct"],
                    "weaknesses": ["Complex"]
                },
                {
                    "strategist": "GPT-5 via Codex",
                    "method": "Method B",
                    "keySteps": ["B1", "B2"],
                    "strengths": ["Explicit"],
                    "weaknesses": ["Long"]
                }
            ],
            "recommendedApproach": {
                "chosenMethod": "Method A",
                "rationale": "More direct",
                "verificationStatus": {
                    "frameworkDependencies": "Verified",
                    "circularReasoning": "No circularity detected",
                    "keyAssumptions": "All assumptions are standard",
                    "crossValidation": "Consensus between strategists"
                }
            }
        },
        "dependencies": {
            "verifiedDependencies": [],
            "missingOrUncertainDependencies": {
                "lemmasToProve": [],
                "uncertainAssumptions": []
            }
        },
        "detailedProof": {
            "overview": "Overview",
            "topLevelOutline": ["1. A1", "2. A2"],
            "steps": [],
            "conclusion": "Q.E.D."
        },
        "validationChecklist": {
            "logicalCompleteness": True,
            "hypothesisUsage": False,
            "conclusionDerivation": True,
            "frameworkConsistency": True,
            "noCircularReasoning": True,
            "constantTracking": False,
            "edgeCases": False,
            "regularityAssumptions": False
        }
    }

    is_valid, errors = validate_full_sketch(sketch_with_dual)
    assert is_valid, f"Dual strategy sketch should validate: {errors}"

    # Verify both strategies are present
    assert len(sketch_with_dual["strategySynthesis"]["strategies"]) == 2
    assert sketch_with_dual["strategySynthesis"]["strategies"][0]["strategist"] == "Gemini 2.5 Pro"
    assert sketch_with_dual["strategySynthesis"]["strategies"][1]["strategist"] == "GPT-5 via Codex"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
