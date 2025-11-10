"""
Validation utilities for proof sketch strategies.

This module provides functions to validate proof sketch JSON files against
the sketch_strategy.json schema.
"""

import json
from pathlib import Path
from typing import Any

try:
    import jsonschema
    from jsonschema import ValidationError, Draft7Validator
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    ValidationError = Exception  # Fallback for type hints


def get_schema_path() -> Path:
    """Get the path to sketch_strategy.json schema."""
    return Path(__file__).parent / "sketch_strategy.json"


def load_schema() -> dict[str, Any]:
    """Load the sketch_strategy.json schema."""
    schema_path = get_schema_path()
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_sketch_strategy(
    strategy_json: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate a proof sketch strategy against the schema.

    Parameters
    ----------
    strategy_json : dict
        The strategy JSON to validate
    schema : dict, optional
        The JSON schema to validate against. If None, loads from sketch_strategy.json

    Returns
    -------
    is_valid : bool
        True if validation passes, False otherwise
    errors : list[str]
        List of validation error messages (empty if valid)

    Examples
    --------
    >>> strategy = {
    ...     "strategist": "Gemini 2.5 Pro",
    ...     "method": "Lyapunov Method",
    ...     "summary": "Use KL-divergence as Lyapunov function",
    ...     "keySteps": ["Step 1", "Step 2"],
    ...     "strengths": ["Direct approach"],
    ...     "weaknesses": ["Requires LSI"],
    ...     "frameworkDependencies": {"theorems": [], "lemmas": [], "axioms": [], "definitions": []},
    ...     "confidenceScore": "High"
    ... }
    >>> is_valid, errors = validate_sketch_strategy(strategy)
    >>> is_valid
    True
    >>> errors
    []
    """
    if not HAS_JSONSCHEMA:
        return False, ["jsonschema library not installed. Run: pip install jsonschema"]

    if schema is None:
        try:
            schema = load_schema()
        except FileNotFoundError as e:
            return False, [str(e)]

    try:
        jsonschema.validate(strategy_json, schema)
        return True, []
    except ValidationError as e:
        # Format error message
        error_msg = f"Validation error at {'.'.join(str(p) for p in e.path)}: {e.message}"
        return False, [error_msg]
    except Exception as e:
        return False, [f"Unexpected validation error: {e}"]


def get_missing_required_fields(
    strategy_json: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> list[str]:
    """
    Get list of missing required fields in a strategy.

    Parameters
    ----------
    strategy_json : dict
        The strategy JSON to check
    schema : dict, optional
        The JSON schema. If None, loads from sketch_strategy.json

    Returns
    -------
    missing_fields : list[str]
        List of required field names that are missing

    Examples
    --------
    >>> strategy = {"strategist": "Gemini", "method": "Lyapunov"}
    >>> missing = get_missing_required_fields(strategy)
    >>> "keySteps" in missing
    True
    """
    if schema is None:
        schema = load_schema()

    required_fields = schema.get("required", [])
    missing = [field for field in required_fields if field not in strategy_json]
    return missing


def fill_missing_fields(
    strategy_json: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """
    Fill missing required fields with default values.

    Parameters
    ----------
    strategy_json : dict
        The strategy JSON to fill
    schema : dict, optional
        The JSON schema. If None, loads from sketch_strategy.json

    Returns
    -------
    filled_strategy : dict
        Strategy with missing fields filled
    filled_fields : list[str]
        List of field names that were filled

    Examples
    --------
    >>> strategy = {"strategist": "Gemini"}
    >>> filled, filled_fields = fill_missing_fields(strategy)
    >>> "method" in filled_fields
    True
    >>> filled["method"]
    '[Missing from AI output]'
    """
    if schema is None:
        schema = load_schema()

    # Create a copy to avoid modifying original
    filled = strategy_json.copy()
    filled_fields = []

    missing = get_missing_required_fields(filled, schema)

    # Default values for each field type
    defaults = {
        "strategist": "Unknown (missing from output)",
        "method": "[Missing from AI output]",
        "summary": "[Missing from AI output]",
        "keySteps": ["[Missing from AI output]"],
        "strengths": ["[Missing from AI output]"],
        "weaknesses": ["[Missing from AI output]"],
        "frameworkDependencies": {
            "theorems": [],
            "lemmas": [],
            "axioms": [],
            "definitions": [],
        },
        "technicalDeepDives": [],
        "confidenceScore": "Low",
    }

    for field in missing:
        if field in defaults:
            filled[field] = defaults[field]
            filled_fields.append(field)

    return filled, filled_fields


def validate_file(file_path: str | Path) -> tuple[bool, list[str]]:
    """
    Validate a sketch strategy JSON file.

    Parameters
    ----------
    file_path : str or Path
        Path to JSON file containing a strategy

    Returns
    -------
    is_valid : bool
        True if validation passes, False otherwise
    errors : list[str]
        List of validation error messages (empty if valid)

    Examples
    --------
    >>> is_valid, errors = validate_file("sketch-thm-example.json")
    >>> if not is_valid:
    ...     print(f"Validation failed: {errors}")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False, [f"File not found: {file_path}"]

    try:
        strategy = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    # Extract strategy from wrapper if present
    if "strategy" in strategy and isinstance(strategy["strategy"], dict):
        strategy_to_validate = strategy["strategy"]
    else:
        strategy_to_validate = strategy

    return validate_sketch_strategy(strategy_to_validate)


# =============================================================================
# Validation Request and Validation Report Functions
# =============================================================================


def load_validation_request_schema() -> dict[str, Any]:
    """Load the sketch_validation_request.json schema."""
    schema_path = Path(__file__).parent / "sketch_validation_request.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def load_validation_report_schema() -> dict[str, Any]:
    """Load the sketch_validation.json schema."""
    schema_path = Path(__file__).parent / "sketch_validation.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_validation_request(
    review_json: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate a single validation review against sketch_validation_request.json schema.

    Parameters
    ----------
    review_json : dict
        The validation review JSON to validate
    schema : dict, optional
        The JSON schema to validate against. If None, loads from sketch_validation_request.json

    Returns
    -------
    is_valid : bool
        True if validation passes, False otherwise
    errors : list[str]
        List of validation error messages (empty if valid)

    Examples
    --------
    >>> review = {
    ...     "reviewer": "Gemini 2.5 Pro",
    ...     "timestamp": "2025-11-10T17:30:00Z",
    ...     "overallAssessment": {
    ...         "confidenceScore": "High (Ready for Expansion)",
    ...         "summary": "Sketch is sound",
    ...         "recommendation": "Proceed to Expansion"
    ...     },
    ...     "detailedAnalysis": {
    ...         "logicalFlowValidation": {"isSound": True, "comments": "Clear logic"},
    ...         "dependencyValidation": {"status": "Complete and Correct"},
    ...         "completenessAndCorrectness": {"coversAllClaims": True}
    ...     }
    ... }
    >>> is_valid, errors = validate_validation_request(review)
    >>> is_valid
    True
    """
    if not HAS_JSONSCHEMA:
        return False, ["jsonschema library not installed. Run: pip install jsonschema"]

    if schema is None:
        try:
            schema = load_validation_request_schema()
        except FileNotFoundError as e:
            return False, [str(e)]

    try:
        jsonschema.validate(review_json, schema)
        return True, []
    except ValidationError as e:
        error_msg = f"Validation error at {'.'.join(str(p) for p in e.path)}: {e.message}"
        return False, [error_msg]
    except Exception as e:
        return False, [f"Unexpected validation error: {e}"]


def validate_validation_report(
    report_json: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate a complete validation report against sketch_validation.json schema.

    Parameters
    ----------
    report_json : dict
        The validation report JSON to validate
    schema : dict, optional
        The JSON schema to validate against. If None, loads from sketch_validation.json

    Returns
    -------
    is_valid : bool
        True if validation passes, False otherwise
    errors : list[str]
        List of validation error messages (empty if valid)

    Examples
    --------
    >>> report = {
    ...     "reportMetadata": {
    ...         "sketchLabel": "thm-example",
    ...         "validationCycleId": "uuid-string",
    ...         "validationTimestamp": "2025-11-10T17:30:00Z"
    ...     },
    ...     "originalProofSketch": {...},
    ...     "reviews": [{...}, {...}],
    ...     "synthesisAndActionPlan": {...}
    ... }
    >>> is_valid, errors = validate_validation_report(report)
    """
    if not HAS_JSONSCHEMA:
        return False, ["jsonschema library not installed. Run: pip install jsonschema"]

    if schema is None:
        try:
            schema = load_validation_report_schema()
        except FileNotFoundError as e:
            return False, [str(e)]

    try:
        jsonschema.validate(report_json, schema)
        return True, []
    except ValidationError as e:
        error_msg = f"Validation error at {'.'.join(str(p) for p in e.path)}: {e.message}"
        return False, [error_msg]
    except Exception as e:
        return False, [f"Unexpected validation error: {e}"]


def load_sketch_for_validation(
    sketch_path: Path | str,
) -> tuple[dict[str, Any], list[str]]:
    """
    Load and preprocess a sketch file for validation.

    This function:
    1. Loads the JSON file
    2. Validates against sketch_strategy.json schema
    3. Fills missing required fields with defaults
    4. Returns the preprocessed sketch and notes about what was filled

    Parameters
    ----------
    sketch_path : Path or str
        Path to the sketch JSON file

    Returns
    -------
    preprocessed_sketch : dict
        The sketch with missing fields filled
    preprocessing_notes : list[str]
        List of notes about preprocessing actions taken

    Examples
    --------
    >>> sketch, notes = load_sketch_for_validation("sketch-thm-example.json")
    >>> if notes:
    ...     print(f"Preprocessing actions: {notes}")
    """
    sketch_path = Path(sketch_path)
    notes = []

    # Load file
    if not sketch_path.exists():
        raise FileNotFoundError(f"Sketch file not found: {sketch_path}")

    try:
        sketch_data = json.loads(sketch_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in sketch file: {e}")

    # Extract strategy
    if "strategy" not in sketch_data:
        raise ValueError("Sketch file missing 'strategy' field")

    strategy = sketch_data["strategy"]

    # Validate schema
    is_valid, errors = validate_sketch_strategy(strategy)
    if not is_valid:
        notes.append(f"Schema validation failed: {'; '.join(errors)}")

    # Fill missing fields
    missing = get_missing_required_fields(strategy)
    if missing:
        filled, filled_fields = fill_missing_fields(strategy)
        sketch_data["strategy"] = filled
        notes.append(f"Auto-filled missing fields: {', '.join(filled_fields)}")

        # Add to metadata
        if "_metadata" not in sketch_data:
            sketch_data["_metadata"] = {}

        sketch_data["_metadata"]["validation_preprocessing"] = {
            "filled_fields": filled_fields,
            "original_missing": missing,
            "note": "Auto-filled by load_sketch_for_validation()"
        }

    return sketch_data, notes


# =============================================================================
# Full Proof Sketch Validation Functions (sketch.json schema)
# =============================================================================


def load_sketch_schema() -> dict[str, Any]:
    """Load the sketch.json schema."""
    schema_path = Path(__file__).parent / "sketch.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_full_sketch(
    sketch_json: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate a complete proof sketch against sketch.json schema.

    Parameters
    ----------
    sketch_json : dict
        The full proof sketch to validate
    schema : dict, optional
        The JSON schema. If None, loads from sketch.json

    Returns
    -------
    is_valid : bool
        True if validation passes
    errors : list[str]
        Validation error messages

    Examples
    --------
    >>> sketch = {
    ...     "title": "Main Theorem",
    ...     "label": "thm-main",
    ...     "type": "Theorem",
    ...     "source": "doc.md",
    ...     "date": "2025-11-10",
    ...     "status": "Sketch",
    ...     "statement": {"formal": "...", "informal": "..."},
    ...     "strategySynthesis": {...},
    ...     "dependencies": {...},
    ...     "detailedProof": {...},
    ...     "validationChecklist": {...}
    ... }
    >>> is_valid, errors = validate_full_sketch(sketch)
    """
    if not HAS_JSONSCHEMA:
        return False, ["jsonschema library not installed"]

    if schema is None:
        try:
            schema = load_sketch_schema()
        except FileNotFoundError as e:
            return False, [str(e)]

    try:
        jsonschema.validate(sketch_json, schema)
        return True, []
    except ValidationError as e:
        error_msg = f"Validation error at {'.'.join(str(p) for p in e.path)}: {e.message}"
        return False, [error_msg]
    except Exception as e:
        return False, [f"Unexpected validation error: {e}"]


def fill_missing_sketch_fields(
    sketch_json: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """
    Fill missing required fields in full sketch with defaults.

    Parameters
    ----------
    sketch_json : dict
        The sketch to fill

    Returns
    -------
    filled_sketch : dict
        Sketch with missing fields filled
    filled_fields : list[str]
        Names of fields that were filled

    Examples
    --------
    >>> sketch = {"label": "thm-example"}
    >>> filled, filled_fields = fill_missing_sketch_fields(sketch)
    >>> "title" in filled_fields
    True
    """
    import datetime

    filled = sketch_json.copy()
    filled_fields = []

    # Default values for required fields
    defaults = {
        "title": "[Title not specified]",
        "label": "unknown-label",
        "type": "Theorem",
        "source": "unknown",
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "status": "Sketch",
        "statement": {
            "formal": "[Statement not provided]",
            "informal": "[Informal statement not provided]"
        },
        "strategySynthesis": {
            "strategies": [],
            "recommendedApproach": {
                "chosenMethod": "Unknown",
                "rationale": "Not specified",
                "verificationStatus": {
                    "frameworkDependencies": "Needs Verification",
                    "circularReasoning": "No circularity detected",
                    "keyAssumptions": "All assumptions are standard",
                    "crossValidation": "Single strategist only"
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
            "overview": "Not provided",
            "topLevelOutline": [],
            "steps": [],
            "conclusion": "Q.E.D."
        },
        "validationChecklist": {
            "logicalCompleteness": False,
            "hypothesisUsage": False,
            "conclusionDerivation": False,
            "frameworkConsistency": False,
            "noCircularReasoning": True,
            "constantTracking": False,
            "edgeCases": False,
            "regularityAssumptions": False
        }
    }

    # Fill top-level missing fields
    for field, default_value in defaults.items():
        if field not in filled:
            filled[field] = default_value
            filled_fields.append(field)

    return filled, filled_fields


def main():
    """CLI for validating sketch strategy files."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python validate_sketch.py <sketch-file.json>")
        sys.exit(1)

    file_path = sys.argv[1]
    is_valid, errors = validate_file(file_path)

    if is_valid:
        print(f"✅ Validation PASSED: {file_path}")
        sys.exit(0)
    else:
        print(f"❌ Validation FAILED: {file_path}")
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
