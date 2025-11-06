"""
Error formatting and detailed error reporting for extraction workflows.

Provides utilities for creating structured error dictionaries and generating
LLM-friendly error reports that help ReAct agents self-correct on retry.
"""

import json
import traceback
from typing import Any

from pydantic import ValidationError


def make_error_dict(error_msg: str, value: Any = None) -> dict:
    """
    Create structured error dictionary for tracking extraction/improvement failures.

    This helper ensures consistent error format across the codebase:
    - 'error': The error message string (human-readable description)
    - 'value': The incorrectly generated value that caused the error (for debugging)

    Args:
        error_msg: Error message string describing what went wrong
        value: The malformed data, entity, or context that caused the error.
               Can be dict, list, exception details, or any JSON-serializable value.
               Use None if no meaningful value is available.

    Returns:
        Dictionary with 'error' and 'value' keys

    Examples:
        >>> make_error_dict("Failed to parse entity", {"label": "def-x", "term": "..."})
        {'error': 'Failed to parse entity', 'value': {'label': 'def-x', 'term': '...'}}

        >>> make_error_dict("LLM timeout", {"attempt": 1, "exception": "TimeoutError"})
        {'error': 'LLM timeout', 'value': {'attempt': 1, 'exception': 'TimeoutError'}}
    """
    return {"error": error_msg, "value": value}


def generate_detailed_error_report(
    error: Exception, attempt_number: int, max_retries: int, extraction_context: dict | None = None
) -> str:
    """
    Generate a detailed, LLM-friendly error report for failed extractions.

    This function transforms technical errors (Pydantic ValidationError, exceptions)
    into actionable feedback that helps the ReAct agent self-correct on retry.

    Args:
        error: The exception that occurred
        attempt_number: Current attempt number (1-indexed)
        max_retries: Maximum number of retry attempts
        extraction_context: Optional context about what was being extracted

    Returns:
        Formatted error report string optimized for LLM consumption

    Example Output:
        ```
        EXTRACTION ERROR REPORT
        =======================

        Attempt: 2/3
        Previous Error: Pydantic validation failed

        VALIDATION ERRORS:
        1. Field: definitions[0].term
           Problem: Missing required field
           Fix: Every definition must have a 'term' field with the exact text being defined
           Example: "term": "Lipschitz continuous"
        ```
    """
    # Build header
    report_lines = [
        "=" * 70,
        "EXTRACTION ERROR REPORT",
        "=" * 70,
        "",
        f"Attempt: {attempt_number}/{max_retries}",
        f"Error Type: {type(error).__name__}",
        "",
    ]

    # Add context if provided
    if extraction_context:
        report_lines.append("EXTRACTION CONTEXT:")
        for key, value in extraction_context.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 100:
                str_value = str_value[:97] + "..."
            report_lines.append(f"  - {key}: {str_value}")
        report_lines.append("")

    # Parse error based on type
    if isinstance(error, ValidationError):
        # Pydantic validation error - parse into detailed field-level feedback
        report_lines.extend(("VALIDATION ERRORS:", ""))

        for i, err in enumerate(error.errors(), 1):
            # Extract error details
            field_path = ".".join(str(loc) for loc in err["loc"])
            error_type = err["type"]
            error_msg = err["msg"]

            report_lines.extend((f"{i}. Field: {field_path}", f"   Problem: {error_msg}"))

            # Provide specific fix guidance based on error type
            if error_type == "missing":
                report_lines.extend((
                    "   Fix: This is a REQUIRED field. You must provide a value.",
                    f"   → Ensure the field '{field_path}' is present in your output",
                ))

            elif error_type in {"string_type", "int_type", "float_type", "bool_type"}:
                expected_type = error_type.replace("_type", "")
                report_lines.extend((
                    f"   Fix: Field must be of type '{expected_type}'",
                    f"   → Check that '{field_path}' has the correct data type",
                ))

            elif "literal_error" in error_type:
                # Extract allowed values from error message
                report_lines.append("   Fix: Field must match one of the allowed literal values")
                if "Input should be" in error_msg:
                    report_lines.append(f"   → {error_msg}")

            elif "list_type" in error_type:
                report_lines.extend((
                    "   Fix: Field must be a list/array",
                    f"   → Wrap value in square brackets: [{field_path}]",
                ))

            else:
                # Generic guidance
                report_lines.append(f"   Fix: {error_msg}")

            # Add field-specific examples
            if "term" in field_path:
                report_lines.append('   Example: "term": "Lipschitz continuous"')
            elif "label" in field_path:
                report_lines.append(
                    '   Example: "label": "def-lipschitz" (must match :label: directive)'
                )
            elif "statement_type" in field_path:
                report_lines.extend((
                    '   Example: "statement_type": "theorem"',
                    '   Allowed values: "theorem", "lemma", "proposition", "corollary"',
                ))
            elif "line_start" in field_path or "line_end" in field_path:
                report_lines.extend((
                    '   Example: "line_start": 42, "line_end": 58',
                    "   → Must be integers within document line range",
                ))

            report_lines.append("")

    elif isinstance(error, json.JSONDecodeError):
        # JSON parsing error
        report_lines.extend((
            "JSON PARSING ERROR:",
            "",
            f"Problem: Invalid JSON syntax at line {error.lineno}, column {error.colno}",
            f"Message: {error.msg}",
            "",
            "Common JSON Errors:",
            "  - Missing closing bracket/brace: }, ], )",
            "  - Trailing comma in last array/object element",
            "  - Unquoted string values",
            "  - Single quotes instead of double quotes",
            "",
            "Fix: Ensure valid JSON formatting:",
            '  - All strings use double quotes: "string"',
            "  - All brackets/braces are properly closed",
            "  - No trailing commas",
            "",
        ))

    elif "timeout" in str(error).lower():
        # Timeout error
        report_lines.extend((
            "TIMEOUT ERROR:",
            "",
            "Problem: Extraction took too long and timed out",
            "",
            "Possible Causes:",
            "  - Chapter is very large (too many entities)",
            "  - Agent got stuck in reasoning loop",
            "  - Network/API latency issues",
            "",
            "Fix: Try to:",
            "  - Focus on extracting only the most important entities",
            "  - Use more concise reasoning steps",
            "  - Validate early to catch errors quickly",
            "",
        ))

    else:
        # Generic exception
        report_lines.extend(("GENERAL ERROR:", "", f"Problem: {error!s}", ""))

        # Include truncated traceback
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        report_lines.append("Traceback (last 10 lines):")
        for line in tb_lines[-10:]:
            report_lines.append(f"  {line.rstrip()}")
        report_lines.append("")

    # Add general retry guidance
    report_lines.extend((
        "=" * 70,
        "RETRY GUIDANCE:",
        "=" * 70,
        "",
        "Read the errors above CAREFULLY and:",
        "1. Fix each field issue mentioned",
        "2. Verify data types match requirements",
        "3. Ensure all required fields are present",
        "4. Validate labels match :label: directives in source",
        "5. Call validate_extraction_tool to check before submitting",
        "",
        f"Remaining attempts: {max_retries - attempt_number}",
        "=" * 70,
    ))

    return "\n".join(report_lines)
