"""
Validation and error handling for mathematical entity extraction.

This module provides validation logic, error reporting, and comparison tools
for ensuring the quality and correctness of extracted mathematical entities.

Submodules:
    - validators: Core validation logic for extraction results
    - errors: Error formatting and detailed error reporting
    - comparators: Comparison tools for validating extraction quality
"""

from mathster.parsing.validation.errors import (
    generate_detailed_error_report,
    make_error_dict,
)
from mathster.parsing.validation.validators import validate_extraction

__all__ = [
    # Validators
    "validate_extraction",
    # Error handling
    "make_error_dict",
    "generate_detailed_error_report",
]
