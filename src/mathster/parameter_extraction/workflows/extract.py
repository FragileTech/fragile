"""
Parameter extraction workflow.

Extracts structured Parameter objects from parameters_mentioned in existing extractions.
Follows the same patterns as the main extraction workflow but specifically targets
parameters that don't have their own Jupyter Book directives.

Workflow:
1. Load existing chapter extraction
2. Collect all parameters_mentioned from entities
3. Find parameter declarations in chapter text
4. Use DSPy ParameterExtractor to create Parameter objects
5. Validate and convert to RawParameter
6. Update chapter extraction with parameters
"""

import json
import logging
from typing import Any

from mathster.dspy_integration import make_error_dict
from mathster.parameter_extraction.conversion import convert_parameter
from mathster.parameter_extraction.text_processing import (
    collect_parameters_from_extraction,
    find_parameter_declarations,
)
from mathster.parameter_extraction.validation import validate_parameter
from mathster.parsing.dspy_components.extractors import ParameterExtractor
from mathster.parsing.models.entities import ParameterExtraction

logger = logging.getLogger(__name__)


def extract_parameters_from_chapter(
    chapter_text: str,
    existing_extraction: dict,
    chapter_number: int,
    file_path: str,
    article_id: str,
    max_retries: int = 3,
) -> tuple[list[dict], list[dict]]:
    """
    Extract parameters from a chapter using existing extraction data.

    This workflow:
    1. Collects all parameters mentioned in definitions/theorems
    2. Finds where those parameters are declared in the text
    3. Uses DSPy ParameterExtractor to create Parameter objects
    4. Validates each parameter
    5. Converts to RawParameter format
    6. Returns validated parameters and any errors

    Args:
        chapter_text: Chapter text with line numbers
        existing_extraction: ChapterExtraction dict (with definitions, theorems, etc.)
        chapter_number: Chapter index
        file_path: Path to source markdown file
        article_id: Article identifier
        max_retries: Maximum retry attempts with fallback

    Returns:
        Tuple of (list of RawParameter dicts, list of error dicts)
    """
    logger.info(f"Extracting parameters from chapter {chapter_number}")

    errors = []
    raw_parameters = []

    # Stage 1: Collect parameters from existing extraction
    parameters_mentioned = collect_parameters_from_extraction(existing_extraction)

    if not parameters_mentioned:
        logger.info("No parameters mentioned in chapter")
        return [], []

    logger.info(f"Found {len(parameters_mentioned)} parameters mentioned: {parameters_mentioned}")

    # Stage 2: Find parameter declarations in text
    parameter_declarations = find_parameter_declarations(
        chapter_text, list(parameters_mentioned)
    )

    logger.info(f"Found declarations for {len(parameter_declarations)} parameters")

    if not parameter_declarations:
        warning = "No parameter declarations found in chapter text"
        logger.warning(warning)
        errors.append(make_error_dict(warning))
        # Continue anyway - agent might find patterns we missed
        parameter_declarations = {}

    # Stage 3: Extract parameters using DSPy agent
    extractor = ParameterExtractor()
    previous_error_report = ""

    for attempt in range(max_retries):
        try:
            logger.debug(f"Parameter extraction attempt {attempt + 1}/{max_retries}")

            # Run agent
            parameter_extractions = extractor(
                chapter_with_lines=chapter_text,
                parameters_mentioned=list(parameters_mentioned),
                parameter_declarations=parameter_declarations,
                file_path=file_path,
                article_id=article_id,
                previous_error_report=previous_error_report,
            )

            # Validate and convert each parameter
            attempt_errors = []
            attempt_parameters = []

            for param_data in parameter_extractions:
                # Convert to dict if it's a Pydantic model
                if hasattr(param_data, "model_dump"):
                    param_dict = param_data.model_dump()
                elif isinstance(param_data, dict):
                    param_dict = param_data
                else:
                    error = f"Unexpected parameter format: {type(param_data)}"
                    attempt_errors.append(make_error_dict(error, value=param_data))
                    continue

                # Validate parameter
                validation_result = validate_parameter(
                    param_dict,
                    file_path=file_path,
                    article_id=article_id,
                    chapter_text=chapter_text,
                )

                if not validation_result.is_valid:
                    error = f"Invalid parameter {param_dict.get('label', 'unknown')}: {validation_result.errors}"
                    attempt_errors.append(make_error_dict(error, value=param_dict))
                    logger.debug(f"  Validation failed: {validation_result.errors}")
                    continue

                # Convert to RawParameter
                try:
                    param_extraction = ParameterExtraction(**param_dict)
                    raw_param, conversion_warnings = convert_parameter(
                        param_extraction,
                        file_path=file_path,
                        article_id=article_id,
                        chapter_text=chapter_text,
                    )

                    # Add to results
                    attempt_parameters.append(raw_param.model_dump())

                    if conversion_warnings:
                        for warning in conversion_warnings:
                            logger.warning(f"  {warning}")

                except Exception as e:
                    error = f"Failed to convert parameter: {e}"
                    attempt_errors.append(make_error_dict(error, value=param_dict))
                    logger.debug(f"  Conversion failed: {e}")

            # Check if we got good results
            if attempt_parameters and len(attempt_errors) == 0:
                # Success!
                logger.info(f"✓ Extracted {len(attempt_parameters)} parameters")
                return attempt_parameters, []

            elif attempt_parameters and len(attempt_errors) < len(parameter_extractions):
                # Partial success
                logger.info(
                    f"✓ Extracted {len(attempt_parameters)} parameters with {len(attempt_errors)} errors"
                )
                return attempt_parameters, attempt_errors

            else:
                # All failed or no results
                if attempt < max_retries - 1:
                    # Prepare error report for next attempt
                    previous_error_report = "\n".join([e["error"] for e in attempt_errors])
                    logger.debug(f"Retrying with error report:\n{previous_error_report}")
                else:
                    # Final attempt failed
                    logger.error(f"Failed to extract parameters after {max_retries} attempts")
                    return [], attempt_errors

        except Exception as e:
            error = f"Parameter extraction failed: {e}"
            logger.error(error)
            errors.append(make_error_dict(error))

            if attempt == max_retries - 1:
                return [], errors

    # Should not reach here
    return raw_parameters, errors


def improve_parameters_from_chapter(
    chapter_text: str,
    existing_parameters: list[dict],
    existing_extraction: dict,
    file_path: str,
    article_id: str,
) -> tuple[list[dict], list[dict]]:
    """
    Improve parameter extraction by finding missed parameters.

    Compares parameters_mentioned with existing extracted parameters
    to find any that were missed.

    Args:
        chapter_text: Chapter text with line numbers
        existing_parameters: List of existing RawParameter dicts
        existing_extraction: ChapterExtraction dict
        file_path: Path to source markdown file
        article_id: Article identifier

    Returns:
        Tuple of (improved parameter list, errors)
    """
    logger.info("Improving parameter extraction")

    # Collect all mentioned parameters
    parameters_mentioned = collect_parameters_from_extraction(existing_extraction)

    # Collect existing parameter labels
    existing_labels = {p.get("label", "") for p in existing_parameters}
    existing_symbols = {p.get("symbol", "") for p in existing_parameters}

    # Find missed parameters
    missed_symbols = parameters_mentioned - existing_symbols

    if not missed_symbols:
        logger.info("No missed parameters found")
        return existing_parameters, []

    logger.info(f"Found {len(missed_symbols)} missed parameters: {missed_symbols}")

    # Extract missed parameters
    new_parameters, errors = extract_parameters_from_chapter(
        chapter_text=chapter_text,
        existing_extraction=existing_extraction,
        chapter_number=0,  # Not used for improvement
        file_path=file_path,
        article_id=article_id,
    )

    # Filter to only missed parameters
    filtered_new = [p for p in new_parameters if p.get("symbol") in missed_symbols]

    # Merge with existing
    all_parameters = existing_parameters + filtered_new

    logger.info(f"Total parameters after improvement: {len(all_parameters)}")

    return all_parameters, errors
