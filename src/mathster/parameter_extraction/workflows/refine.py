"""
Parameter line number refinement workflow using DSPy agents.

This workflow uses DSPy agents to find line numbers for parameters that the
automated regex extraction couldn't locate (those still at line 1).

Used as Stage 2 in the hybrid parameter extraction pipeline:
- Stage 1: Automated regex (fast, 86% success)
- Stage 2: DSPy refinement (slow, handles remaining 14%)

Usage:
    from mathster.parsing.workflows.refine_parameters import refine_parameter_line_numbers

    updated, failed, errors = refine_parameter_line_numbers(
        chapter_file=Path("chapter_3.json"),
        full_document_text=document_with_line_numbers,
        file_path="docs/source/.../doc.md",
        article_id="doc_id",
    )
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_usage_context(symbol: str, chapter_data: dict) -> str:
    """
    Extract context showing how a parameter is used.

    Searches definitions and theorems for where the parameter is mentioned.

    Args:
        symbol: Parameter symbol
        chapter_data: Chapter extraction data

    Returns:
        Context string showing parameter usage
    """
    contexts = []

    # Search in definitions
    for defn in chapter_data.get("definitions", []):
        params_mentioned = defn.get("parameters_mentioned", [])
        if symbol in params_mentioned:
            term = defn.get("term", "")
            contexts.append(f"Used in definition '{term}'")
            # Could add snippet of definition text if available

    # Search in theorems
    for thm in chapter_data.get("theorems", []):
        params_mentioned = thm.get("parameters_mentioned", [])
        if symbol in params_mentioned:
            label = thm.get("label", "")
            contexts.append(f"Used in theorem '{label}'")

    if contexts:
        return " | ".join(contexts[:3])  # Limit to 3 contexts
    else:
        return f"Parameter '{symbol}' mentioned in chapter"


def refine_parameter_line_numbers(
    chapter_file: Path,
    full_document_text: str,
    file_path: str,
    article_id: str,
    max_retries: int = 2,
) -> tuple[int, int, list[str]]:
    """
    Refine parameters at line 1 using DSPy agent.

    Args:
        chapter_file: Path to chapter_N.json
        full_document_text: Full document with line numbers (NNN: content)
        file_path: Source markdown file path
        article_id: Document ID
        max_retries: Max retries per parameter

    Returns:
        Tuple of (updated_count, failed_count, error_messages)
    """
    from mathster.parameter_extraction.dspy_components.line_finder import ParameterLineFinder
    from mathster.parameter_extraction.text_processing.analysis import _get_symbol_variants

    # Load chapter data
    try:
        with open(chapter_file, encoding="utf-8") as f:
            chapter_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {chapter_file}: {e}")
        return 0, 0, [str(e)]

    # Find parameters at line 1 (need refinement)
    params_at_line_1 = []
    param_indices = {}  # Track original index for updating

    for idx, param in enumerate(chapter_data.get("parameters", [])):
        line_range = param.get("source", {}).get("line_range", {}).get("lines", [[1, 1]])
        if line_range[0][0] == 1:
            params_at_line_1.append(param)
            param_indices[param.get("symbol")] = idx

    if not params_at_line_1:
        logger.info(f"No parameters need refinement in {chapter_file.name}")
        return 0, 0, []

    logger.info(f"Refining {len(params_at_line_1)} parameters at line 1 in {chapter_file.name}")

    # Initialize DSPy agent
    agent = ParameterLineFinder()

    updated = 0
    failed = 0
    errors = []

    # Refine each parameter
    for param in params_at_line_1:
        symbol = param.get("symbol", "")
        logger.info(f"  Searching for: {symbol}")

        # Get symbol variants
        variants = _get_symbol_variants(symbol)

        # Get usage context
        context = extract_usage_context(symbol, chapter_data)

        # Call DSPy agent with retry logic
        result = None
        for attempt in range(max_retries):
            try:
                result = agent(
                    parameter_symbol=symbol,
                    symbol_variants=variants,
                    document_with_lines=full_document_text,
                    context_from_entity=context,
                )

                # Validate line numbers
                if not isinstance(result["line_start"], int) or not isinstance(result["line_end"], int):
                    raise ValueError(f"Invalid line numbers: {result}")

                if result["line_start"] < 1 or result["line_end"] < result["line_start"]:
                    raise ValueError(f"Invalid line range: {result['line_start']}-{result['line_end']}")

                # Success
                break

            except Exception as e:
                logger.warning(f"    Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    errors.append(f"{symbol}: {e}")
                    result = None

        # Process result
        if result and result["confidence"] in ["high", "medium"]:
            # Update parameter in chapter_data
            idx = param_indices[symbol]
            chapter_data["parameters"][idx]["source"]["line_range"] = {
                "lines": [[result["line_start"], result["line_end"]]]
            }

            # Add DSPy metadata
            chapter_data["parameters"][idx]["_dspy_refined"] = True
            chapter_data["parameters"][idx]["_dspy_confidence"] = result["confidence"]
            chapter_data["parameters"][idx]["_dspy_reasoning"] = result["reasoning"]

            logger.info(
                f"    ✓ Found at lines {result['line_start']}-{result['line_end']} "
                f"(confidence: {result['confidence']})"
            )
            updated += 1

        elif result and result["confidence"] == "low":
            logger.warning(f"    ⚠ Low confidence, keeping line 1")
            logger.warning(f"      Reasoning: {result['reasoning']}")
            failed += 1

        else:
            logger.error(f"    ✗ Agent failed after {max_retries} retries")
            failed += 1

    # Save updated chapter
    try:
        with open(chapter_file, "w", encoding="utf-8") as f:
            json.dump(chapter_data, f, indent=2)
        logger.info(f"✓ Saved refined parameters to {chapter_file.name}")
    except Exception as e:
        error_msg = f"Failed to save {chapter_file}: {e}"
        logger.error(error_msg)
        errors.append(error_msg)

    return updated, failed, errors
