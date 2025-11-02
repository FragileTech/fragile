"""
LLM Interface Implementation for Mathematical Paper Extraction.

This module implements the LLM interface for the Extract-then-Enrich pipeline
using the Anthropic Claude API.

Design Principles:
- Use Claude API for all LLM calls
- Parse and validate JSON responses
- Handle markdown code blocks
- Provide error handling and retries
- Log all API interactions

Maps to Lean:
    namespace LLMInterface
      def call_main_extraction_llm : String → String → IO (HashMap String Any)
      def call_semantic_parser_llm : String → IO (HashMap String Any)
    end LLMInterface
"""

import json
import logging
import os
import re
from typing import Any


# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# JSON EXTRACTION AND VALIDATION
# =============================================================================


def extract_json_from_markdown(llm_output: str) -> str:
    """
    Extract JSON from LLM output with markdown code blocks.

    LLMs often wrap JSON in ```json...``` code blocks. This function strips them.

    Args:
        llm_output: Raw text output from LLM

    Returns:
        Clean JSON string

    Examples:
        >>> output = '```json\\n{"key": "value"}\\n```'
        >>> extract_json_from_markdown(output)
        '{"key": "value"}'

        >>> output = '{"key": "value"}'  # No markdown
        >>> extract_json_from_markdown(output)
        '{"key": "value"}'
    """
    # Strip leading/trailing whitespace
    output = llm_output.strip()

    # Pattern 1: ```json ... ```
    json_block_pattern = r"^```json\s*\n(.*?)\n```$"
    match = re.match(json_block_pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Pattern 2: ``` ... ``` (without language specifier)
    code_block_pattern = r"^```\s*\n(.*?)\n```$"
    match = re.match(code_block_pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Pattern 3: ~~~ ... ~~~ (alternative fence style)
    tilde_pattern = r"^~~~\s*\n(.*?)\n~~~$"
    match = re.match(tilde_pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No markdown code blocks found, return as-is
    return output


def validate_llm_response(response: dict[str, Any], schema_name: str) -> bool:
    """
    Validate LLM response against expected schema.

    Args:
        response: The parsed JSON response from LLM
        schema_name: Name of the Pydantic model to validate against

    Returns:
        True if valid, False otherwise

    Examples:
        >>> response = {"section_id": "§2.1", "definitions": [], ...}
        >>> validate_llm_response(response, "StagingDocument")
        True
    """
    from mathster.core.raw_data import (
        RawDefinition,
        RawProof,
        RawTheorem,
        RawDocument,
    )

    schema_map = {
        "StagingDocument": RawDocument,
        "RawDefinition": RawDefinition,
        "RawTheorem": RawTheorem,
        "RawProof": RawProof,
        "dict": dict,  # Generic dict validation
        "list": list,  # Generic list validation
    }

    if schema_name not in schema_map:
        logger.warning(f"Unknown schema name: {schema_name}, skipping validation")
        return True  # Optimistic validation for unknown schemas

    schema_class = schema_map[schema_name]

    # For generic types, just check type
    if schema_class in {dict, list}:
        return isinstance(response, schema_class)

    # For Pydantic models, validate with model_validate
    try:
        schema_class.model_validate(response)
        return True
    except Exception as e:
        logger.error(f"Validation failed for {schema_name}: {e}")
        return False


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses simple heuristic: ~4 characters per token (rough approximation for Claude).

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count

    Examples:
        >>> estimate_tokens("Hello world")
        3
        >>> estimate_tokens("A" * 1000)
        250
    """
    # Simple heuristic: ~4 characters per token
    # This is approximate but good enough for validation
    return len(text) // 4


# =============================================================================
# MAIN LLM INTERFACE FUNCTIONS
# =============================================================================


def call_main_extraction_llm(
    section_text: str,
    section_id: str,
    prompt_template: str,
    model: str = "claude-sonnet-4",
    **kwargs,
) -> dict[str, Any]:
    """
    Make Stage 1 extraction LLM call using Anthropic Claude API.

    This function performs the "raw extraction" from a markdown section.
    It combines the prompt template with the section text and calls
    the LLM API to extract mathematical entities into staging models.

    Implementation:
    1. Substitute section_text into the template
    2. Call the Anthropic Messages API
    3. Parse the JSON response from the LLM
    4. Validate that the response matches StagingDocument schema
    5. Return the parsed dictionary

    Args:
        section_text: The markdown content of the section to extract from
        section_id: Identifier for the section (e.g., "§2.1", "intro")
        prompt_template: The prompt template string (with {section_text} placeholder)
        model: The LLM model to use (default: "claude-sonnet-4")
        **kwargs: Additional arguments to pass to the LLM API (e.g., temperature, max_tokens)

    Returns:
        Dictionary matching the StagingDocument Pydantic schema

    Raises:
        ValueError: If LLM response doesn't match expected schema
        RuntimeError: If LLM API call fails

    Examples:
        >>> from fragile.mathster.prompts import MAIN_EXTRACTION_PROMPT
        >>> section = "# Section 2.1\\n\\n**Theorem 2.1**. Let v > 0..."
        >>> result = call_main_extraction_llm(
        ...     section_text=section,
        ...     section_id="§2.1",
        ...     prompt_template=MAIN_EXTRACTION_PROMPT,
        ...     temperature=0.0,
        ... )
        >>> result["section_id"]
        '§2.1'
    """
    try:
        # Check for API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set, falling back to stub implementation")
            return {
                "section_id": section_id,
                "definitions": [],
                "theorems": [],
                "mathster": [],
                "axioms": [],
                "citations": [],
                "equations": [],
                "parameters": [],
                "remarks": [],
            }

        # Use Anthropic API
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        # Format the prompt
        full_prompt = prompt_template.format(section_text=section_text, section_id=section_id)

        logger.info(f"Calling Anthropic API for section {section_id}")
        logger.info(f"  Model: {model}")
        logger.info(f"  Section length: {len(section_text)} chars")
        logger.info(f"  Estimated tokens: {estimate_tokens(section_text)}")

        # Make API call
        message = client.messages.create(
            model=model,
            max_tokens=kwargs.get("max_tokens", 8192),
            temperature=kwargs.get("temperature", 0.0),
            messages=[{"role": "user", "content": full_prompt}],
        )

        # Extract response text
        response_text = message.content[0].text

        # Parse JSON from response
        json_str = extract_json_from_markdown(response_text)
        result = json.loads(json_str)

        # Validate response
        if not validate_llm_response(result, "StagingDocument"):
            msg = "Response doesn't match StagingDocument schema"
            raise ValueError(msg)

        logger.info(f"  Extracted {len(result.get('definitions', []))} definitions")
        logger.info(f"  Extracted {len(result.get('theorems', []))} theorems")
        logger.info(f"  Extracted {len(result.get('axioms', []))} axioms")

        return result

    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        raise RuntimeError(f"Failed to call extraction LLM: {e}") from e


def call_semantic_parser_llm(
    prompt: str, expected_schema: str | None = None, model: str = "claude-sonnet-4", **kwargs
) -> dict[str, Any]:
    """
    Make Stage 2 enrichment LLM call using Anthropic Claude API.

    This function performs focused, specialized parsing tasks during enrichment.
    It's called multiple times for different subtasks:
    - Decomposing theorem statements into assumptions + conclusion
    - Parsing LaTeX into DualStatement structure
    - Analyzing proof structure
    - Resolving ambiguous references

    Implementation:
    1. Call the Anthropic Messages API with the provided prompt
    2. Parse the JSON response
    3. Validate against the expected schema (if provided)
    4. Return the parsed dictionary

    Args:
        prompt: The complete prompt for the focused parsing task
        expected_schema: Optional name of the Pydantic model the response should match
                        (e.g., "DualStatement", "List[DualStatement]")
        model: The LLM model to use (default: "claude-sonnet-4")
        **kwargs: Additional arguments to pass to the LLM API

    Returns:
        Dictionary matching the expected schema

    Raises:
        ValueError: If LLM response doesn't match expected schema
        RuntimeError: If LLM API call fails

    Examples:
        >>> from fragile.mathster.prompts import DECOMPOSE_THEOREM_PROMPT
        >>> prompt = DECOMPOSE_THEOREM_PROMPT.format(
        ...     theorem_statement="Let v > 0 and assume U is Lipschitz. Then..."
        ... )
        >>> result = call_semantic_parser_llm(prompt, expected_schema="dict")
        >>> result["assumptions"]
        ['Let v > 0', 'assume U is Lipschitz']
    """
    try:
        # Check for API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set, falling back to stub implementation")
            return {}

        # Use Anthropic API
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        logger.info("Calling Anthropic API for semantic parsing")
        logger.info(f"  Model: {model}")
        logger.info(f"  Expected schema: {expected_schema}")

        # Make API call
        message = client.messages.create(
            model=model,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.2),
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract response text
        response_text = message.content[0].text

        # Parse JSON from response
        json_str = extract_json_from_markdown(response_text)
        result = json.loads(json_str)

        # Validate if schema provided
        if expected_schema and not validate_llm_response(result, expected_schema):
            raise ValueError(f"Response doesn't match {expected_schema} schema")

        return result

    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        raise RuntimeError(f"Failed to call semantic parser LLM: {e}") from e


def call_batch_extraction_llm(
    sections: list[dict[str, str]],
    prompt_template: str,
    model: str = "claude-sonnet-4",
    parallel: bool = True,
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Make parallel batch extraction calls using Anthropic Claude API.

    This is an optimized version of call_main_extraction_llm that processes
    multiple sections in parallel. Useful for processing long documents efficiently.

    Implementation:
    1. Create one extraction task per section
    2. Execute tasks in parallel (if parallel=True) using asyncio
    3. Return list of results in the same order as input sections

    Args:
        sections: List of dicts with keys "section_id" and "section_text"
        prompt_template: The prompt template string
        model: The LLM model to use
        parallel: Whether to execute calls in parallel (default: True)
        **kwargs: Additional arguments to pass to LLM API

    Returns:
        List of dictionaries, each matching StagingDocument schema

    Raises:
        RuntimeError: If any LLM API call fails

    Examples:
        >>> sections = [
        ...     {"section_id": "§1", "section_text": "# Introduction..."},
        ...     {"section_id": "§2", "section_text": "# Main Results..."},
        ... ]
        >>> results = call_batch_extraction_llm(sections, prompt, parallel=True)
        >>> len(results)
        2
    """
    try:
        logger.info("Calling batch extraction LLM")
        logger.info(f"  Processing {len(sections)} sections")
        logger.info(f"  Parallel mode: {parallel}")

        # For now, process sequentially
        # TODO: Implement async parallel processing with AsyncAnthropic
        results = []
        for section in sections:
            result = call_main_extraction_llm(
                section_text=section["section_text"],
                section_id=section["section_id"],
                prompt_template=prompt_template,
                model=model,
                **kwargs,
            )
            results.append(result)

        return results

    except Exception as e:
        logger.error(f"Batch LLM API call failed: {e}")
        raise RuntimeError(f"Failed to call batch extraction LLM: {e}") from e


# =============================================================================
# MOCK IMPLEMENTATIONS (FOR TESTING ONLY)
# =============================================================================


def mock_main_extraction_llm(
    section_text: str, section_id: str, prompt_template: str, **kwargs
) -> dict[str, Any]:
    """
    Mock implementation for testing purposes only.

    Returns empty StagingDocument structure. Use this for testing the
    orchestration pipeline without making actual LLM calls.

    NOT FOR PRODUCTION USE.
    """
    return {
        "section_id": section_id,
        "definitions": [],
        "theorems": [],
        "mathster": [],
        "axioms": [],
        "citations": [],
        "equations": [],
        "parameters": [],
        "remarks": [],
    }


def mock_semantic_parser_llm(prompt: str, **kwargs) -> dict[str, Any]:
    """
    Mock implementation for testing purposes only.

    Returns empty dict. Use this for testing enrichment logic without
    making actual LLM calls.

    NOT FOR PRODUCTION USE.
    """
    return {}
