"""
DSPy configuration utilities.

Provides functions for configuring DSPy models and settings for the
extraction and improvement workflows.
"""

import os

from dotenv import load_dotenv
import dspy


def configure_dspy(
    model: str = "gemini/gemini-2.0-flash-exp",
    temperature: float = 0.0,
    max_tokens: int = 8192,
    verbose: bool = False,
) -> None:
    """
    Configure DSPy with the specified model and settings.

    Args:
        model: Model identifier (e.g., "gemini/gemini-2.0-flash-exp", "anthropic/claude-haiku-4-5")
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum tokens in response
        verbose: Whether to print verbose output

    Example:
        >>> configure_dspy("gemini/gemini-flash-lite-latest")
        >>> # DSPy is now configured to use Gemini Flash Lite
    """
    # Load environment variables
    load_dotenv()

    # Configure LiteLLM-based DSPy model
    lm = dspy.LM(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=os.getenv("ANTHROPIC_API_KEY") if "anthropic" in model else None,
    )

    dspy.configure(lm=lm)

    if verbose:
        print(f"✓ DSPy configured with model: {model}")
        print(f"  Temperature: {temperature}")
        print(f"  Max tokens: {max_tokens}")
